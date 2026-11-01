"""The experiment registry (docs/concepts/runs-and-coordinates.md): runs and units, versions,
slots, decision scopes and holdout checks, persisted in one SQLite file next to the store.

Group and slot rules:
    counter key = H(experiment, phase, granularity, replicate values, version[, case])
    slot        = the smallest s not yet used under the counter key (a decision scope: the first
                  member's slot if this member has not used it, else its own smallest unused one)
    group       = H(experiment, phase, granularity, replicate values[, case], slot)
Compare coordinates and the version are left out of the group, so compared runs share; replicate
values and cases are in it, so replicates and cases are separate. A run outside any experiment gets
a fresh group (O1).

Versions: a run is an unchanged rerun of an earlier run of the experiment when its compare
values are equal and every project file that run read still has the content it had; otherwise it
is a new version. Matching and the slot claim happen in one write transaction, so launches at the
same moment see each other. The safe direction is a rerun (new samples), so whatever is not known
counts as unchanged: a run still running (its read set is not known yet), a finished run that read
no project file, and a file whose content cannot be read now (deleted, or not visible where the
digests are taken). A run whose read set was lost (status unknown: the child failed) matches only
when no finished run with a known read set is more recent; otherwise it is ignored. A run left
running by a launcher that died on this host is marked unknown when it is next matched.
`process_run` records why it matched: 'unchanged', 'running' or 'unknown'.
"""
import hashlib
import hmac
import json
import os
import socket
import sqlite3
import time
import uuid

from .coords import Classes, PHASES

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, parent TEXT, experiment TEXT, phase TEXT, recipe TEXT,
    overrides TEXT, granularity TEXT NOT NULL, coords TEXT NOT NULL, version TEXT, counter TEXT,
    slot INTEGER, grp TEXT NOT NULL, identity TEXT UNIQUE, reads TEXT, created REAL NOT NULL,
    rule TEXT NOT NULL, status TEXT NOT NULL, rerun_of TEXT);
CREATE TABLE IF NOT EXISTS slots (counter TEXT NOT NULL, slot INTEGER NOT NULL,
    run_id TEXT NOT NULL, PRIMARY KEY (counter, slot));
CREATE TABLE IF NOT EXISTS decisions (decision TEXT NOT NULL, kase TEXT NOT NULL,
    slot INTEGER NOT NULL, PRIMARY KEY (decision, kase));
CREATE TABLE IF NOT EXISTS cases (experiment TEXT NOT NULL, phase TEXT NOT NULL,
    kase TEXT NOT NULL, PRIMARY KEY (experiment, phase, kase));
CREATE TABLE IF NOT EXISTS responses (id TEXT PRIMARY KEY, context TEXT NOT NULL,
    response TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS owners (run_id TEXT PRIMARY KEY, host TEXT NOT NULL,
    pid INTEGER NOT NULL);
"""


class HoldoutOverlap(ValueError):
    """A holdout unit's case was already used in the search phase of the experiment (W5a)."""


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def h(*parts, n=24):
    return hashlib.sha256(canon(list(parts)).encode('utf-8')).hexdigest()[:n]


BIG = 256 << 20            # files larger than this are fingerprinted by size and mtime only


def file_digest(path):
    """sha256 of a file's content (by size and mtime above BIG), or None if it cannot be read."""
    try:
        st = os.stat(path)
        if st.st_size > BIG:
            return 'stat:%d:%d' % (st.st_size, st.st_mtime_ns)
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                hasher.update(block)
        return hasher.hexdigest()
    except OSError:
        return None


class Registry:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript(SCHEMA)
        row = self.db.execute("SELECT value FROM meta WHERE name='secret'").fetchone()
        if row is None:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('secret', ?)", (os.urandom(32).hex(),))
            row = self.db.execute("SELECT value FROM meta WHERE name='secret'").fetchone()
        self._secret = bytes.fromhex(row['value'])

    def close(self):
        self.db.close()

    # -- tokens ------------------------------------------------------------------------------

    def token(self, run_id):
        """A run token: the run id plus an HMAC under the registry's secret (not forgeable)."""
        mac = hmac.new(self._secret, run_id.encode(), hashlib.sha256).hexdigest()[:20]
        return '%s.%s' % (run_id, mac)

    def run_of(self, token):
        run_id, _, mac = str(token).rpartition('.')
        if not run_id or not hmac.compare_digest(self.token(run_id), str(token)):
            return None
        return self.get(run_id)

    # -- records -----------------------------------------------------------------------------

    def get(self, run_id):
        row = self.db.execute('SELECT * FROM runs WHERE run_id=?', (run_id,)).fetchone()
        return None if row is None else self._record(row)

    def find(self, run_id):
        """The run with this id, or the only one whose id starts with it."""
        rec = self.get(run_id)
        if rec is not None:
            return rec
        hits = [r for r in self.runs() if r['run_id'].startswith(run_id)]
        return hits[0] if len(hits) == 1 else None

    def runs(self, experiment=None):
        q, args = 'SELECT * FROM runs', ()
        if experiment is not None:
            q, args = q + ' WHERE experiment=?', (experiment,)
        return [self._record(r) for r in self.db.execute(q + ' ORDER BY created', args)]

    @staticmethod
    def _record(row):
        rec = dict(row)
        for k in ('coords', 'overrides', 'reads'):
            rec[k] = json.loads(rec[k]) if rec[k] else ({} if k != 'reads' else None)
        return rec

    def count_units(self, parent):
        return self.db.execute('SELECT COUNT(*) FROM runs WHERE parent=?', (parent,)).fetchone()[0]

    def classes(self, rec):
        return Classes(rec['recipe'] or 'dev', rec['overrides'])

    # -- versions ----------------------------------------------------------------------------

    def match_version(self, experiment, compare, digest_now=file_digest):
        """(version, run it reruns or None) for a new process run of `experiment` (read only;
        process_run does the same inside its write transaction)."""
        return self._match(experiment, compare, digest_now)[:2]

    def version_paths(self, experiment, compare):
        """The files whose content decides the version of a new run of `experiment` with these
        compare values: the union of the read sets of its earlier process runs (a launcher on
        another host digests them itself and passes the digests to process_run)."""
        out = set()
        for rec in self.runs(experiment):
            if rec['granularity'] == 'process' and rec['reads'] and \
                    self.classes(rec).split(rec['coords'])['compare'] == compare:
                out |= set(rec['reads'])
        return sorted(out)

    def _owner_dead(self, run_id):
        row = self.db.execute('SELECT host, pid FROM owners WHERE run_id=?', (run_id,)).fetchone()
        if row is None or row['host'] != socket.gethostname():
            return False
        try:
            os.kill(int(row['pid']), 0)
            return False
        except ProcessLookupError:
            return True
        except (PermissionError, OverflowError, ValueError):
            return False

    def _match(self, experiment, compare, digest_now):
        memo = {}

        def dig(p):
            if p not in memo:
                memo[p] = digest_now(p)
            return memo[p]
        fallback = None
        known_seen = False
        for rec in reversed(self.runs(experiment)):
            if rec['granularity'] != 'process' or rec['version'] is None:
                continue
            if self.classes(rec).split(rec['coords'])['compare'] != compare:
                continue
            if rec['status'] == 'running' and self._owner_dead(rec['run_id']):
                self.db.execute("UPDATE runs SET status='unknown' WHERE run_id=?",
                                (rec['run_id'],))
                rec['status'] = 'unknown'
            if rec['status'] == 'running':
                return rec['version'], rec['run_id'], 'running'
            if rec['status'] == 'unknown':
                if not known_seen and fallback is None:
                    fallback = rec
                continue
            known_seen = True
            if all(dig(p) in (d, None) for p, d in (rec['reads'] or {}).items()):
                return rec['version'], rec['run_id'], 'unchanged'
        if fallback is not None:
            return fallback['version'], fallback['run_id'], 'unknown'
        return 'v-' + uuid.uuid4().hex[:16], None, None

    # -- slots -------------------------------------------------------------------------------

    def _used(self, counter):
        return {r[0] for r in self.db.execute('SELECT slot FROM slots WHERE counter=?', (counter,))}

    def _take_slot(self, counter, run_id, decision=None, kase=''):
        """Inside a write transaction: claim a slot under `counter`."""
        used = self._used(counter)
        smallest = 1
        while smallest in used:
            smallest += 1
        slot = smallest
        if decision is not None:
            row = self.db.execute('SELECT slot FROM decisions WHERE decision=? AND kase=?',
                                  (decision, kase)).fetchone()
            if row is None:
                self.db.execute('INSERT INTO decisions VALUES (?, ?, ?)', (decision, kase, slot))
            elif row['slot'] not in used:
                slot = row['slot']
        self.db.execute('INSERT INTO slots VALUES (?, ?, ?)', (counter, slot, run_id))
        return slot

    # -- runs and units ----------------------------------------------------------------------

    def process_run(self, experiment=None, recipe='dev', phase='search', coords=None,
                    overrides=None, version=None, run_id=None, digest_now=file_digest,
                    owner=None):
        """Register a launched process as one run, status 'running'. Without `version`,
        the version is matched inside the same transaction as the slot claim."""
        if phase not in PHASES:
            raise ValueError('phase is one of %s' % ', '.join(PHASES))
        coords = dict(coords or {})
        cls = Classes(recipe, overrides)
        split = cls.split(coords)
        run_id = run_id or 'r-' + uuid.uuid4().hex[:16]
        base = dict(run_id=run_id, parent=None, experiment=experiment, phase=phase, recipe=recipe,
                    overrides=cls.overrides, granularity='process', coords=coords, identity=None,
                    reads=None, created=time.time(), status='running', rerun_of=None)
        if experiment is None:
            rec = dict(base, version=None, counter=None, slot=None,
                       grp='g-' + uuid.uuid4().hex[:20], rule='no experiment: fresh group (O1)')
            self._insert(rec)
            self._own(run_id, owner)
            return rec
        rep = split['replicate']
        kase = canon(split['case']) if split['case'] else ''
        self.db.execute('BEGIN IMMEDIATE')
        try:
            rerun_of = match = None
            if version is None:
                version, rerun_of, match = self._match(experiment, split['compare'], digest_now)
            counter = 'c-' + h(experiment, phase, 'process', rep, version, kase)
            slot = self._take_slot(counter, run_id)
            grp = 'g-' + h(experiment, phase, 'process', rep, kase, slot)
            rec = dict(base, version=version, counter=counter, slot=slot, grp=grp,
                       rerun_of=rerun_of,
                       rule='%s: version %s, slot %d' % (recipe, version, slot))
            self._insert(rec)
            self._own(run_id, owner)
            self.db.execute('COMMIT')
            rec['match'] = match
        except BaseException:
            self.db.execute('ROLLBACK')
            raise
        return rec

    def _own(self, run_id, owner):
        if owner:
            self.db.execute('INSERT OR REPLACE INTO owners VALUES (?, ?, ?)',
                            (run_id, str(owner.get('host')), int(owner.get('pid') or 0)))

    def finish(self, run_id, reads, ok):
        """A process run ended: its read set, and status done (ok) or unknown (reads lost)."""
        self.db.execute('UPDATE runs SET reads=?, status=? WHERE run_id=?',
                        (canon(reads or {}), 'done' if ok else 'unknown', run_id))

    def unit(self, parent, coords, ordinal=1, decision=None, phase=None):
        """The unit of `parent` (a process run record) with `coords` (merged over the parent's),
        created once: the same (parent, coords, ordinal, phase) is the same unit, also after a
        resume. `phase` overrides the parent's (an adapter's holdout units in a search run)."""
        phase = phase or parent['phase']
        if phase not in PHASES:
            raise ValueError('phase is one of %s' % ', '.join(PHASES))
        merged = dict(parent['coords'])
        merged.update(coords or {})
        identity = 'i-' + h(parent['run_id'], merged, int(ordinal), phase)
        row = self.db.execute('SELECT * FROM runs WHERE identity=?', (identity,)).fetchone()
        if row is not None:
            return self._record(row)
        cls = self.classes(parent)
        split = cls.split(merged)
        run_id = 'u-' + identity[2:22]
        exp = parent['experiment']
        kase = canon(split['case']) if split['case'] else ''
        if exp is None:
            rec = dict(run_id=run_id, parent=parent['run_id'], experiment=None, phase=phase,
                       recipe=parent['recipe'], overrides=parent['overrides'], granularity='unit',
                       coords=merged, version=None, counter=None, slot=None,
                       grp='g-' + uuid.uuid4().hex[:20], identity=identity, reads=None,
                       created=time.time(), rule='no experiment: fresh group (O1)',
                       status='unit', rerun_of=None)
            self._insert(rec)
            return rec
        version = 'v-' + h(parent['version'], split['compare'], n=16)
        rep = split['replicate']
        counter = 'c-' + h(exp, phase, 'unit', rep, version, kase)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT * FROM runs WHERE identity=?', (identity,)).fetchone()
            if row is not None:                 # created meanwhile by another process
                self.db.execute('COMMIT')
                return self._record(row)
            if kase and phase == 'holdout' and self.db.execute(
                    "SELECT 1 FROM cases WHERE experiment=? AND phase='search' AND kase=?",
                    (exp, kase)).fetchone():
                raise HoldoutOverlap('holdout case %s was used in the search phase of %s'
                                     % (kase, exp))
            if kase and phase == 'search':
                self.db.execute('INSERT OR IGNORE INTO cases VALUES (?, ?, ?)', (exp, phase, kase))
            scope = None if decision is None else 'd-' + h(exp, phase, parent['run_id'], decision)
            slot = self._take_slot(counter, run_id, scope, kase)
            grp = 'g-' + h(exp, phase, 'unit', rep, kase, slot)
            rule = '%s unit: version %s, slot %d%s' % (parent['recipe'], version, slot,
                                                        ', decision %s' % decision if decision else '')
            rec = dict(run_id=run_id, parent=parent['run_id'], experiment=exp, phase=phase,
                       recipe=parent['recipe'], overrides=parent['overrides'], granularity='unit',
                       coords=merged, version=version, counter=counter, slot=slot, grp=grp,
                       identity=identity, reads=None, created=time.time(), rule=rule,
                       status='unit', rerun_of=None)
            self._insert(rec)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise
        return rec

    # -- Responses API chains -------------------------------------------------------------------

    def put_response(self, rid, context, response):
        """Store an issued response: its chat context (input without instructions, plus the
        output turn) and the response object. Ids derive from sample addresses, so a re-store
        of the same id stores the same content."""
        self.db.execute('INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?)',
                        (rid, canon(context), canon(response), time.time()))

    def response_context(self, rid):
        row = self.db.execute('SELECT context FROM responses WHERE id=?', (rid,)).fetchone()
        return None if row is None else json.loads(row[0])

    def response(self, rid):
        row = self.db.execute('SELECT response FROM responses WHERE id=?', (rid,)).fetchone()
        return None if row is None else json.loads(row[0])

    def route(self, rec, role, rule):
        """The group of `role`'s calls in run `rec` under a plan route."""
        if rule in (None, 'run', 'fresh') or rec['experiment'] is None:
            return rec['grp']                   # 'fresh' calls are uncached (proxy)
        if rule == 'global':
            return 'g-' + h(rec['experiment'], rec['phase'], 'role', role)
        if rule.startswith('per:'):
            name = rule[4:]
            return 'g-' + h(rec['experiment'], rec['phase'], 'role', role, name,
                            rec['coords'].get(name))
        raise ValueError('unknown route %r' % rule)

    def record_reads(self, run_id, reads):
        self.finish(run_id, reads, True)

    def _insert(self, rec):
        cols = ('run_id', 'parent', 'experiment', 'phase', 'recipe', 'overrides', 'granularity',
                'coords', 'version', 'counter', 'slot', 'grp', 'identity', 'reads', 'created',
                'rule', 'status', 'rerun_of')
        vals = [canon(rec[c]) if c in ('coords', 'overrides') else
                (canon(rec[c]) if c == 'reads' and rec[c] is not None else rec[c]) for c in cols]
        self.db.execute('INSERT INTO runs (%s) VALUES (%s)' % (', '.join(cols),
                                                                ', '.join('?' * len(cols))), vals)


def check_route(rule):
    """A plan route: run, global, fresh or per:<coordinate>."""
    if rule in ('run', 'global', 'fresh') or (isinstance(rule, str) and rule.startswith('per:')
                                              and len(rule) > 4):
        return rule
    raise ValueError('unknown route %r (run, global, fresh or per:<coordinate>)' % (rule,))


def seed_group(group, seed):
    """The group of seeded requests: the seed is part of the stream's identity."""
    return 'g-' + h(group, 'seed', seed)
