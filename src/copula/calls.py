"""The call log of a Copula server and the reasons behind each generated call (DESIGN.md §10.4).

The proxy writes one row per answered request occurrence to .copula/calls.db: its run (process run
and unit), group, node, request digest and statuses. Request bodies are kept once per digest, so
`copula explain` can say what changed. The log is diagnostic only: nothing reads it to decide what
a request returns, so it cannot affect a sample (P).

Why a call was generated rather than reused, in order of precedence:
    uncached       its node is uncached (stateful request, server-side tool, route fresh)
    after_error    earlier runs sent this request, but every earlier call of it failed
    new_request    no earlier run sent this request; the closest earlier request and where it
                   differs are shown
    model          earlier runs sent it to another revision of the model (weights, server
                   version, sampling table)
    more_repeats   earlier runs sent it fewer times in this group (occurrence k > stored), or
                   had fewer identical units
    other_group    earlier runs sent it in another group, because a coordinate differs:
                   experiment, phase, replicate value, case, seed, route, or the slot -- an
                   unchanged rerun (a new slot by design), concurrent identical units numbered by
                   arrival, or the slot of another run
"""
import collections
import difflib
import json
import os
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS bodies (digest TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS calls (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, t REAL NOT NULL, exp TEXT, proc TEXT NOT NULL,
    run TEXT NOT NULL, ordinal INTEGER, grp TEXT NOT NULL, node TEXT NOT NULL, model TEXT,
    body TEXT NOT NULL, n INTEGER NOT NULL, seed TEXT, role TEXT, statuses TEXT NOT NULL,
    ranks TEXT NOT NULL, reason TEXT, proto TEXT, reread INTEGER NOT NULL, error TEXT,
    rev TEXT);
CREATE INDEX IF NOT EXISTS calls_by_body ON calls (body);
CREATE INDEX IF NOT EXISTS calls_by_proc ON calls (proc);
CREATE INDEX IF NOT EXISTS calls_by_exp ON calls (exp, model);
"""
COLS = ('t', 'exp', 'proc', 'run', 'ordinal', 'grp', 'node', 'model', 'body', 'n', 'seed', 'role',
        'statuses', 'ranks', 'reason', 'proto', 'reread', 'error', 'rev')
REUSED = ('hit', 'own')
LABELS = {
    'uncached': 'uncached node',
    'after_error': 'earlier calls failed',
    'new_request': 'new request',
    'model': 'other model revision',
    'more_repeats': 'more repeats than stored',
    'rerun': 'unchanged rerun (new slot)',
    'arrival_order': 'arrival order of identical units',
    'replicate': 'other replicate',
    'case': 'other case',
    'phase': 'other phase',
    'seed': 'other seed',
    'route': 'other route',
    'experiment': 'other experiment',
    'granularity': 'process run vs unit',
    'slot': 'other slot',
    'unexplained': 'unexplained',
}


def _connect(path):
    db = sqlite3.connect(path, timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=NORMAL')
    db.executescript(SCHEMA)
    cols = {r[1] for r in db.execute('PRAGMA table_info(calls)')}
    if 'rev' not in cols:                       # a log written before the column existed
        db.execute('ALTER TABLE calls ADD COLUMN rev TEXT')
    return db


class CallLog:
    """Buffered writer (one asyncio thread); rows reach the file at least every `every` s."""

    def __init__(self, path, every=1.0, batch=256):
        self.path = path
        self.db = _connect(path)
        self.every, self.batch = every, batch
        self._rows, self._bodies = [], {}
        self._last = time.time()

    def add(self, row, body_digest, body):
        self._rows.append(tuple(row.get(c) for c in COLS))
        if body_digest not in self._bodies:
            self._bodies[body_digest] = body
        if len(self._rows) >= self.batch or time.time() - self._last >= self.every:
            self.flush()

    def flush(self):
        self._last = time.time()
        if not self._rows and not self._bodies:
            return
        rows, bodies = self._rows, self._bodies
        self._rows, self._bodies = [], {}
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.db.executemany('INSERT OR IGNORE INTO bodies VALUES (?, ?)',
                                [(d, json.dumps(b, sort_keys=True, ensure_ascii=False))
                                 for d, b in bodies.items()])
            self.db.executemany('INSERT INTO calls (%s) VALUES (%s)' % (
                ', '.join(COLS), ', '.join('?' * len(COLS))), rows)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def close(self):
        try:
            self.flush()
        finally:
            self.db.close()


# -- diagnosis -------------------------------------------------------------------------------------

def _text(msg):
    c = msg.get('content') if isinstance(msg, dict) else msg
    if isinstance(c, str):
        out = c
    else:
        out = json.dumps(c, sort_keys=True, ensure_ascii=False)
    if isinstance(msg, dict) and msg.get('tool_calls'):
        out += ' ' + json.dumps(msg['tool_calls'], sort_keys=True, ensure_ascii=False)
    return out


def _snip(a, b, width=36):
    """('...old...', '...new...') around the first differing character."""
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    lo = max(0, i - width // 3)

    def cut(s):
        piece = s[lo:lo + width]
        return ('...' if lo else '') + piece + ('...' if lo + width < len(s) else '')
    return cut(a), cut(b)


def body_diff(old, new):
    """Where request `new` differs from `old`: a short human string, and a closeness score."""
    om, nm = old.get('messages') or [], new.get('messages') or []
    i = 0
    while i < min(len(om), len(nm)) and om[i] == nm[i]:
        i += 1
    parts = []
    params = sorted(k for k in set(old.get('params') or {}) | set(new.get('params') or {})
                    if (old.get('params') or {}).get(k) != (new.get('params') or {}).get(k))
    if old.get('model') != new.get('model'):
        parts.append('model %s -> %s' % (old.get('model'), new.get('model')))
    if params:
        parts.append('parameters ' + ', '.join(
            '%s %s -> %s' % (k, json.dumps((old.get('params') or {}).get(k)),
                             json.dumps((new.get('params') or {}).get(k))) for k in params[:4]))
    if i < max(len(om), len(nm)):
        if i >= len(om):
            parts.append('%d more message(s) from messages[%d]' % (len(nm) - len(om), i))
        elif i >= len(nm):
            parts.append('%d fewer message(s) from messages[%d]' % (len(om) - len(nm), i))
        else:
            role = nm[i].get('role') if isinstance(nm[i], dict) else '?'
            a, b = _snip(_text(om[i]), _text(nm[i]))
            parts.append('messages[%d] (%s): %r -> %r' % (i, role, a, b))
    similar = 0.0
    if i < min(len(om), len(nm)):
        similar = difflib.SequenceMatcher(None, _text(om[i])[:500], _text(nm[i])[:500]).ratio()
    score = (i, -len(params), similar)
    return '; '.join(parts) or 'identical', score


class Explainer:
    """Reasons for the calls of process runs, from the call log and the registry."""

    def __init__(self, calls_path, registry):
        self.db = sqlite3.connect('file:%s?mode=ro' % calls_path, uri=True, timeout=30) \
            if os.path.exists(calls_path) else None
        if self.db is not None:
            self.db.row_factory = sqlite3.Row
        self.reg = registry
        self._recs = {}
        self._bodies = {}

    def close(self):
        if self.db is not None:
            self.db.close()

    def rec(self, run_id):
        if run_id not in self._recs:
            self._recs[run_id] = self.reg.get(run_id)
        return self._recs[run_id]

    def body(self, digest):
        if digest not in self._bodies:
            row = self.db.execute('SELECT body FROM bodies WHERE digest=?', (digest,)).fetchone()
            self._bodies[digest] = json.loads(row[0]) if row else None
        return self._bodies[digest]

    def rows(self, proc):
        if self.db is None:
            return []
        return [dict(r) for r in self.db.execute(
            'SELECT * FROM calls WHERE proc=? ORDER BY seq', (proc,))]

    def explain(self, proc, nearest=True):
        """[(row, outcome, reason, detail)] for every call of process run `proc` and its units.
        outcome: reused, joined, generated, error. Without `nearest`, a new request's detail is
        None (the closest earlier request is the costly part; see `nearest`)."""
        self.nearest_on = nearest
        out = []
        for row in self.rows(proc):
            statuses = [s for s in row['statuses'].split(',') if s]
            if row['error']:
                out.append((row, 'error', 'error', row['error']))
            elif statuses and all(s in REUSED for s in statuses):
                out.append((row, 'reused', None, None))
            elif 'generated' not in statuses:
                out.append((row, 'joined', None, None))
            else:
                reason, detail = self.why(row)
                out.append((row, 'generated', reason, detail))
        return out

    def why(self, row):
        if row['node'].endswith(':u'):
            return 'uncached', 'uncached node (%s)' % (row['reason'] or 'not cached')
        ranks = [r for r, s in zip(row['ranks'].split(','), row['statuses'].split(','))
                 if s == 'generated']
        rank = ranks[0] if ranks else '-'
        sent = [dict(r) for r in self.db.execute(
            'SELECT * FROM calls WHERE body=? AND node=? AND proc<>? AND seq<? '
            'ORDER BY seq DESC LIMIT 200', (row['body'], row['node'], row['proc'], row['seq']))]
        earlier = [e for e in sent if not e['error']]
        if sent and not earlier:
            return 'after_error', 'earlier runs sent this request, but it failed (%s, run %s)' % (
                sent[0]['error'], sent[0]['proc'])
        if not earlier:
            return 'new_request', self._nearest(row) if self.__dict__.get('nearest_on', True) \
                else None
        if row.get('rev') and all(e.get('rev') and e['rev'] != row['rev'] for e in earlier):
            return 'model', 'earlier runs sent it to model revision %s, this run to %s' % (
                earlier[0]['rev'], row['rev'])
        same = [e for e in earlier if e['grp'] == row['grp']]
        if same:
            stored = set()
            for e in same:
                stored |= {r for r, s in zip(e['ranks'].split(','), e['statuses'].split(','))
                           if r != '-' and s != 'error'}
            if rank != '-' and rank not in stored:
                return 'more_repeats', 'occurrence %s of this request in its group; earlier runs ' \
                    'stored %d occurrence(s)' % (rank, len(stored))
            return 'unexplained', 'stored in this group by run %s (generated concurrently?)' % (
                same[0]['proc'])
        return self._other_group(row, earlier[0])

    def _other_group(self, row, prev):
        a, b = self.rec(row['run']), self.rec(prev['run'])
        where = 'run %s' % prev['proc']
        if a is None or b is None:
            return 'unexplained', 'stored in another group by %s' % where
        if a['experiment'] != b['experiment']:
            return 'experiment', 'sent by %s of experiment %s; experiments never share' % (
                where, b['experiment'] or '(none)')
        if a['phase'] != b['phase']:
            return 'phase', 'sent by %s in the %s phase; phases never share' % (where, b['phase'])
        if a['granularity'] != b['granularity']:
            return 'granularity', 'sent by %s as a %s, here a %s' % (
                where, b['granularity'], a['granularity'])
        ca, cb = self.reg.classes(a).split(a['coords']), self.reg.classes(b).split(b['coords'])
        if ca['replicate'] != cb['replicate']:
            diff = sorted(k for k in set(ca['replicate']) | set(cb['replicate'])
                          if ca['replicate'].get(k) != cb['replicate'].get(k))
            return 'replicate', '%s here, %s in %s: replicates never share' % (
                ', '.join('%s=%s' % (k, ca['replicate'].get(k, '(unset)')) for k in diff[:3]),
                ', '.join('%s=%s' % (k, cb['replicate'].get(k, '(unset)')) for k in diff[:3]),
                where)
        if ca['case'] != cb['case']:
            return 'case', 'sent by %s for another case' % where
        if (row['seed'] or None) != (prev['seed'] or None):
            return 'seed', 'sent by %s with seed %s, here %s' % (where, prev['seed'], row['seed'])
        if (row['role'] or None) != (prev['role'] or None):
            return 'route', 'sent by %s as role %s, here %s' % (where, prev['role'], row['role'])
        top_a = self.rec(a['parent']) if a['parent'] else a
        top_b = self.rec(b['parent']) if b['parent'] else b
        ordinal = row['ordinal'] or 1
        if a['granularity'] == 'unit' and a['coords'] == b['coords'] and \
                ordinal != (prev['ordinal'] or 1):
            # how many identical units (these coordinates) the earlier run had
            seen = [o or 1 for r, o in self.db.execute(
                'SELECT DISTINCT run, ordinal FROM calls WHERE proc=?', (prev['proc'],))
                if (self.rec(r) or {}).get('coords') == a['coords']]
            if seen and ordinal > max(seen):
                return 'more_repeats', 'unit ordinal %d of identical units; run %s had %d' % (
                    ordinal, prev['proc'], max(seen))
            return 'arrival_order', 'stored by the identical unit with ordinal %s (slot %s) of %s; ' \
                'this unit has ordinal %s (slot %s): units with equal coordinates are numbered ' \
                'in the order they start, which can change between runs; a coordinate that ' \
                'tells them apart (e.g. case) makes it stable' % (
                    prev['ordinal'] or 1, b['slot'], where, row['ordinal'] or 1, a['slot'])
        if top_a and top_b and top_a.get('rerun_of') and (
                top_a['rerun_of'] == top_b['run_id'] or top_a['version'] == top_b['version']):
            return 'rerun', 'the program is unchanged since run %s, so this run is a rerun: it ' \
                'takes a new slot (%s; that run: %s) and draws new samples by design' % (
                    top_a['rerun_of'], a['slot'], b['slot'])
        return 'slot', 'stored by %s in slot %s; this run holds slot %s' % (
            where, b['slot'], a['slot'])

    def nearest(self, row):
        """The closest earlier request of the experiment to `row`'s, and where they differ."""
        memo = self.__dict__.setdefault('_near', {})
        if row['body'] not in memo:
            memo[row['body']] = self._nearest_uncached(row)
        return memo[row['body']]

    _nearest = nearest

    def _candidates(self, row):
        """Up to 100 distinct request bodies earlier runs of the experiment sent to this model
        before this process run began (computed once per process run and model)."""
        key = (row['proc'], row['exp'], row['model'])
        cache = self.__dict__.setdefault('_cands', {})
        if key not in cache:
            first = self.db.execute('SELECT MIN(seq) FROM calls WHERE proc=?',
                                    (row['proc'],)).fetchone()[0] or row['seq']
            cache[key] = [(r[0], r[1]) for r in self.db.execute(
                'SELECT body, proc FROM calls WHERE exp=? AND model IS ? AND proc<>? AND seq<? '
                'ORDER BY seq DESC LIMIT 2000', (row['exp'], row['model'], row['proc'], first))]
            seen, out = set(), []
            for body, proc in cache[key]:
                if body not in seen:
                    seen.add(body)
                    out.append((body, proc))
                if len(out) >= 100:
                    break
            cache[key] = out
        return cache[key]

    def _nearest_uncached(self, row):
        new = self.body(row['body'])
        if new is None or row['exp'] is None:
            return 'no earlier run sent it'
        rough = []
        for body, proc in self._candidates(row):
            old = self.body(body)
            if old is None or body == row['body']:
                continue
            om, nm = old.get('messages') or [], new.get('messages') or []
            i = 0
            while i < min(len(om), len(nm)) and om[i] == nm[i]:
                i += 1
            rough.append(((i, old.get('params') == new.get('params')), old, proc))
        rough.sort(key=lambda t: t[0], reverse=True)
        best = None
        for _, old, proc in rough[:10]:          # the exact (difflib) score for the best few
            text, score = body_diff(old, new)
            if best is None or score > best[0]:
                best = (score, text, proc)
        if best is None:
            return 'no earlier run of the experiment called this model'
        return 'closest earlier request (run %s): %s' % (best[2], best[1])


def run_summary(calls_path, registry_path, proc, limit=3):
    """({outcome: n}, [[reason, n, example]]) of a process run, for the end-of-run summary: with
    its own connections (it runs in a worker thread), and the closest earlier request computed
    only for the example shown."""
    from .registry import Registry
    reg = Registry(registry_path)
    ex = Explainer(calls_path, reg)
    try:
        items = ex.explain(proc, nearest=False)
        counts, top, _ = summarize(items, limit)
        rows = {r: row for row, o, r, d in reversed(items) if o == 'generated'}
        top = [[r, n, ex.nearest(rows[r]) if r == 'new_request' and d is None else d]
               for r, n, d in top]
        return dict(counts), top
    finally:
        ex.close()
        reg.close()


def summarize(items, limit=3):
    """Counts by outcome and the most frequent generation reasons with one example each."""
    counts = collections.Counter(o for _, o, _, _ in items)
    reasons = collections.Counter(r for _, o, r, _ in items if o == 'generated')
    example = {}
    for _, o, r, d in items:
        if o == 'generated' and r not in example:
            example[r] = d
    top = [(r, n, example[r]) for r, n in reasons.most_common(limit)]
    return counts, top, reasons
