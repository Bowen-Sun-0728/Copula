"""The sample store: write-once, never evicted, opened by exactly one manager process.

The first `put_once` of a key wins; a later write of the same key changes nothing and returns the
stored row (RUNTIME_V0 section 4.4). `put_once` returns only after the row is committed with
`synchronous=FULL`, which is what lets the manager write before it wakes any waiter.

The database is opened with an exclusive lock held for the connection's lifetime. A second manager on
the same store therefore fails at start instead of silently breaking single flight, and WAL runs
without a shared-memory file, which network file systems handle badly.

Group commit (`batched=True`, sysval build 2026-10-09): writes go into one open transaction that
`commit()` closes, so one fsync covers every write made since the last commit. Reads on the
connection see the uncommitted writes at once. The manager commits before it answers any request or
wakes any waiter (`Manager.durable`), so what a requester is told is durable exactly as before; a
crash loses only writes nobody was told about. Without it, every write is its own transaction.
"""
import json
import secrets
import sqlite3
import time

from ..core.canon import canonical

SCHEMA_VERSION = 5        # 2 added pool_claims (key v3 ledgers); 3 adds the case to every claim
                          # (pool_claims_v3); 4 adds `computed` (results of registered kinds with
                          # memo='store'); 5 adds the store's `secret` (seeds of stochastic
                          # registered kinds) and `pool_claims_v5`, the ledger of case-free sample
                          # addresses (key v5, N14); older stores are upgraded in place, their old
                          # rows kept for history, never aliased and never read


class StoreError(RuntimeError):
    """The store cannot be opened or written."""


def _text(obj):
    return canonical(obj).decode('utf-8')


class EntryStore:
    def __init__(self, path, synchronous='FULL', timeout=5.0, batched=False):
        if synchronous not in ('FULL', 'NORMAL', 'OFF'):
            raise StoreError('bad synchronous mode %r' % (synchronous,))
        self.path = str(path)
        db = sqlite3.connect(self.path, timeout=timeout, isolation_level=None)
        try:
            db.execute('PRAGMA locking_mode=EXCLUSIVE')
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('PRAGMA synchronous=%s' % synchronous)
            db.execute('BEGIN EXCLUSIVE')      # take the lock now: fail fast if a manager holds it
            db.execute('CREATE TABLE IF NOT EXISTS meta '
                       '(name TEXT PRIMARY KEY, value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS entries ('
                       ' key_hash TEXT PRIMARY KEY,'
                       ' key_json TEXT NOT NULL,'
                       ' node TEXT NOT NULL,'
                       ' sample_json TEXT NOT NULL,'
                       ' generated_by TEXT NOT NULL,'
                       ' created_at REAL NOT NULL,'
                       ' attempts INTEGER NOT NULL) WITHOUT ROWID')
            db.execute('CREATE INDEX IF NOT EXISTS entries_by_ticket ON entries (generated_by)')
            # Key v3 ledger: which sample numbers of a pool each program has read in a partition.
            db.execute('CREATE TABLE IF NOT EXISTS pool_claims ('
                       ' part TEXT NOT NULL, consumer TEXT NOT NULL, pool TEXT NOT NULL,'
                       ' idx INTEGER NOT NULL, ticket TEXT NOT NULL, created_at REAL NOT NULL,'
                       ' PRIMARY KEY (part, consumer, pool, idx)) WITHOUT ROWID')
            db.execute('CREATE INDEX IF NOT EXISTS claims_by_ticket ON pool_claims (ticket, pool)')
            # Schema 3: claims per case (samples are shared within a case only).
            # `role` is the claiming draw's role in its job (1 + earlier draws of the job and
            # case with the identical request): a restarted or resumed job re-reads by role.
            db.execute('CREATE TABLE IF NOT EXISTS pool_claims_v3 ('
                       ' part TEXT NOT NULL, consumer TEXT NOT NULL, case_key TEXT NOT NULL,'
                       ' pool TEXT NOT NULL, idx INTEGER NOT NULL, ticket TEXT NOT NULL,'
                       ' created_at REAL NOT NULL, role INTEGER,'
                       ' PRIMARY KEY (part, consumer, case_key, pool, idx)) WITHOUT ROWID')
            db.execute('CREATE INDEX IF NOT EXISTS claims_v3_by_ticket ON pool_claims_v3'
                       ' (ticket, part, case_key, pool)')
            # Schema 5: the ledger of key-v5 addresses (claims stay per case; the addresses they
            # name have no case). A fresh table, so numbers claimed under v4 addresses, which name
            # other samples, are not skipped.
            db.execute('CREATE TABLE IF NOT EXISTS pool_claims_v5 ('
                       ' part TEXT NOT NULL, consumer TEXT NOT NULL, case_key TEXT NOT NULL,'
                       ' pool TEXT NOT NULL, idx INTEGER NOT NULL, ticket TEXT NOT NULL,'
                       ' created_at REAL NOT NULL, role INTEGER,'
                       ' PRIMARY KEY (part, consumer, case_key, pool, idx)) WITHOUT ROWID')
            db.execute('CREATE INDEX IF NOT EXISTS claims_v5_by_ticket ON pool_claims_v5'
                       ' (ticket, part, case_key, pool)')
            # Schema 4: results of registered kinds (pure), by (case key, call identity); no
            # partition, since a deterministic result may cross partitions. Write-once.
            db.execute('CREATE TABLE IF NOT EXISTS computed ('
                       ' address TEXT PRIMARY KEY, kind TEXT NOT NULL, kind_rev TEXT NOT NULL,'
                       ' value_json TEXT NOT NULL, ticket TEXT NOT NULL,'
                       ' created_at REAL NOT NULL) WITHOUT ROWID')
            row = db.execute("SELECT value FROM meta WHERE name = 'schema'").fetchone()
            if row is None:
                db.execute("INSERT INTO meta VALUES ('schema', ?)", (str(SCHEMA_VERSION),))
            elif int(row[0]) in (1, 2, 3, 4):
                db.execute("UPDATE meta SET value = ? WHERE name = 'schema'", (str(SCHEMA_VERSION),))
            elif int(row[0]) != SCHEMA_VERSION:
                raise StoreError('store schema %s, this code reads %d' % (row[0], SCHEMA_VERSION))
            # The secret that seeds stochastic registered kinds: created once, lives with the store
            # (whoever copies the store copies the draws it can still compute).
            secret = db.execute("SELECT value FROM meta WHERE name = 'secret'").fetchone()
            if secret is None:
                secret = (secrets.token_hex(32),)
                db.execute("INSERT INTO meta VALUES ('secret', ?)", secret)
            db.execute('COMMIT')
        except (sqlite3.Error, StoreError) as exc:
            db.close()
            if isinstance(exc, StoreError):
                raise
            raise StoreError('cannot open store %s: %s' % (self.path, exc))
        self.db = db
        self.secret = secret[0]
        self.batched = bool(batched)
        self.dirty = False              # an open transaction holds uncommitted writes
        self.lost = False               # an error rolled the open batch back
        self.commits = 0

    def _write(self, sql, args):
        if self.batched:
            if self.dirty and not self.db.in_transaction:
                self.lost = True        # SQLite rolled the batch back (disk full, I/O error)
            if not self.db.in_transaction:
                self.db.execute('BEGIN')
                self.dirty = True
        return self.db.execute(sql, args)

    def commit(self):
        """Make every write so far durable (one transaction, one fsync). Raises StoreError when
        an error had rolled back writes made since the last commit: whoever waits on this commit
        must be told it failed (the store itself is usable again)."""
        lost = self.lost or (self.dirty and not self.db.in_transaction)
        self.lost = False
        if self.dirty:
            self.dirty = False
            if self.db.in_transaction:
                self.db.execute('COMMIT')
                self.commits += 1
        if lost:
            raise StoreError('a write error rolled back uncommitted writes')

    @staticmethod
    def _row(row):
        key_hash, key_json, _node, sample_json, generated_by, created_at, attempts = row
        return {'key_hash': key_hash, 'key': json.loads(key_json), 'sample': json.loads(sample_json),
                'generated_by': generated_by, 'created_at': created_at, 'attempts': attempts}

    def get(self, key_hash):
        row = self.db.execute('SELECT * FROM entries WHERE key_hash = ?', (key_hash,)).fetchone()
        return None if row is None else self._row(row)

    def put_once(self, key_hash, key, sample, generated_by, attempts, created_at=None, node=None):
        """Insert unless present. Returns (stored row, whether this call inserted it). `node` labels
        the entry for billing; it defaults to the key's own node (pool keys have none)."""
        created_at = time.time() if created_at is None else created_at
        try:
            cursor = self._write(
                'INSERT OR IGNORE INTO entries VALUES (?, ?, ?, ?, ?, ?, ?)',
                (key_hash, _text(key), node if node is not None else key['node'], _text(sample),
                 generated_by, float(created_at), int(attempts)))
        except sqlite3.Error as exc:
            raise StoreError('write of %s failed: %s' % (key_hash, exc))
        return self.get(key_hash), cursor.rowcount == 1

    def generated_counts(self, ticket):
        """{node: number of entries this ticket generated} -- what the ticket is billed for."""
        rows = self.db.execute('SELECT node, COUNT(*) FROM entries WHERE generated_by = ? '
                               'GROUP BY node', (ticket,)).fetchall()
        return {node: n for node, n in rows}

    # -- key v3 ledgers ---------------------------------------------------------------------------

    def claimed(self, part, consumer, case_key, pool):
        """Sample numbers of `pool` that program `consumer` has claimed for one case in `part`."""
        rows = self.db.execute('SELECT idx FROM pool_claims_v5 WHERE part = ? AND consumer = ? '
                               'AND case_key = ? AND pool = ?',
                               (part, consumer, case_key, pool)).fetchall()
        return {r[0] for r in rows}

    def ticket_claims(self, ticket, part, case_key, pool):
        """[(number, role)] `ticket` claimed from `pool` for one case in `part`, in claim order (a
        restarted or resumed job re-reads them, by role when the draw has one)."""
        # INDEXED BY: without statistics the planner prefers the primary key's `part` prefix and
        # scans every claim of the partition (sysval build: O(claims) per draw, found by E3)
        rows = self.db.execute('SELECT idx, role FROM pool_claims_v5 INDEXED BY claims_v5_by_ticket'
                               ' WHERE ticket = ? AND part = ? AND case_key = ? AND pool = ?'
                               ' ORDER BY created_at, idx',
                               (ticket, part, case_key, pool)).fetchall()
        return [(r[0], r[1]) for r in rows]

    def claim(self, part, consumer, case_key, pool, idx, ticket, created_at=None, role=None):
        created_at = time.time() if created_at is None else created_at
        try:
            self._write('INSERT INTO pool_claims_v5 VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                            (part, consumer, case_key, pool, int(idx), ticket, float(created_at),
                             None if role is None else int(role)))
        except sqlite3.Error as exc:
            raise StoreError('claim of %s/%s #%d failed: %s' % (part, pool, idx, exc))

    # -- results of registered kinds ---------------------------------------------------------------

    def get_computed(self, address):
        """(value, ticket that wrote it), or (None, None)."""
        row = self.db.execute('SELECT value_json, ticket FROM computed WHERE address = ?',
                              (address,)).fetchone()
        return (None, None) if row is None else (json.loads(row[0]), row[1])

    def put_computed(self, address, kind, kind_rev, value, ticket, created_at=None):
        """Insert unless present (the first write wins; a deterministic result, or a seeded draw,
        is the same whoever computes it). True when this write was the first."""
        created_at = time.time() if created_at is None else created_at
        try:
            cur = self._write('INSERT OR IGNORE INTO computed VALUES (?, ?, ?, ?, ?, ?)',
                              (address, kind, kind_rev, _text(value), ticket, float(created_at)))
        except sqlite3.Error as exc:
            raise StoreError('write of computed %s failed: %s' % (address, exc))
        return cur.rowcount == 1

    def count(self):
        return self.db.execute('SELECT COUNT(*) FROM entries').fetchone()[0]

    def close(self):
        try:
            self.commit()
        finally:
            self.db.close()
