"""The generation ledger: one JSONL row per HTTP attempt the manager makes (campaign 2026-10-02,
SCHEMA section 4). It is the authoritative source of what generation cost: every attempt, failed
ones included, with the cache salt sent, the replica, timing and the reply's token counts.

Rows are built on the event-loop thread only and buffered. Every `flush_s` seconds a periodic task
hands what is buffered to a writer thread of its own, which does one `write()` and one `flush()` per
file, so a slow or stalled filesystem never blocks the manager's event loop (the loop only appends
to the buffers; rows are serialised by the writer). Writes are serialised under one lock and take
the buffers inside it, so rows reach the file in the order they were made, whichever thread writes
them. The `start` row is written
before the manager's socket is bound and the buffer is flushed again (synchronously, on the
calling thread, after any write in progress) on drain and stop. There is no fsync: a SIGKILL may
lose about a second of rows, which the cost reconciliation catches (PLAN 3.7).

`bodies` (optional) keeps each distinct prompt once, by `prompt_digest` (the digest of the
normalised messages; the salt is not part of it), for an exact prefix-cache simulation. A body row
is always flushed before the first attempt row that references it.
"""
import asyncio
import json
import os
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

LEDGER_VERSION = 1
META_FIELDS = ('t_send', 't_done', 'http_status', 'error', 'response_id', 'prompt_tokens',
               'cached_tokens', 'completion_tokens', 'cached_present', 'finish_reason')


def manager_id(start_t, host=None, pid=None):
    """'{hostname}:{pid}:{int(start time)}': names one manager process in every row."""
    return '%s:%d:%d' % (host or socket.gethostname(), os.getpid() if pid is None else pid,
                         int(start_t))


def meta_fields(meta):
    """The ledger columns an attempt's meta fills (all None without meta)."""
    meta = meta or {}
    return {name: meta.get(name) for name in META_FIELDS}


def _line(row):
    return json.dumps(row, ensure_ascii=False) + '\n'


class GenLedger:
    def __init__(self, path, bodies=None, flush_s=1.0, mgr=None, clock=time.time):
        self.path = path
        self.bodies_path = bodies
        self.flush_s = flush_s
        self.clock = clock
        self.mgr = mgr or manager_id(clock())
        self.rows = 0                   # attempt rows written or buffered by this process
        self.probe = 'off'              # 'ok' once the startup probe passed
        self._rows = []
        self._bodies = []
        self._seen = set()              # first 16 bytes of each prompt digest already written
        self._task = None
        self._warned = 0.0
        self._buffer = threading.Lock()     # the buffers (appends on the loop, swaps by writers)
        self._writing = threading.Lock()    # one writer at a time, buffers taken inside it
        self._writer = None                 # the periodic flush's thread (started with `start`)
        self._out = open(path, 'a', encoding='utf-8')
        self._body_out = open(bodies, 'a', encoding='utf-8') if bodies else None

    def row(self, ev, **fields):
        row = {'v': LEDGER_VERSION, 'ev': ev, 'src': 'manager', 'mgr': self.mgr}
        row.update(fields)
        with self._buffer:
            self._rows.append(row)
        if ev == 'attempt':
            self.rows += 1
        return row

    def body(self, prompt_digest, messages):
        """Queue the prompt behind `prompt_digest` unless this process already wrote it."""
        if self._body_out is None:
            return
        seen = bytes.fromhex(prompt_digest[:32])
        if seen in self._seen:
            return
        self._seen.add(seen)
        with self._buffer:
            self._bodies.append({'v': LEDGER_VERSION, 'prompt_digest': prompt_digest,
                                 'messages': messages})

    def flush(self):
        """Write what is buffered, on the calling thread (after any write in progress). A failed
        write (a full disk) keeps the rows for the next flush and is reported on stderr, never
        raised: metering must not stop generation."""
        with self._writing:
            with self._buffer:
                bodies, self._bodies = self._bodies, []
                rows, self._rows = self._rows, []
            try:
                if bodies and self._body_out is not None:
                    self._body_out.write(''.join(_line(b) for b in bodies))
                    self._body_out.flush()
                    bodies = []
                if rows:
                    self._out.write(''.join(_line(r) for r in rows))
                    self._out.flush()
                    rows = []
            except OSError as err:
                with self._buffer:          # back in front: the next flush writes them first
                    self._bodies[:0] = bodies
                    self._rows[:0] = rows
                now = time.time()
                if now - self._warned > 60:
                    self._warned = now
                    print(json.dumps({'event': 'ledger_write_failed', 'path': self.path,
                                      'error': repr(err)[:300], 'buffered': len(self._rows)}),
                          file=sys.stderr, flush=True)
                return False
        return True

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(self.flush_s)
            await loop.run_in_executor(self._writer, self.flush)

    def start(self):
        """Start the periodic flush (in the running event loop; the writes on a thread)."""
        if self._task is None:
            if self._writer is None:
                self._writer = ThreadPoolExecutor(max_workers=1,
                                                  thread_name_prefix='gen-ledger')
            self._task = asyncio.ensure_future(self._run())

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        self.flush()                    # waits for a write the cancelled task left running

    def close(self, stats=None):
        """Write the `stop` row (with `stats`) when given, flush and close the files."""
        if stats is not None:
            self.row('stop', t=self.clock(), stats=stats)
        self.flush()
        if self._writer is not None:
            self._writer.shutdown(wait=True)
            self._writer = None
        self._out.close()
        if self._body_out is not None:
            self._body_out.close()
