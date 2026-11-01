"""The manager's dispatch queue, group commit and replica failover (reports/sysval_20261009,
build section):
- the heap queue picks exactly what the original scan picked, under random adds, joins, waiter
  removals, owner transfers, served stamps and per-backend caps;
- with a batched store, no reply leaves before its writes are committed, and a crash keeps every
  sample a requester was given;
- a dead replica is skipped after it fails, and a retry goes to another replica.
    PYTHONPATH=src:. python3 -m unittest tests.kernel.test_evalrt_queue_commit
"""
import asyncio
import os
import random
import shutil
import sqlite3
import tempfile
import unittest

from copula.kernel.manager.flights import _Queue
from copula.kernel.store.entries import EntryStore
from copula.kernel.sdk import Session
from copula.kernel.testing.rig import Rig


class F:
    """A queued flight as the queue sees it."""

    def __init__(self, key_hash, seq, owner, model, waiters):
        self.key_hash, self.seq, self.owner, self.model = key_hash, seq, owner, model
        self.waiters = list(waiters)


def scan(flights, served, room):
    """The original `_pick`: the minimum over ready flights of (-distinct waiters, owner's last
    dispatch, arrival)."""
    ready = [f for f in flights if room(f.model)]
    if not ready:
        return None
    return min(ready, key=lambda f: (-len(set(f.waiters)), served.get(f.owner, -1), f.seq))


class QueueOrder(unittest.TestCase):
    def test_heap_queue_picks_what_the_scan_picked(self):
        for trial in range(300):
            rng = random.Random(trial)
            served, ref = {}, []
            q = _Queue(lambda o: served.get(o, -1))
            owners = ['t%d' % i for i in range(rng.randint(1, 6))]
            models = ['actor', 'small'][:rng.randint(1, 2)]
            seq, stamp, picked = 0, 0, 0
            full = set()
            for _ in range(rng.randint(20, 200)):
                op = rng.random()
                if op < 0.4 or not ref:
                    f = F('k%d' % seq, seq, rng.choice(owners), rng.choice(models),
                          [rng.choice(owners)])
                    seq += 1
                    ref.append(f)
                    q.add(f)
                elif op < 0.55:                       # another ticket joins a queued flight
                    f = rng.choice(ref)
                    f.waiters.append(rng.choice(owners))
                    q.touched(f)
                elif op < 0.62:                       # a cancel drops a waiter or transfers
                    f = rng.choice(ref)
                    gone = rng.choice(f.waiters)
                    others = [w for w in f.waiters if w != gone]
                    if not others:
                        ref.remove(f)
                        q.remove(f)
                    else:
                        if f.owner == gone:
                            f.owner = others[0]
                        f.waiters = others
                        q.touched(f)
                elif op < 0.67:                       # a backend fills or frees
                    m = rng.choice(models)
                    full.symmetric_difference_update({m})
                else:
                    room = lambda m: m not in full    # noqa: E731
                    want = scan(ref, served, room)
                    got = q.pick(room)
                    self.assertIs(got, want, 'trial %d' % trial)
                    if got is not None:
                        ref.remove(got)
                        served[got.owner] = stamp       # dispatched: the owner was served
                        stamp += 1
                        picked += 1
                self.assertEqual(len(q), len(ref))
            room = lambda m: True                     # noqa: E731
            while ref:                                # drain: the full order agrees
                want = scan(ref, served, room)
                got = q.pick(room)
                self.assertIs(got, want)
                ref.remove(got)
                served[got.owner] = stamp
                stamp += 1
            self.assertIsNone(q.pick(room))


    def test_cost_per_dispatch_does_not_grow_with_the_queue(self):
        import time
        for owners in (1, 50):
            served, stamp = {}, 0
            q = _Queue(lambda o: served.get(o, -1))
            flights = [F('k%d' % i, i, 't%d' % (i % owners), 'actor', ['t%d' % (i % owners)])
                       for i in range(20000)]
            t0 = time.perf_counter()
            for f in flights:
                q.add(f)
            while q:
                f = q.pick(lambda m: True)
                served[f.owner] = stamp
                stamp += 1
            self.assertLess(time.perf_counter() - t0, 5.0, '%d owners' % owners)


class GroupCommit(unittest.IsolatedAsyncioTestCase):
    async def test_replies_follow_commits_and_survive_a_crash(self):
        rig = await Rig('commit').start()
        self.assertTrue(rig.store.batched)
        try:
            s = await Session.connect(rig.socket)
            s.register_llm('gen', cache=True, randomness='stochastic', max_tokens=8)
            got = []
            async with s.run(s.group('c')):
                rs = await asyncio.gather(*[s.chat('gen', [{'role': 'user', 'content': 'q%d' % i}])
                                            for i in range(50)])
                got = [r['choices'][0]['message']['content'] for r in rs]
                # every reply was sent after a commit: nothing is pending now
                self.assertFalse(rig.store.dirty)
            self.assertLess(rig.store.commits, 50 * 2)          # grouped: fewer commits than writes
            await s.aclose()
            path = rig.store.path
            # a crash: the manager stops without draining; what was handed out is in the file
            await rig.manager.stop()
            rig.server.close()
            rig.store.db.close()                                 # no final commit
            db = sqlite3.connect(path)
            stored = {row[0] for row in db.execute('SELECT sample_json FROM entries')}
            db.close()
            for content in got:
                self.assertTrue(any(content in row for row in stored))
        finally:
            shutil.rmtree(rig.dir, ignore_errors=True)

    async def test_claim_lookups_use_the_ticket_index(self):
        d = tempfile.mkdtemp()
        try:
            st = EntryStore(os.path.join(d, 'e.db'), batched=True)
            for i in range(3000):                   # one partition, many pools
                st.claim('grp/g', 'r', 'r', 'pool%d' % i, 1, 'r/g', float(i))
            st.commit()
            import time
            t0 = time.perf_counter()
            for i in range(300):
                self.assertEqual(st.ticket_claims('r/g', 'grp/g', 'r', 'pool%d' % i), [(1, None)])
            per = (time.perf_counter() - t0) / 300
            self.assertLess(per, 0.0005)            # an index lookup, not a partition scan
            st.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    async def test_a_rolled_back_batch_fails_its_waiters_then_recovers(self):
        """Review fix D3: SQLite rolls the open batch back on a full disk; the next commit says so
        once, and the store works again when space returns."""
        from copula.kernel.store.entries import StoreError
        d = tempfile.mkdtemp()
        try:
            st = EntryStore(os.path.join(d, 'e.db'), batched=True)
            pages = st.db.execute('PRAGMA page_count').fetchone()[0]
            st.db.execute('PRAGMA max_page_count = %d' % (pages + 2))
            with self.assertRaises(StoreError):
                for i in range(10000):
                    st.put_once('k%d' % i, {'node': 'n'}, {'content': 'x' * 2000}, 't', 1, 0.0)
            with self.assertRaises(StoreError):
                st.commit()
            st.db.execute('PRAGMA max_page_count = 1073741823')
            st.put_once('after', {'node': 'n'}, {'content': 'y'}, 't', 1, 0.0)
            st.commit()
            self.assertIsNotNone(st.get('after'))
            st.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    async def test_unbatched_store_unchanged(self):
        d = tempfile.mkdtemp()
        try:
            st = EntryStore(os.path.join(d, 'e.db'))
            self.assertFalse(st.batched)
            st.put_once('k', {'node': 'n'}, {'content': 'x'}, 't', 1, 0.0)
            self.assertFalse(st.dirty)
            st.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)


class Failover(unittest.IsolatedAsyncioTestCase):
    def test_concurrent_failures_do_not_compound(self):
        from copula.kernel.manager.vllm_client import DOWN_S, VLLMClient
        from copula.kernel.testing.mock_model import MOCK_DEFAULTS
        c = VLLMClient('http://127.0.0.1:1,http://127.0.0.1:2', 'tiny', MOCK_DEFAULTS, {'m': 1})
        for _ in range(64):
            c.mark_down(1)
        self.assertEqual(c._down[1][1], DOWN_S)
        self.assertEqual(c._up(), {0})

    async def test_a_dead_replica_fails_no_flight(self):
        from copula.kernel.manager.flights import Manager
        from copula.kernel.manager.vllm_client import VLLMClient
        from copula.kernel.testing.fake_vllm import FakeVLLM
        from copula.kernel.testing.mock_model import MOCK_DEFAULTS
        import socket as sk
        with sk.socket() as probe:                  # a port nobody listens on
            probe.bind(('127.0.0.1', 0))
            dead = 'http://127.0.0.1:%d' % probe.getsockname()[1]
        live = FakeVLLM(served='tiny').start()
        d = tempfile.mkdtemp()
        try:
            client = VLLMClient(dead + ',' + live.url, 'tiny', MOCK_DEFAULTS, {'m': 1},
                                timeout=10, workers=16)
            m = Manager(EntryStore(os.path.join(d, 'e.db'), batched=True), client,
                        max_attempts=3, backoff_base=0.0)
            await m.start()
            m.register_pool_ticket('t', ['grp/g'], 't', client.model_rev, {'c': 'c'})
            rs = await asyncio.gather(*[m.draw('t', 'n', {'messages': [{'role': 'user',
                                                                        'content': 'q%d' % i}],
                                                          'max_tokens': 4}, 'c',
                                               scope=('s',)) for i in range(200)])
            self.assertTrue(all('error' not in r['sample'] for r in rs))
            self.assertLess(m.counters['transport_errors'], 20)   # the dead one is skipped
            await m.stop()
            m.store.close()
        finally:
            live.close()
            shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
