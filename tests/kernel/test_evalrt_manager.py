"""evalrt cache manager: the core invariant and its rules (RUNTIME_V0 sections 4.4, 12.5, 13 step 1).

Every test runs against the mock model (sample = f(nonce, key)), so none needs a GPU:
    PYTHONPATH=src python3 -m unittest tests.kernel.test_evalrt_manager
"""
import asyncio
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest

from copula.kernel.manager.flights import Manager, ManagerError
from copula.kernel.manager.server import AsyncClient, serve
from copula.kernel.store.entries import EntryStore
from copula.kernel.testing.mock_model import MockModel

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src')
CASES = {'c%d' % i: 'casekey-%d' % i for i in range(4)}


def body(text='q', max_tokens=16):
    return {'messages': [{'role': 'user', 'content': text}], 'max_tokens': max_tokens}


def request(case, node, rep, text='q'):
    return (case, node, rep, text)


class Rig:
    """A manager on a fresh store, served on a Unix socket, in this test's event loop."""

    def __init__(self, **model_kw):
        self.dir = tempfile.mkdtemp(prefix='evalrt-')
        self.socket = os.path.join(self.dir, 'm.sock')
        self.model = MockModel(model_kw.pop('nonce', 'test'), **model_kw)
        self.store = None
        self.manager = None
        self.server = None
        self.clients = []

    async def start(self, **manager_kw):
        manager_kw.setdefault('backoff_base', 0.0)
        self.store = EntryStore(os.path.join(self.dir, 'entries.db'))
        self.manager = Manager(self.store, self.model, **manager_kw)
        await self.manager.start()
        self.server = await serve(self.manager, self.socket)
        return self

    async def client(self, *tickets):
        c = await AsyncClient.connect(self.socket)
        self.clients.append(c)
        for t in tickets:
            await c.call('register_ticket', ticket=t, generation='exec/0',
                         model_rev=self.model.model_rev, case_keys=CASES)
        return c

    async def close(self):
        for c in self.clients:
            await c.close()
        self.server.close()
        await self.server.wait_closed()
        await self.manager.stop()
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)


async def ask(client, ticket, req):
    case, node, rep, text = req
    return await asyncio.wait_for(client.get_or_generate(ticket, case, node, rep, 0, body(text)), 30)


async def until(condition, timeout=10.0):
    """Wait for manager state rather than for wall-clock time."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise AssertionError('condition not reached in %.1f s' % timeout)
        await asyncio.sleep(0.002)


def job_set(rng, n_jobs=6):
    """Overlapping jobs: each draws a random subset of one pool of 30 distinct requests."""
    pool = []
    for case in ('c0', 'c1', 'c2'):
        for node, reps in (('constrain', 2), ('draft', 4), ('answer', 4)):
            for rep in range(1, reps + 1):
                pool.append(request(case, node, rep))
    return [rng.sample(pool, rng.randint(len(pool) // 2, len(pool))) for _ in range(n_jobs)]


class ManagerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.rigs = []

    async def asyncTearDown(self):
        for rig in self.rigs:
            await rig.close()

    async def rig(self, manager_kw=None, **model_kw):
        rig = Rig(**model_kw)
        self.rigs.append(rig)
        return await rig.start(**(manager_kw or {}))

    async def run_jobs(self, rig, jobs, rng):
        """Each job is its own ticket and connection; requests go out in a shuffled order."""
        async def one(i, job):
            ticket = 'T%d' % i
            client = await rig.client(ticket)
            order = list(job)
            rng.shuffle(order)
            results = await asyncio.gather(*(ask(client, ticket, req) for req in order))
            return ticket, results
        return await asyncio.gather(*(one(i, job) for i, job in enumerate(jobs)))

    async def test_each_key_reaches_the_model_once_under_concurrency(self):
        """T4: N concurrent jobs with overlapping keys; each key is generated exactly once."""
        rng = random.Random(4)
        rig = await self.rig(latency=lambda k, a: rng.random() * 0.02)
        outcomes = await self.run_jobs(rig, job_set(rng), rng)
        by_key = {}
        for ticket, results in outcomes:
            for res in results:
                by_key.setdefault(res['key_hash'], []).append(res)
        self.assertTrue(all(n == 1 for n in rig.model.calls.values()))
        self.assertEqual(set(rig.model.calls), set(by_key))
        for kh, results in by_key.items():
            self.assertEqual(len({r['sample']['content'] for r in results}), 1)
            self.assertEqual(sum(r['source'] == 'generated' for r in results), 1)
            self.assertEqual(len({r['generated_by'] for r in results}), 1)
        self.assertEqual(rig.store.count(), len(by_key))

    async def test_interleavings_give_identical_calls_and_samples(self):
        """T5: the same jobs under different schedules and caps: same calls, same samples."""
        seen = []
        for seed, cap in ((5, 1), (6, 3), (7, 64)):
            rng = random.Random(seed)
            jobs = job_set(random.Random(99))               # identical jobs every time
            rig = await self.rig(manager_kw={'inflight_cap': cap},
                                 latency=lambda k, a, r=rng: r.random() * 0.01)
            outcomes = await self.run_jobs(rig, jobs, rng)
            samples = {res['key_hash']: res['sample']['content']
                       for _, results in outcomes for res in results}
            seen.append((sum(rig.model.calls.values()), samples))
        self.assertEqual(len({calls for calls, _ in seen}), 1)
        self.assertEqual(seen[0][0], len(seen[0][1]))
        self.assertTrue(all(samples == seen[0][1] for _, samples in seen))

    async def test_check_and_create_are_atomic(self):
        rig = await self.rig(latency=0.05)
        tickets = ['T%d' % i for i in range(50)]
        client = await rig.client(*tickets)
        req = request('c0', 'draft', 1)
        results = await asyncio.gather(*(ask(client, t, req) for t in tickets))
        self.assertEqual(sum(rig.model.calls.values()), 1)
        self.assertEqual(sorted(r['source'] for r in results), ['generated'] + ['waited'] * 49)

    async def test_nothing_is_delivered_before_it_is_durable(self):
        rig = await self.rig(latency=lambda k, a: random.random() * 0.01)
        client = await rig.client('A', 'B')
        reqs = [request('c1', 'draft', r) for r in range(1, 30)]

        async def checked(ticket, req):
            res = await ask(client, ticket, req)
            self.assertIsNotNone(rig.store.get(res['key_hash']))   # already committed
            return res
        await asyncio.gather(*(checked(t, q) for q in reqs for t in ('A', 'B')))

    async def test_a_killed_client_loses_nothing(self):
        """T6 for manager-owned generation: the flight finishes and a restart hits it."""
        rig = await self.rig(latency=1.0)
        await rig.client('A')
        req = {'id': 1, 'op': 'get_or_generate', 'ticket': 'A', 'case': 'c0', 'node': 'draft',
               'rep': 1, 'sample': 0, 'body': body()}
        code = ('import json, socket, sys, time\n'
                's = socket.socket(socket.AF_UNIX); s.connect(sys.argv[1])\n'
                's.sendall((sys.argv[2] + "\\n").encode()); time.sleep(60)\n')
        proc = subprocess.Popen([sys.executable, '-c', code, rig.socket, json.dumps(req)])
        try:
            for _ in range(200):
                if rig.manager.inflight:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(rig.manager.inflight, 1)
            proc.kill()
        finally:
            proc.wait()
        await asyncio.wait_for(rig.manager.drain(), 10)
        self.assertEqual(rig.store.count(), 1)
        restarted = await rig.client('A')                   # same ticket, as a restart would use
        res = await ask(restarted, 'A', request('c0', 'draft', 1))
        self.assertEqual((res['source'], res['generated_by']), ('existing', 'A'))
        self.assertEqual(sum(rig.model.calls.values()), 1)

    async def test_retried_transport_failures_are_written_once(self):
        rig = await self.rig(manager_kw={'max_attempts': 4},
                             fault=lambda k, a: 'transport' if a < 3 else None)
        client = await rig.client('A')
        res = await ask(client, 'A', request('c0', 'draft', 1))
        self.assertEqual((res['source'], res['attempts']), ('generated', 3))
        self.assertEqual(sum(rig.model.calls.values()), 3)
        self.assertEqual(rig.store.count(), 1)

    async def test_request_errors_are_samples(self):
        rig = await self.rig(fault='request')
        client = await rig.client('A', 'B')
        first = await ask(client, 'A', request('c0', 'draft', 1))
        again = await ask(client, 'B', request('c0', 'draft', 1))
        self.assertEqual(first['sample']['error'], 'context_overflow')
        self.assertEqual((again['source'], again['sample']), ('existing', first['sample']))
        self.assertEqual(sum(rig.model.calls.values()), 1)

    async def test_exhausted_retries_fail_every_waiter_and_write_nothing(self):
        rig = await self.rig(manager_kw={'max_attempts': 3}, latency=0.02, fault='transport')
        client = await rig.client('A', 'B')
        req = request('c0', 'draft', 1)
        results = await asyncio.gather(ask(client, 'A', req), ask(client, 'B', req),
                                       return_exceptions=True)
        self.assertTrue(all(isinstance(r, ManagerError) and r.kind == 'flight_failed'
                            for r in results))
        self.assertEqual(rig.store.count(), 0)
        self.assertEqual(sum(rig.model.calls.values()), 3)
        rig.model.fault = None                              # the next request opens a new flight
        res = await ask(client, 'A', req)
        self.assertEqual(res['source'], 'generated')
        self.assertEqual(sum(rig.model.calls.values()), 4)

    async def test_malformed_samples_are_retried_and_unclassified_errors_are_not(self):
        rig = await self.rig(manager_kw={'max_attempts': 3},
                             fault=lambda k, a: 'malformed' if a == 1 else None)
        client = await rig.client('A')
        res = await ask(client, 'A', request('c0', 'draft', 1))
        self.assertEqual(res['attempts'], 2)
        rig.model.fault = 'crash'
        with self.assertRaises(ManagerError) as ctx:
            await ask(client, 'A', request('c0', 'draft', 2))
        self.assertEqual(ctx.exception.kind, 'model_client_error')
        self.assertEqual(rig.store.count(), 1)

    async def test_cancel_drops_unsent_transfers_awaited_and_bills_sent(self):
        sent = []                                   # the first two flights hold the only slot
        rig = await self.rig(manager_kw={'inflight_cap': 1},
                             latency=lambda k, a: (sent.append(k), 0.4 if len(sent) <= 2 else 0.02)[1])
        m = rig.manager
        client = await rig.client('X', 'A', 'B')
        slow = asyncio.ensure_future(ask(client, 'X', request('c3', 'draft', 1)))
        await until(lambda: m.inflight == 1)
        a_sent = asyncio.ensure_future(ask(client, 'A', request('c0', 'draft', 1)))
        await until(lambda: len(m.queued) == 1)
        await slow                                  # now A's first flight is the one in the slot
        await until(lambda: m.inflight == 1 and not m.queued)
        a_q = [asyncio.ensure_future(ask(client, 'A', request('c0', 'draft', r)))
               for r in (2, 3, 4)]
        await until(lambda: len(m.queued) == 3)
        b_wait = asyncio.ensure_future(ask(client, 'B', request('c0', 'draft', 4)))
        await until(lambda: any(len(set(f.waiters)) == 2 for f in m.queued))
        summary = await client.call('cancel', ticket='A')
        self.assertEqual(summary, {'dropped': 2, 'transferred': 1, 'sent': 1})
        results = await asyncio.gather(*a_q, return_exceptions=True)
        self.assertEqual(sum(isinstance(r, ManagerError) and r.kind == 'cancelled'
                             for r in results), 2)
        b_res = await b_wait
        self.assertEqual((b_res['source'], b_res['generated_by']), ('generated', 'B'))
        await a_sent
        status = await client.call('ticket_status', ticket='A')
        self.assertEqual(status['generated'], {'draft': 1})   # the sent call is billed to A
        with self.assertRaises(ManagerError) as ctx:
            await ask(client, 'A', request('c0', 'draft', 9))
        self.assertEqual(ctx.exception.kind, 'cancelled')

    async def test_dispatch_prefers_waiters_then_round_robin_then_arrival(self):
        order = []                                  # the first flight holds the only slot
        rig = await self.rig(manager_kw={'inflight_cap': 1},
                             latency=lambda k, a: (order.append(k), 0.4 if len(order) == 1 else 0.01)[1])
        m = rig.manager
        client = await rig.client('X', 'A', 'B', 'C')
        block = asyncio.ensure_future(ask(client, 'X', request('c3', 'draft', 1)))
        await until(lambda: m.inflight == 1)
        queued = []
        for ticket, req in (('A', request('c0', 'draft', 1)), ('A', request('c0', 'draft', 2)),
                            ('B', request('c1', 'draft', 1)), ('A', request('c2', 'draft', 1))):
            queued.append(asyncio.ensure_future(ask(client, ticket, req)))
            await until(lambda n=len(queued): len(m.queued) == n)   # fixes the arrival order
        extra = [asyncio.ensure_future(ask(client, t, request('c2', 'draft', 1)))
                 for t in ('B', 'C')]
        await until(lambda: any(len(set(f.waiters)) == 3 for f in m.queued))
        self.assertEqual(m.inflight, 1)             # still blocked: nothing reordered yet
        done = await asyncio.gather(block, *queued, *extra)
        k = [r['key_hash'] for r in done[:5]]       # blocker, A1, A2, B1, A-shared
        # most waiters first (the key A, B and C await), then B (never served) before A,
        # then A's two by arrival
        self.assertEqual(order, [k[0], k[4], k[3], k[1], k[2]])

    async def test_one_job_attempt_cannot_ask_twice_for_a_key(self):
        """Two draws of one job sharing a sample would break independence: refused, not served."""
        rig = await self.rig()
        client = await rig.client('A', 'B')
        req = request('c0', 'draft', 1)
        await ask(client, 'A', req)
        await ask(client, 'B', req)                          # another ticket may share it
        with self.assertRaises(ManagerError) as ctx:
            await ask(client, 'A', req)
        self.assertEqual(ctx.exception.kind, 'duplicate_key')
        restarted = await rig.client()                       # a restart is a new attempt
        res = await ask(restarted, 'A', req)
        self.assertEqual(res['source'], 'existing')

    async def test_ticket_rules(self):
        rig = await self.rig()
        client = await rig.client('A')
        with self.assertRaises(ManagerError) as ctx:
            await client.call('register_ticket', ticket='Z', generation='exec/0',
                              model_rev='some-other-model', case_keys=CASES)
        self.assertEqual(ctx.exception.kind, 'model_mismatch')
        with self.assertRaises(ManagerError) as ctx:
            await client.call('register_ticket', ticket='A', generation='exec/1',
                              model_rev=rig.model.model_rev, case_keys=CASES)
        self.assertEqual(ctx.exception.kind, 'ticket_conflict')
        await client.call('register_ticket', ticket='A', generation='exec/0',
                          model_rev=rig.model.model_rev, case_keys=CASES)   # idempotent
        with self.assertRaises(ManagerError) as ctx:
            await client.call('register_ticket', ticket='Q', generation='exec',
                              model_rev=rig.model.model_rev, case_keys=CASES)
        self.assertEqual(ctx.exception.kind, 'bad_request')
        for ticket, case, kind in (('nobody', 'c0', 'unknown_ticket'), ('A', 'zz', 'unknown_case')):
            with self.assertRaises(ManagerError) as ctx:
                await ask(client, ticket, request(case, 'draft', 1))
            self.assertEqual(ctx.exception.kind, kind)

    async def test_generations_never_share_stochastic_draws(self):
        rig = await self.rig()
        client = await rig.client('A')
        await client.call('register_ticket', ticket='F', generation='fresh/F',
                          model_rev=rig.model.model_rev, case_keys=CASES)
        a = await ask(client, 'A', request('c0', 'draft', 1))
        f = await ask(client, 'F', request('c0', 'draft', 1))
        self.assertNotEqual(a['key_hash'], f['key_hash'])
        self.assertEqual((a['source'], f['source']), ('generated', 'generated'))

    async def test_lookup_uses_the_same_key_as_execution(self):
        rig = await self.rig()
        client = await rig.client('A')
        probe = dict(generation='exec/0', model_rev=rig.model.model_rev, case_key=CASES['c0'],
                     node='draft', rep=1, sample=0, body=body())
        self.assertIsNone((await client.call('lookup', **probe))['row'])
        res = await ask(client, 'A', request('c0', 'draft', 1))
        row = (await client.call('lookup', **probe))['row']
        self.assertEqual(row['key_hash'], res['key_hash'])
        self.assertEqual(sum(rig.model.calls.values()), 1)

    async def test_waiters_are_not_billed(self):
        rig = await self.rig(latency=0.05)
        client = await rig.client('A', 'B')
        req = request('c0', 'draft', 1)
        await asyncio.gather(ask(client, 'A', req), ask(client, 'B', req))
        a = await client.call('ticket_status', ticket='A')
        b = await client.call('ticket_status', ticket='B')
        self.assertEqual(sorted([a['generated'], b['generated']], key=len), [{}, {'draft': 1}])


class ManagerProcessTest(unittest.IsolatedAsyncioTestCase):
    """The manager as its own process: a crash writes nothing partial; one manager per store."""

    async def asyncSetUp(self):
        self.dir = tempfile.mkdtemp(prefix='evalrt-')
        self.store = os.path.join(self.dir, 'entries.db')
        self.procs = []

    async def asyncTearDown(self):
        for proc in self.procs:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        shutil.rmtree(self.dir, ignore_errors=True)

    async def launch(self, socket_name, latency):
        env = dict(os.environ, PYTHONPATH=SRC)
        proc = await asyncio.create_subprocess_exec(
            sys.executable, '-m', 'copula.kernel.manager', '--store', self.store,
            '--socket', os.path.join(self.dir, socket_name), '--mock', 'proc',
            '--latency', str(latency), env=env, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        self.procs.append(proc)
        return proc

    async def ready(self, proc):
        line = await asyncio.wait_for(proc.stdout.readline(), 30)
        return json.loads(line)

    async def test_a_killed_manager_writes_nothing_and_the_store_reopens(self):
        proc = await self.launch('m.sock', latency=2.0)
        info = await self.ready(proc)
        client = await AsyncClient.connect(info['socket'])
        await client.call('register_ticket', ticket='A', generation='exec/0',
                          model_rev=info['model_rev'], case_keys=CASES)
        pending = asyncio.ensure_future(client.get_or_generate('A', 'c0', 'draft', 1, 0, body()))
        await asyncio.sleep(0.3)
        proc.send_signal(signal.SIGKILL)
        await proc.wait()
        with self.assertRaises(ManagerError) as ctx:
            await asyncio.wait_for(pending, 10)
        self.assertEqual(ctx.exception.kind, 'disconnected')
        await client.close()

        again = await self.launch('m.sock', latency=0.0)    # the stale socket is reclaimed
        info = await self.ready(again)
        client = await AsyncClient.connect(info['socket'])
        await client.call('register_ticket', ticket='A', generation='exec/0',
                          model_rev=info['model_rev'], case_keys=CASES)
        res = await client.get_or_generate('A', 'c0', 'draft', 1, 0, body())
        self.assertEqual(res['source'], 'generated')        # nothing was written before the kill
        self.assertEqual((await client.call('stats'))['entries'], 1)
        await client.close()
        again.send_signal(signal.SIGTERM)
        self.assertEqual(await asyncio.wait_for(again.wait(), 30), 0)

    async def test_a_second_manager_on_the_same_store_refuses_to_start(self):
        first = await self.launch('a.sock', latency=0.0)
        await self.ready(first)
        second = await self.launch('b.sock', latency=0.0)
        code = await asyncio.wait_for(second.wait(), 30)
        self.assertNotEqual(code, 0)
        err = (await second.stderr.read()).decode()
        self.assertIn('cannot open store', err)
        first.send_signal(signal.SIGTERM)
        self.assertEqual(await asyncio.wait_for(first.wait(), 30), 0)


if __name__ == '__main__':
    unittest.main()
