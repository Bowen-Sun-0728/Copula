"""Cross-process single flight for cached tools (manager leases, 2026-10-13). Mock models, no GPU:
    PYTHONPATH=src:. python3 -m unittest tests.kernel.test_evalrt_leases
Two sessions are two connections, as two job processes would be; one test kills a real process.
"""
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from copula.kernel.core.canon import digest
from copula.kernel.sdk import Session
from copula.kernel.testing.rig import Rig

EXECUTED = []                       # (tool, argument) per execution, in this process


async def slow_upper(text):
    """In a child started by the kill test it signals and hangs; here it is a slow upper()."""
    started = os.environ.get('LEASE_CHILD_STARTED')
    if started:
        open(started, 'w').close()
        await asyncio.sleep(120)
    EXECUTED.append(('slow_upper', text))
    await asyncio.sleep(0.2)
    return text.upper()


def slow_sync(x):
    EXECUTED.append(('slow_sync', x))
    time.sleep(0.4)
    return x * 2


async def dice(sides, rng):
    EXECUTED.append(('dice', sides))
    await asyncio.sleep(0.2)
    return rng.randrange(sides)


def register(s):
    return (s.tool(version=1, randomness='deterministic', cache=True)(slow_upper),
            s.tool(version=1, randomness='stochastic', cache=True)(dice))


def statuses(trace):
    with open(trace) as f:
        return [r['status'] for r in map(json.loads, f) if r.get('ev') == 'call']


async def _child(sock):
    s = await Session.connect(sock)
    upper, _ = register(s)
    async with s.run(s.group('k')):
        await upper('killed')


class ManagerLeaseTest(unittest.IsolatedAsyncioTestCase):
    """The ops themselves, on raw connections."""

    async def asyncSetUp(self):
        self.rig = await Rig('lease').start()
        self.a = await self.rig.client('ta', case_keys={'c': 'kc'})
        self.b = await self.rig.client('tb', case_keys={'c': 'kc'})

    async def asyncTearDown(self):
        await self.rig.close()

    def get(self, client, ticket, ident, **kw):
        return client.call('memo_get', ticket=ticket, case='c', identity=ident, **kw)

    async def pending(self, coro):
        task = asyncio.ensure_future(coro)
        await asyncio.sleep(0.1)
        self.assertFalse(task.done())                   # it waits on the lease
        return task

    async def test_put_wakes_waiters(self):
        ident = digest({'t': 'put'})
        first = await self.get(self.a, 'ta', ident, lease=True)
        self.assertIsNone(first['value'])
        self.assertTrue(first['lease'])
        waiter = await self.pending(self.get(self.b, 'tb', ident, lease=True))
        plain = await self.get(self.b, 'tb', ident)     # a requester without leases never waits
        self.assertIsNone(plain['value'])
        await self.a.call('memo_put', ticket='ta', case='c', identity=ident, value='"V"',
                          kind='k', kind_rev='r')
        got = await asyncio.wait_for(waiter, 2)
        self.assertEqual((got['value'], got['ticket'], got['lease'], got['waited']),
                         ('"V"', 'ta', None, True))
        again = await self.get(self.b, 'tb', ident, lease=True)
        self.assertEqual((again['value'], again['waited']), ('"V"', False))
        stats = await self.a.call('stats')
        self.assertEqual((stats['leases'], stats['leases_held']), (1, 0))

    async def test_release_hands_over(self):
        ident = digest({'t': 'release'})
        first = await self.get(self.a, 'ta', ident, lease=True)
        waiter = await self.pending(self.get(self.b, 'tb', ident, lease=True))
        self.assertEqual(await self.b.call('lease_release', lease=first['lease']),
                         {'released': False})                        # not b's lease
        self.assertEqual(await self.a.call('lease_release', lease=first['lease']),
                         {'released': True})
        got = await asyncio.wait_for(waiter, 2)
        self.assertIsNone(got['value'])
        self.assertTrue(got['waited'])
        self.assertNotEqual(got['lease'], first['lease'])            # the waiter holds it now
        self.assertEqual(await self.a.call('lease_release', lease=first['lease']),
                         {'released': False})                        # an old token is inert
        third = await self.pending(self.get(self.a, 'ta', ident, lease=True))
        await self.b.call('memo_put', ticket='tb', case='c', identity=ident, value='"W"',
                          kind='k', kind_rev='r')
        self.assertEqual((await asyncio.wait_for(third, 2))['value'], '"W"')

    async def test_closed_connection_releases(self):
        ident = digest({'t': 'close'})
        await self.get(self.a, 'ta', ident, lease=True)
        waiter = await self.pending(self.get(self.b, 'tb', ident, lease=True))
        await self.a.close()
        got = await asyncio.wait_for(waiter, 2)
        self.assertIsNone(got['value'])
        self.assertTrue(got['lease'])
        stats = await self.b.call('stats')
        self.assertEqual((stats['leases_dropped'], stats['leases_held']), (1, 1))


class SDKLeaseTest(unittest.IsolatedAsyncioTestCase):
    """Two sessions calling the same cached tool at the same time."""

    async def asyncSetUp(self):
        EXECUTED.clear()
        self.rig = await Rig('sdk-lease').start()
        self.dir = tempfile.mkdtemp(prefix='evalrt-lease-')
        self.traces = [os.path.join(self.dir, 't%d.jsonl' % i) for i in range(2)]
        self.s = [await Session.connect(self.rig.socket, trace=t) for t in self.traces]
        self.tools = [register(s) for s in self.s]

    async def asyncTearDown(self):
        for s in self.s:
            await s.aclose()
        await self.rig.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    async def both(self, call, groups=('g', 'g')):
        async def one(i):
            s = self.s[i]
            async with s.run(s.group(groups[i])):
                return await call(i)
        return await asyncio.gather(one(0), one(1))

    def all_statuses(self):
        return sorted(statuses(self.traces[0]) + statuses(self.traces[1]))

    async def test_deterministic_tool_executes_once(self):
        got = await self.both(lambda i: self.tools[i][0]('ab'), groups=('g', 'h'))
        self.assertEqual(got, ['AB', 'AB'])
        self.assertEqual(EXECUTED, [('slow_upper', 'ab')])
        self.assertEqual(self.all_statuses(), ['generated', 'lease_waited'])

    async def test_stochastic_tool_executes_once_per_group(self):
        same = await self.both(lambda i: self.tools[i][1](10 ** 9))
        self.assertEqual(same[0], same[1])
        self.assertEqual(EXECUTED, [('dice', 10 ** 9)])
        self.assertEqual(self.all_statuses(), ['generated', 'lease_waited'])
        EXECUTED.clear()
        other = await self.both(lambda i: self.tools[i][1](10 ** 9), groups=('x', 'y'))
        self.assertNotEqual(other[0], other[1])         # independent groups share nothing
        self.assertEqual(len(EXECUTED), 2)

    async def test_failure_hands_over(self):
        calls = []

        async def flaky(x):
            calls.append(x)
            await asyncio.sleep(0.2)
            if len(calls) == 1:
                raise ValueError('first execution fails')
            return x
        tools = [s.tool(version=1, randomness='deterministic', cache=True)(flaky)
                 for s in self.s]
        got = await asyncio.gather(*[self._in_run(i, tools[i](7), delay=0.05 * i)
                                     for i in range(2)], return_exceptions=True)
        self.assertIsInstance(got[0], ValueError)
        self.assertEqual(got[1], 7)                     # the waiter took the lease and computed it
        self.assertEqual(calls, [7, 7])

    async def test_cancel_hands_over(self):
        upper = [t[0] for t in self.tools]
        first = self._in_run(0, asyncio.wait_for(upper[0]('cd'), 0.05))
        second = self._in_run(1, upper[1]('cd'), delay=0.02)
        got = await asyncio.gather(first, second, return_exceptions=True)
        self.assertIsInstance(got[0], asyncio.TimeoutError)
        self.assertEqual(got[1], 'CD')
        await asyncio.sleep(0.05)
        self.assertEqual(self.rig.manager.stats()['leases_held'], 0)

    async def test_cancelled_waiter_leaves_no_lease(self):
        calls = []

        async def flaky2(x):
            calls.append(x)
            await asyncio.sleep(0.3)
            if len(calls) == 1:
                raise ValueError('first execution fails')
            return x
        tools = [s.tool(version=1, randomness='deterministic', cache=True)(flaky2)
                 for s in self.s]
        holder = self._in_run(0, tools[0](7))
        impatient = self._in_run(1, asyncio.wait_for(tools[1](7), 0.1), delay=0.05)
        patient = self._in_run(1, tools[1](7), delay=0.15)
        got = await asyncio.wait_for(asyncio.gather(holder, impatient, patient,
                                                    return_exceptions=True), 5)
        self.assertIsInstance(got[0], ValueError)
        self.assertIsInstance(got[1], asyncio.TimeoutError)
        self.assertEqual(got[2], 7)                     # not stuck behind a lease nobody holds
        self.assertEqual(calls, [7, 7])
        await asyncio.sleep(0.1)
        self.assertEqual(self.rig.manager.stats()['leases_held'], 0)

    async def test_failed_commit_releases_the_lease(self):
        store, manager = self.rig.store, self.rig.manager
        orig, armed = store.commit, [True]

        def commit():
            if armed[0] and manager._leases:
                armed[0] = False
                raise RuntimeError('injected commit failure')
            return orig()
        store.commit = commit
        with self.assertRaises(Exception):
            await self._in_run(0, self.tools[0][1](6))
        self.assertFalse(armed[0])
        self.assertEqual(manager.stats()['leases_held'], 0)
        self.assertIsInstance(await asyncio.wait_for(self._in_run(1, self.tools[1][1](6)), 5),
                              int)

    async def test_cancelled_sync_holder_still_writes(self):
        tools = [s.tool(version=1, randomness='deterministic', cache=True)(slow_sync)
                 for s in self.s]
        first = self._in_run(0, asyncio.wait_for(tools[0].aio(5), 0.1))
        second = self._in_run(1, tools[1].aio(5), delay=0.05)
        got = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 5)
        self.assertIsInstance(got[0], asyncio.TimeoutError)
        self.assertEqual(got[1], 10)
        self.assertEqual(EXECUTED, [('slow_sync', 5)])  # the waiter did not run it again
        self.assertIn('lease_waited', statuses(self.traces[1]))
        self.assertEqual(self.rig.manager.stats()['leases_held'], 0)
        await asyncio.sleep(0.1)
        with open(self.traces[0]) as f:                 # the late write is traced (generated once)
            late = [r for r in map(json.loads, f) if r.get('late')]
        self.assertEqual([(r['status'], r['node']) for r in late], [('generated', 'slow_sync')])

    async def test_cancelled_async_holder_releases_after_unwinding(self):
        marks = []

        async def tidy(x):
            marks.append(('start', time.time()))
            try:
                await asyncio.sleep(1.0)
                return x
            finally:
                await asyncio.sleep(0.2)                # cleanup after the cancel
                marks.append(('end', time.time()))
        tools = [s.tool(version=1, randomness='deterministic', cache=True)(tidy) for s in self.s]
        first = self._in_run(0, asyncio.wait_for(tools[0](3), 0.1))
        second = self._in_run(1, tools[1](3), delay=0.05)
        got = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 5)
        self.assertIsInstance(got[0], asyncio.TimeoutError)
        self.assertEqual(got[1], 3)
        self.assertEqual([m for m, _ in marks], ['start', 'end', 'start', 'end'])
        self.assertGreaterEqual(marks[2][1], marks[1][1])   # no overlap with the cleanup

    async def test_closing_session_waits_for_its_late_tool(self):
        holder = await Session.connect(self.rig.socket)
        h_tool = holder.tool(version=1, randomness='deterministic', cache=True)(slow_sync)
        w_tool = self.s[1].tool(version=1, randomness='deterministic', cache=True)(slow_sync)

        async def cancelled_then_closed():
            try:
                async with holder.run(holder.group('h')):
                    await asyncio.wait_for(h_tool.aio(9), 0.1)
            except asyncio.TimeoutError:
                pass
            await holder.aclose()                       # waits for the thread and its write
        got = await asyncio.wait_for(asyncio.gather(
            cancelled_then_closed(), self._in_run(1, w_tool.aio(9), delay=0.05)), 5)
        self.assertEqual(got[1], 18)
        self.assertEqual(EXECUTED, [('slow_sync', 9)])

    async def test_tool_exception_keeps_its_own_context(self):
        async def broken(x, rng):
            raise KeyError(x)
        tools = [s.tool(version=1, randomness='stochastic', cache=True)(broken) for s in self.s]

        async def det(x):
            raise KeyError(x)
        dets = [s.tool(version=1, randomness='deterministic', cache=True)(det) for s in self.s]
        for call in (tools[0], dets[0]):
            with self.assertRaises(KeyError) as ctx:
                await self._in_run(0, call(1))
            self.assertIsNone(ctx.exception.__context__)

    async def _in_run(self, i, coro, delay=0.0):
        s = self.s[i]
        await asyncio.sleep(delay)
        async with s.run(s.group('g%d' % i)):
            return await coro

    async def test_killed_holder_releases(self):
        started = os.path.join(self.dir, 'child-started')
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env = dict(os.environ, LEASE_CHILD_STARTED=started,
                   PYTHONPATH=os.pathsep.join([os.path.join(root, 'src'), root]))
        child = subprocess.Popen(
            [sys.executable, '-c', 'import asyncio, sys; from tests.kernel.test_evalrt_leases import '
             '_child; asyncio.run(_child(sys.argv[1]))', self.rig.socket], cwd=root, env=env)
        try:
            t0 = time.time()
            while not os.path.exists(started):
                self.assertIsNone(child.poll(), 'the child exited before holding the lease')
                self.assertLess(time.time() - t0, 60)
                await asyncio.sleep(0.05)
            waiter = asyncio.ensure_future(self._in_run(0, self.tools[0][0]('killed')))
            await asyncio.sleep(0.3)
            self.assertFalse(waiter.done())             # the child holds the lease
            os.kill(child.pid, signal.SIGKILL)
            self.assertEqual(await asyncio.wait_for(waiter, 10), 'KILLED')
            self.assertEqual(EXECUTED, [('slow_upper', 'killed')])
            self.assertEqual(self.rig.manager.stats()['leases_dropped'], 1)
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()


if __name__ == '__main__':
    unittest.main()
