"""Lease expiry (2026-10-13, evening): a renewing holder's lease expires after the manager's TTL
without a renewal, and `max_hold` (the holder's or a waiter's) bounds the wait on one execution.
Mock models, no GPU:
    PYTHONPATH=src:. python3 -m unittest tests.kernel.test_evalrt_lease_expiry
One test freezes a real holder process with SIGSTOP (its connection stays open).
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
from copula.kernel.errors import EvalrtError
from copula.kernel.manager.__main__ import main as manager_main
from copula.kernel.sdk import Session
from copula.kernel.testing.rig import Rig

EXECUTED = []                       # (tool, argument) per execution, in this process
TTL = 1.0


async def frozen_upper(text):
    """In the child of the freeze test it signals, then sleeps (it is frozen meanwhile)."""
    started = os.environ.get('LEASE_CHILD_STARTED')
    if started:
        open(started, 'w').close()
        await asyncio.sleep(120)
    EXECUTED.append(('frozen_upper', text))
    await asyncio.sleep(0.1)
    return text.upper()


async def _child(sock):
    s = await Session.connect(sock)
    upper = s.tool(version=1, randomness='deterministic', cache=True)(frozen_upper)
    async with s.run(s.group('k')):
        await upper('frozen')


def statuses(trace):
    with open(trace) as f:
        return [r['status'] for r in map(json.loads, f) if r.get('ev') == 'call']


class ManagerExpiryTest(unittest.IsolatedAsyncioTestCase):
    """The ops on raw connections."""

    async def asyncSetUp(self):
        self.rig = await Rig('expiry').start(lease_ttl=TTL)
        self.a = await self.rig.client('ta', case_keys={'c': 'kc'})
        self.b = await self.rig.client('tb', case_keys={'c': 'kc'})

    async def asyncTearDown(self):
        await self.rig.close()

    def get(self, client, ticket, ident, **kw):
        return client.call('memo_get', ticket=ticket, case='c', identity=ident, **kw)

    def put(self, client, ticket, ident, value):
        return client.call('memo_put', ticket=ticket, case='c', identity=ident, value=value,
                           kind='k', kind_rev='r')

    async def test_grant_reports_ttl_to_renewing_holders_only(self):
        got = await self.get(self.a, 'ta', digest({'t': 'ttl'}), lease=True, renew=True)
        self.assertEqual(got['ttl'], TTL)
        old = await self.get(self.a, 'ta', digest({'t': 'ttl-old'}), lease=True)
        self.assertIsNone(old['ttl'])
        self.assertEqual((await self.a.call('ping'))['leases']['ttl'], TTL)

    async def test_unrenewed_lease_expires(self):
        ident = digest({'t': 'expire'})
        first = await self.get(self.a, 'ta', ident, lease=True, renew=True)
        t0 = time.monotonic()
        got = await asyncio.wait_for(self.get(self.b, 'tb', ident, lease=True, renew=True), 10)
        waited = time.monotonic() - t0
        self.assertGreater(waited, TTL * 0.8)            # it waited for the lease to expire
        self.assertLess(waited, TTL + 1.5)
        self.assertEqual((got['value'], got['waited']), (None, True))
        self.assertNotEqual(got['lease'], first['lease'])
        self.assertEqual(await self.a.call('lease_renew', lease=first['lease']),
                         {'renewed': False})              # the expired holder learns it
        self.assertEqual(await self.a.call('lease_release', lease=first['lease']),
                         {'released': False})
        self.assertEqual(await self.put(self.a, 'ta', ident, '"late"'), {})   # first write
        self.assertEqual(await self.put(self.b, 'tb', ident, '"taker"'),      # lost: told
                         {'value': '"late"', 'ticket': 'ta'})                 # the stored one
        again = await self.get(self.b, 'tb', ident, lease=True)
        self.assertEqual(again['value'], '"late"')
        stats = await self.a.call('stats')
        self.assertEqual((stats['leases_expired'], stats['leases_held'],
                          stats['computed_lost_races']), (1, 0, 1))

    async def test_a_holder_that_does_not_renew_never_expires_by_ttl(self):
        ident = digest({'t': 'old-client'})
        first = await self.get(self.a, 'ta', ident, lease=True)       # no `renew`: an old SDK
        waiter = asyncio.ensure_future(self.get(self.b, 'tb', ident, lease=True, renew=True))
        await asyncio.sleep(2 * TTL)
        self.assertFalse(waiter.done())
        await self.a.call('lease_release', lease=first['lease'])
        self.assertTrue((await asyncio.wait_for(waiter, 2))['lease'])

    async def test_renewals_keep_the_lease(self):
        ident = digest({'t': 'renew'})
        first = await self.get(self.a, 'ta', ident, lease=True, renew=True)
        waiter = asyncio.ensure_future(self.get(self.b, 'tb', ident, lease=True, renew=True))
        for _ in range(8):                                # 2.4 s = 2.4 TTLs, renewed every 0.3 s
            await asyncio.sleep(0.3)
            self.assertEqual(await self.a.call('lease_renew', lease=first['lease']),
                             {'renewed': True})
            self.assertFalse(waiter.done())
        self.assertEqual(await self.b.call('lease_renew', lease=first['lease']),
                         {'renewed': False})              # not b's lease
        await self.put(self.a, 'ta', ident, '"V"')
        got = await asyncio.wait_for(waiter, 2)
        self.assertEqual((got['value'], got['waited']), ('"V"', True))
        stats = await self.a.call('stats')
        self.assertEqual((stats['lease_renewals'], stats.get('leases_expired', 0)), (8, 0))

    async def test_several_waiters_one_handover(self):
        ident = digest({'t': 'many'})
        clients = [self.b] + [await self.rig.client('t%d' % i, case_keys={'c': 'kc'})
                              for i in range(2)]
        tickets = ['tb', 't0', 't1']
        await self.get(self.a, 'ta', ident, lease=True, renew=True)
        waiters = [asyncio.ensure_future(self.get(c, t, ident, lease=True, renew=True))
                   for c, t in zip(clients, tickets)]
        done, pending = await asyncio.wait(waiters, timeout=TTL + 1.5,
                                           return_when=asyncio.FIRST_COMPLETED)
        self.assertEqual(len(done), 1)                    # exactly one took the lease over
        taker = list(done)[0]
        self.assertTrue(taker.result()['lease'])
        await asyncio.sleep(0.3)
        self.assertTrue(all(not w.done() for w in pending))   # the rest wait on the taker
        i = waiters.index(taker)
        await self.put(clients[i], tickets[i], ident, '"T"')
        rest = await asyncio.wait_for(asyncio.gather(*pending), 2)
        self.assertEqual([r['value'] for r in rest], ['"T"', '"T"'])
        stats = await self.a.call('stats')
        self.assertEqual((stats['leases_expired'], stats['lease_handovers']), (1, 1))

    async def test_max_hold_caps_the_renewals(self):
        ident = digest({'t': 'cap'})
        first = await self.get(self.a, 'ta', ident, lease=True, renew=True, max_hold=0.5)
        t0 = time.monotonic()
        waiter = asyncio.ensure_future(self.get(self.b, 'tb', ident, lease=True, renew=True))
        renewed = []
        while not waiter.done() and time.monotonic() - t0 < 3:
            await asyncio.sleep(0.1)
            renewed.append((await self.a.call('lease_renew', lease=first['lease']))['renewed'])
        got = await asyncio.wait_for(waiter, 1)
        self.assertLess(time.monotonic() - t0, 1.2)      # 0.5 s, not as long as renewals go on
        self.assertTrue(got['lease'])
        self.assertTrue(renewed[0])
        self.assertFalse(renewed[-1])

    async def test_a_waiters_max_hold_counts_too(self):
        ident = digest({'t': 'waiter-cap'})
        first = await self.get(self.a, 'ta', ident, lease=True, renew=True)   # no bound
        t0 = time.monotonic()
        waiter = asyncio.ensure_future(self.get(self.b, 'tb', ident, lease=True, renew=True,
                                                max_hold=0.5))
        while not waiter.done() and time.monotonic() - t0 < 3:
            await asyncio.sleep(0.1)
            await self.a.call('lease_renew', lease=first['lease'])
        got = await asyncio.wait_for(waiter, 1)
        self.assertLess(time.monotonic() - t0, 1.2)
        self.assertTrue(got['lease'])

    async def test_max_hold_starts_when_the_grant_leaves(self):
        manager, armed = self.rig.manager, [True]
        orig = manager.durable

        async def slow_durable():
            if armed[0]:
                armed[0] = False
                await asyncio.sleep(0.8)                  # the grant's commit is slow
            return await orig()
        manager.durable = slow_durable
        ident = digest({'t': 'slow-commit'})
        holder = asyncio.ensure_future(self.get(self.a, 'ta', ident, lease=True, renew=True,
                                                max_hold=0.5))
        await asyncio.sleep(0.05)
        waiter = asyncio.ensure_future(self.get(self.b, 'tb', ident, lease=True, renew=True))
        self.assertTrue((await asyncio.wait_for(holder, 3))['lease'])
        await asyncio.sleep(0.2)                          # 1.0 s after the grant was made,
        self.assertFalse(waiter.done())                   # 0.2 s after the holder learnt it
        self.assertTrue((await asyncio.wait_for(waiter, 2))['lease'])

    async def test_renew_and_release_do_not_wait_for_a_commit(self):
        ident = digest({'t': 'no-commit'})
        first = await self.get(self.a, 'ta', ident, lease=True, renew=True)
        manager = self.rig.manager
        orig = manager.durable

        async def failing_durable():                      # e.g. someone else's writes failed
            raise EvalrtError('store_failed', 'injected')
        manager.durable = failing_durable
        try:
            self.assertEqual(await self.a.call('lease_renew', lease=first['lease']),
                             {'renewed': True})
            self.assertEqual(await self.a.call('lease_release', lease=first['lease']),
                             {'released': True})
            with self.assertRaises(EvalrtError):          # a reply reporting stored state fails
                await self.get(self.b, 'tb', ident, lease=True, renew=True)
        finally:
            manager.durable = orig
        self.assertEqual(manager.stats()['leases_held'], 0)   # the failed grant was released

    async def test_a_waiters_bound_survives_a_handover(self):
        ident = digest({'t': 'handover-bound'})
        first = await self.get(self.a, 'ta', ident, lease=True, renew=True)
        x = await self.rig.client('tx', case_keys={'c': 'kc'})
        later = asyncio.ensure_future(self.get(x, 'tx', ident, lease=True, renew=True))
        await asyncio.sleep(0.05)                         # x waits first, so it wakes first
        bounded = asyncio.ensure_future(self.get(self.b, 'tb', ident, lease=True, renew=True,
                                                 max_hold=1.0))
        await asyncio.sleep(0.2)
        await self.a.call('lease_release', lease=first['lease'])
        second = await asyncio.wait_for(later, 2)         # x took over, and renews
        t0 = time.monotonic()
        while not bounded.done() and time.monotonic() - t0 < 4:
            await asyncio.sleep(0.2)
            await x.call('lease_renew', lease=second['lease'])
        self.assertTrue(bounded.done())                   # b's own bound applied to x's lease
        self.assertLess(time.monotonic() - t0, 1.8)
        self.assertTrue(bounded.result()['lease'])

    async def test_a_departed_waiters_bound_is_dropped(self):
        ident = digest({'t': 'departed'})
        first = await self.get(self.a, 'ta', ident, lease=True, renew=True)
        w = await self.rig.client('tw', case_keys={'c': 'kc'})
        gone = asyncio.ensure_future(self.get(w, 'tw', ident, lease=True, renew=True,
                                              max_hold=0.5))
        waiter = asyncio.ensure_future(self.get(self.b, 'tb', ident, lease=True, renew=True))
        await asyncio.sleep(0.1)
        await w.close()                                   # w's process exits
        await asyncio.gather(gone, return_exceptions=True)
        renewed = []
        for _ in range(6):                                # 1.2 s, past w's 0.5 s bound
            await asyncio.sleep(0.2)
            renewed.append((await self.a.call('lease_renew', lease=first['lease']))['renewed'])
        self.assertEqual(renewed, [True] * 6)
        self.assertFalse(waiter.done())
        await self.put(self.a, 'ta', ident, '"A"')
        self.assertEqual((await asyncio.wait_for(waiter, 2))['value'], '"A"')

    async def test_bad_max_hold_on_a_draw_claims_no_index(self):
        mrev = self.rig.model.model_rev
        await self.a.call('register_pool_ticket', ticket='tp', partitions=['exec/0'],
                          consumer='cons', model_rev=mrev, case_keys={'c': 'kc'})
        ident = digest({'t': 'draw'})
        with self.assertRaises(EvalrtError):
            await self.a.call('kind_draw', ticket='tp', case='c', partition='exec/0',
                              identity=ident, lease=True, renew=True, max_hold=-1)
        got = await self.a.call('kind_draw', ticket='tp', case='c', partition='exec/0',
                                identity=ident, lease=True, renew=True)
        self.assertEqual(got['index'], 1)                 # the refused request took no number

    async def test_bad_max_hold_is_refused(self):
        ident = digest({'t': 'bad'})
        for bad in (0, -1, 'x', True, [1]):
            with self.assertRaises(EvalrtError):
                await self.get(self.a, 'ta', ident, lease=True, max_hold=bad)
        self.assertEqual((await self.a.call('stats')).get('leases', 0), 0)


class NoExpiryTest(unittest.IsolatedAsyncioTestCase):
    async def test_ttl_none_keeps_old_behaviour(self):
        rig = await Rig('noexpiry').start(lease_ttl=None)
        try:
            a = await rig.client('ta', case_keys={'c': 'kc'})
            b = await rig.client('tb', case_keys={'c': 'kc'})
            ident = digest({'t': 'none'})
            first = await a.call('memo_get', ticket='ta', case='c', identity=ident, lease=True,
                                 renew=True)
            self.assertIsNone(first['ttl'])
            waiter = asyncio.ensure_future(b.call('memo_get', ticket='tb', case='c',
                                                  identity=ident, lease=True))
            await asyncio.sleep(1.0)
            self.assertFalse(waiter.done())               # no expiry: it waits for the holder
            await a.call('lease_release', lease=first['lease'])
            self.assertTrue((await asyncio.wait_for(waiter, 2))['lease'])
        finally:
            await rig.close()

    def test_cli_rejects_a_bad_ttl(self):
        for bad in ('0.5', '4', '-1', 'inf', 'nan'):
            with self.assertRaises(SystemExit):
                manager_main(['--store', '/nonexistent/x.db', '--socket', '/nonexistent/s',
                              '--mock', 'n', '--lease-ttl', bad])


class SDKExpiryTest(unittest.IsolatedAsyncioTestCase):
    """Sessions against a manager with a short TTL."""

    async def asyncSetUp(self):
        EXECUTED.clear()
        self.rig = await Rig('sdk-expiry').start(lease_ttl=TTL)
        self.dir = tempfile.mkdtemp(prefix='evalrt-expiry-')
        self.traces = [os.path.join(self.dir, 't%d.jsonl' % i) for i in range(2)]
        self.s = [await Session.connect(self.rig.socket, trace=t) for t in self.traces]

    async def asyncTearDown(self):
        for s in self.s:
            await s.aclose()
        await self.rig.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    async def _in_run(self, i, coro, delay=0.0):
        s = self.s[i]
        await asyncio.sleep(delay)
        async with s.run(s.group('g%d' % i)):
            return await coro

    async def test_long_tool_is_renewed_and_runs_once(self):
        def long_sync(x):
            EXECUTED.append(('long_sync', x))
            time.sleep(3 * TTL)                           # well past the TTL
            return x + 1
        tools = [s.tool(version=1, randomness='deterministic', cache=True)(long_sync)
                 for s in self.s]
        got = await asyncio.wait_for(asyncio.gather(
            self._in_run(0, tools[0].aio(1)), self._in_run(1, tools[1].aio(1), delay=0.1)), 15)
        self.assertEqual(got, [2, 2])
        self.assertEqual(EXECUTED, [('long_sync', 1)])
        stats = self.rig.manager.stats()
        self.assertGreaterEqual(stats['lease_renewals'], 6)
        self.assertEqual((stats.get('leases_expired', 0), stats['leases_held']), (0, 0))
        self.assertEqual(sorted(statuses(self.traces[0]) + statuses(self.traces[1])),
                         ['generated', 'lease_waited'])
        self.assertEqual(self.s[0]._renewals, {})         # renewal stopped with the write

    async def test_a_renewal_error_is_not_the_end(self):
        def long_sync2(x):
            EXECUTED.append(('long_sync2', x))
            time.sleep(2.5 * TTL)
            return x
        holder = self.s[0]
        orig, failures = holder.manager.call, []

        async def flaky_call(op, **kw):
            if op == 'lease_renew' and not failures:
                failures.append(op)
                raise EvalrtError('store_failed', 'injected')
            return await orig(op, **kw)
        holder.manager.call = flaky_call
        tools = [s.tool(version=1, randomness='deterministic', cache=True)(long_sync2)
                 for s in self.s]
        got = await asyncio.wait_for(asyncio.gather(
            self._in_run(0, tools[0].aio(6)), self._in_run(1, tools[1].aio(6), delay=0.1)), 15)
        self.assertEqual(got, [6, 6])
        self.assertEqual(failures, ['lease_renew'])
        self.assertEqual(EXECUTED, [('long_sync2', 6)])   # still renewed after the error
        self.assertEqual(self.rig.manager.stats().get('leases_expired', 0), 0)

    async def test_max_hold_bounds_a_hung_execution_across_processes(self):
        async def hangs_once(x):
            EXECUTED.append(('hangs_once', x))
            if len(EXECUTED) == 1:
                await asyncio.sleep(2.0)                  # the first execution "hangs"
            return x * 3
        tools = [s.tool(version=1, randomness='deterministic', cache=True, max_hold=0.3)(
            hangs_once) for s in self.s]
        t0 = time.monotonic()
        first = asyncio.ensure_future(self._in_run(0, tools[0](5)))
        second = await asyncio.wait_for(self._in_run(1, tools[1](5), delay=0.05), 5)
        self.assertEqual(second, 15)
        self.assertLess(time.monotonic() - t0, 1.5)      # not held up by the 2 s execution
        self.assertEqual(await asyncio.wait_for(first, 5), 15)
        self.assertEqual(len(EXECUTED), 2)
        self.assertEqual(self.rig.manager.stats()['leases_expired'], 1)
        self.assertEqual(statuses(self.traces[1]), ['generated'])
        self.assertEqual(statuses(self.traces[0]), ['duplicate'])     # its write came second

    async def test_everyone_gets_the_value_stored_first(self):
        """A late holder's write lands while the taker still computes: the taker's lease ends,
        its renewal stops, its own result is dropped, and both callers get the stored value."""
        async def who(x):                                 # breaks the purity rule on purpose
            EXECUTED.append(('who', x))
            if len(EXECUTED) == 1:
                await asyncio.sleep(1.0)                  # the holder: late, writes first
                return 'holder'
            await asyncio.sleep(2.5 * TTL)                # the taker: renews, writes second
            return 'taker'
        tools = [s.tool(version=1, randomness='deterministic', cache=True, max_hold=0.3)(who)
                 for s in self.s]
        got = await asyncio.wait_for(asyncio.gather(
            self._in_run(0, tools[0](1)), self._in_run(1, tools[1](1), delay=0.05)), 15)
        self.assertEqual(got, ['holder', 'holder'])
        self.assertEqual(statuses(self.traces[0]), ['generated'])
        self.assertEqual(statuses(self.traces[1]), ['duplicate'])
        self.assertEqual((self.s[0]._renewals, self.s[1]._renewals), ({}, {}))
        stats = self.rig.manager.stats()
        self.assertEqual((stats['leases_held'], stats['computed_lost_races']), (0, 1))

    async def test_max_hold_stochastic_value_is_the_same_whoever_computes(self):
        async def slow_draw(sides, rng):
            EXECUTED.append(('slow_draw', sides))
            if len(EXECUTED) == 1:
                await asyncio.sleep(1.5)
            return rng.randrange(sides)
        tools = [s.tool(version=1, randomness='stochastic', cache=True, max_hold=0.3)(slow_draw)
                 for s in self.s]

        async def in_group(i, delay):
            s = self.s[i]
            await asyncio.sleep(delay)
            async with s.run(s.group('shared')):
                return await tools[i](10 ** 9)
        got = await asyncio.wait_for(asyncio.gather(in_group(0, 0), in_group(1, 0.05)), 10)
        self.assertEqual(got[0], got[1])                  # one address, one seed, one value
        self.assertEqual(len(EXECUTED), 2)

    async def test_max_hold_bounds_waiters_in_the_same_process(self):
        async def hangs_here(x):
            EXECUTED.append(('hangs_here', x))
            if len(EXECUTED) == 1:
                await asyncio.sleep(3.0)
            return -x
        tool = self.s[0].tool(version=1, randomness='deterministic', cache=True,
                              max_hold=0.3)(hangs_here)
        s = self.s[0]

        async def call(delay):
            await asyncio.sleep(delay)
            async with s.run(s.group('one')):
                return await tool(4)
        t0 = time.monotonic()
        first = asyncio.ensure_future(call(0))
        second = await asyncio.wait_for(call(0.05), 5)
        self.assertEqual(second, -4)
        self.assertLess(time.monotonic() - t0, 1.5)
        t1 = time.monotonic()                             # the holder still hangs: later calls
        third = await asyncio.wait_for(call(0), 5)        # read the settled value at once
        self.assertEqual(third, -4)
        self.assertLess(time.monotonic() - t1, 0.2)
        self.assertEqual(await asyncio.wait_for(first, 5), -4)
        self.assertEqual(len(EXECUTED), 2)
        self.assertEqual(sorted(statuses(self.traces[0])), ['duplicate', 'generated', 'waited'])

    async def test_a_hung_holders_failure_still_gets_the_stored_value(self):
        async def hangs_then_fails(x):
            EXECUTED.append(('hangs_then_fails', x))
            if len(EXECUTED) == 1:
                await asyncio.sleep(1.5)
                raise TimeoutError('the hung execution finally gives up')
            return x + 100
        tool = self.s[0].tool(version=1, randomness='deterministic', cache=True,
                              max_hold=0.3)(hangs_then_fails)
        s = self.s[0]

        async def call(delay):
            await asyncio.sleep(delay)
            async with s.run(s.group('r3')):
                return await tool(1)
        got = await asyncio.wait_for(asyncio.gather(call(0), call(0.05)), 10)
        self.assertEqual(got, [101, 101])                 # the hung call's caller gets it too
        self.assertEqual(len(EXECUTED), 2)
        self.assertEqual(sorted(statuses(self.traces[0])), ['duplicate', 'generated'])  # it ran

    async def test_a_cancelled_calls_late_write_that_loses_is_traced_duplicate(self):
        def slow_first(x):
            EXECUTED.append(('slow_first', x))
            if len(EXECUTED) == 1:
                time.sleep(1.2)                           # still running after its call is cancelled
            return x * 2
        tools = [s.tool(version=1, randomness='deterministic', cache=True, max_hold=0.3)(
            slow_first) for s in self.s]
        first = self._in_run(0, asyncio.wait_for(tools[0].aio(7), 0.1))
        second = self._in_run(1, tools[1].aio(7), delay=0.05)
        got = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 10)
        self.assertIsInstance(got[0], asyncio.TimeoutError)
        self.assertEqual(got[1], 14)
        await asyncio.sleep(1.5)                          # the thread ends, writes, loses
        with open(self.traces[0]) as f:
            late = [r for r in map(json.loads, f) if r.get('late')]
        self.assertEqual([(r['status'], r['node']) for r in late], [('duplicate', 'slow_first')])
        self.assertEqual(self.s[0]._renewals, {})

    async def test_without_max_hold_waiters_wait(self):
        async def slowish(x):
            EXECUTED.append(('slowish', x))
            await asyncio.sleep(2 * TTL)                  # past the TTL; renewed, so no expiry
            return x
        tools = [s.tool(version=1, randomness='deterministic', cache=True)(slowish)
                 for s in self.s]
        got = await asyncio.wait_for(asyncio.gather(
            self._in_run(0, tools[0](8)), self._in_run(1, tools[1](8), delay=0.05)), 10)
        self.assertEqual(got, [8, 8])
        self.assertEqual(len(EXECUTED), 1)

    async def test_bad_max_hold_is_refused_at_registration(self):
        for i, bad in enumerate((0, -2, 'soon', True)):
            with self.assertRaises(EvalrtError):
                self.s[0].tool(name='bad_%d' % i, version=1, randomness='deterministic',
                               cache=True, max_hold=bad)(lambda x: x)
        with self.assertRaises(EvalrtError):              # nothing to wait on
            self.s[0].tool(name='uncached', version=1, cache=False, max_hold=1)(lambda x: x)
        self.s[0].lease_caps = None                       # a manager without lease expiry
        with self.assertRaises(EvalrtError):
            self.s[0].tool(name='old_mgr', version=1, randomness='deterministic', cache=True,
                           max_hold=1)(lambda x: x)

    async def test_closing_with_a_tool_running_stops_its_renewal(self):
        async def endless(x):
            await asyncio.sleep(60)
            return x
        tool = self.s[0].tool(version=1, randomness='deterministic', cache=True)(endless)
        call = asyncio.ensure_future(self._in_run(0, tool(1)))
        await asyncio.sleep(0.5)
        self.assertEqual(len(self.s[0]._renewals), 1)
        await self.s[0].aclose()
        self.assertEqual(self.s[0]._renewals, {})
        call.cancel()
        await asyncio.gather(call, return_exceptions=True)
        await asyncio.sleep(0.1)
        self.assertEqual(self.rig.manager.stats()['leases_held'], 0)   # dropped on close
        self.s[0] = await Session.connect(self.rig.socket)               # for tearDown

    async def test_frozen_holder_expires(self):
        started = os.path.join(self.dir, 'child-started')
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env = dict(os.environ, LEASE_CHILD_STARTED=started,
                   PYTHONPATH=os.pathsep.join([os.path.join(root, 'src'), root]))
        child = subprocess.Popen(
            [sys.executable, '-c', 'import asyncio, sys; from tests.kernel.test_evalrt_lease_expiry '
             'import _child; asyncio.run(_child(sys.argv[1]))', self.rig.socket], cwd=root,
            env=env)
        upper = self.s[0].tool(version=1, randomness='deterministic', cache=True)(frozen_upper)
        try:
            t0 = time.time()
            while not os.path.exists(started):
                self.assertIsNone(child.poll(), 'the child exited before holding the lease')
                self.assertLess(time.time() - t0, 60)
                await asyncio.sleep(0.05)
            waiter = asyncio.ensure_future(self._in_run(0, upper('frozen')))
            await asyncio.sleep(2.5 * TTL)                # alive: it renews, nothing expires
            self.assertFalse(waiter.done())
            self.assertEqual(self.rig.manager.stats().get('leases_expired', 0), 0)
            os.kill(child.pid, signal.SIGSTOP)            # frozen, its connection still open
            t1 = time.monotonic()
            got = await asyncio.wait_for(waiter, 10)
            self.assertEqual(got, 'FROZEN')
            self.assertLess(time.monotonic() - t1, TTL + 2.0)
            self.assertEqual(EXECUTED, [('frozen_upper', 'frozen')])
            stats = self.rig.manager.stats()
            self.assertEqual((stats['leases_expired'], stats.get('leases_dropped', 0)), (1, 0))
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()


class NoTTLSessionTest(unittest.IsolatedAsyncioTestCase):
    async def test_no_ttl_no_renewal_task(self):
        rig = await Rig('nottl').start(lease_ttl=None)
        s = await Session.connect(rig.socket)
        try:
            async def slow(x):
                await asyncio.sleep(0.5)
                return x
            tool = s.tool(version=1, randomness='deterministic', cache=True)(slow)
            async with s.run(s.group('g')):
                call = asyncio.ensure_future(tool(3))
                await asyncio.sleep(0.2)
                self.assertEqual(s._renewals, {})
                self.assertEqual(await call, 3)
        finally:
            await s.aclose()
            await rig.close()


if __name__ == '__main__':
    unittest.main()
