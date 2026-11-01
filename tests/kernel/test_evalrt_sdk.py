"""The packaged system (reports/evalrt_sdk_20261009/INTERFACE.md): runs, groups, registered calls.
Mock models, no GPU:
    PYTHONPATH=src:. python3 -m unittest tests.kernel.test_evalrt_sdk
"""
import asyncio
import json
import os
import shutil
import tempfile
import types
import unittest

from copula.kernel.sdk import SDKError, Session
from copula.kernel.testing.mock_model import MockModel, MockReward
from copula.kernel.testing.rig import Rig

graph = node = None          # static graphs are not in the package


def msgs(text):
    return [{'role': 'user', 'content': text}]


class SDKTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.small = MockModel('sdk-small')
        self.prm = MockReward('sdk-prm')
        self.rig = await Rig('sdk', backends={'small': self.small, 'prm': self.prm}).start()
        self.dir = tempfile.mkdtemp(prefix='evalrt-sdk-')
        self.trace = os.path.join(self.dir, 'trace.jsonl')
        self.s = await Session.connect(self.rig.socket, trace=self.trace)
        self.s.register_llm('gen', randomness='stochastic', cache=True, pin={'temperature': 0.9},
                            max_tokens=64)
        self.s.register_llm('judge', randomness='deterministic', cache=True,
                            pin={'temperature': 0.0}, max_tokens=16)
        self.s.register_llm('solve', backend='small', randomness='stochastic', cache=True)
        self.s.register_llm('free', randomness='stochastic', cache=False)
        self.s.register_llm('score', backend='prm', randomness='deterministic', cache=True)

    async def asyncTearDown(self):
        await self.s.aclose()
        await self.rig.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    async def ask(self, model, text, **kw):
        r = await self.s.chat(model, msgs(text), **kw)
        return r, r['evalrt']['calls']

    def rows(self):
        with open(self.trace) as f:
            return [json.loads(l) for l in f]

    async def test_registration_requires_both_fields(self):
        with self.assertRaises(TypeError):
            self.s.register_llm('x', randomness='stochastic')          # cache missing
        with self.assertRaises(TypeError):
            self.s.tool(randomness='stochastic')                      # cache missing
        with self.assertRaises(SDKError):
            self.s.tool(cache=True)                                   # cached: randomness needed
        with self.assertRaises(SDKError):
            self.s.register_llm('x', cache=True)
        bad = [dict(name='x', randomness='random', cache=True),
               dict(name='x', randomness='stochastic', cache='yes'),
               dict(name='x', randomness='deterministic', cache=True),  # not greedy
               dict(name='x', randomness='stochastic', cache=True, backend='prm'),
               dict(name='x', randomness='stochastic', cache=True, backend='big'),
               dict(name='gen', randomness='stochastic', cache=True),   # registered already
               dict(name='x', randomness='stochastic', cache=True, pin={'seed': 1})]
        for kw in bad:
            name = kw.pop('name')
            with self.assertRaises(SDKError, msg=kw):
                self.s.register_llm(name, **kw)
        with self.assertRaises(SDKError):
            @self.s.tool(randomness='stochastic', cache=True)
            def noisy(x):                                             # no rng
                return x

    async def test_calls_need_a_run_and_a_registered_node(self):
        with self.assertRaises(SDKError) as ctx:
            await self.s.chat('gen', msgs('x'))
        self.assertEqual(ctx.exception.kind, 'no_run')
        async with self.s.run(self.s.group('g')):
            for model, kw, kind in (('nope', {}, 'unregistered'),
                                    ('gen', {'temperature': 0.5}, 'invalid_call'),
                                    ('gen', {'seed': 3}, 'invalid_call'),
                                    ('gen', {'stream': True}, 'invalid_call'),
                                    ('gen', {'tools': [{'type': 'function'}]}, 'invalid_call'),
                                    ('gen', {'frobnicate': 1}, 'invalid_call')):
                with self.assertRaises(SDKError, msg=(model, kw)) as ctx:
                    await self.s.chat(model, msgs('x'), **kw)
                self.assertEqual(ctx.exception.kind, kind)

    async def test_stochastic_cached_shares_within_a_group_only(self):
        g = self.s.group('g')
        async with self.s.run(g) as a:
            _, c1 = await self.ask('gen', 'x')
            _, c2 = await self.ask('gen', 'x')
            r3, c3 = await self.ask('gen', 'y', n=3)
        self.assertEqual([c['index'] for c in c1 + c2], [1, 2])         # a repeat: a new sample
        self.assertNotEqual(c1[0]['key_hash'], c2[0]['key_hash'])
        self.assertEqual(sorted(c['index'] for c in c3), [1, 2, 3])
        self.assertEqual(len({ch['message']['content'] for ch in r3.choices}), 3)
        async with self.s.run(g):                                       # another run: shares
            _, b1 = await self.ask('gen', 'x')
            _, b5 = await self.ask('gen', 'y', n=5)
        self.assertEqual((b1[0]['index'], b1[0]['status'], b1[0]['key_hash']),
                         (1, 'hit', c1[0]['key_hash']))
        self.assertEqual(sorted((c['index'], c['status']) for c in b5),
                         [(1, 'hit'), (2, 'hit'), (3, 'hit'), (4, 'generated'),
                          (5, 'generated')])
        async with self.s.run(self.s.group()):                          # a new group: fresh
            _, f1 = await self.ask('gen', 'x')
        self.assertEqual((f1[0]['index'], f1[0]['status']), (1, 'generated'))
        self.assertNotEqual(f1[0]['key_hash'], c1[0]['key_hash'])
        async with self.s.run(g):
            with self.s.in_group(self.s.group('other')):                # this block: another pool
                _, o1 = await self.ask('gen', 'x')
            _, back = await self.ask('gen', 'x')
        self.assertEqual(o1[0]['status'], 'generated')
        self.assertEqual(back[0]['status'], 'hit')
        self.assertEqual(a.generated, 5)

    async def test_deterministic_cached_is_global_and_never_copied(self):
        async with self.s.run(self.s.group('g1')):
            _, j1 = await self.ask('judge', 'p')
            _, j2 = await self.ask('judge', 'p', n=2)
        async with self.s.run(self.s.group('g2')):
            _, j3 = await self.ask('judge', 'p')
        keys = {c['key_hash'] for c in j1 + j2 + j3}
        self.assertEqual(len(keys), 1)
        self.assertEqual([c['index'] for c in j1 + j2 + j3], [None] * 4)
        self.assertEqual(j3[0]['status'], 'hit')
        self.assertEqual(self.rig.model.calls[keys.pop()], 1)          # computed once

    async def test_uncached_is_drawn_on_every_call(self):
        g = self.s.group('g')
        async with self.s.run(g):
            _, a = await self.ask('free', 'x')
        async with self.s.run(g):
            _, b = await self.ask('free', 'x')
        self.assertEqual([a[0]['status'], b[0]['status']], ['generated', 'generated'])
        self.assertNotEqual(a[0]['key_hash'], b[0]['key_hash'])

    async def test_backends_rewards_and_wrapped_clients(self):
        async def create(**kw):
            raise AssertionError('the original client must not be called')
        client = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)))
        self.s.wrap(client)
        async with self.s.run(self.s.group('g')):
            r = await client.chat.completions.create(model='solve', messages=msgs('2+2'),
                                                     max_tokens=32)
            sc = await client.chat.completions.create(model='score',
                                                      messages=msgs('a<extra_0>b<extra_0>'))
        self.assertTrue(r.choices[0].message.content.startswith('mock:'))
        self.assertEqual(r.usage.completion_tokens, 8)
        self.assertEqual(r.model, 'solve')
        self.assertEqual(len(self.small.calls), 1)
        self.assertEqual(len(json.loads(sc.choices[0].message.content)), 2)
        self.assertIsNone(sc['evalrt']['calls'][0]['index'])

    async def test_tools(self):
        calls = []

        @self.s.tool(version=1, randomness='deterministic', cache=True)
        def upper(text):
            calls.append(('upper', text))
            return text.upper()

        @self.s.tool(version=1, randomness='stochastic', cache=True)
        async def dice(sides, rng):
            calls.append(('dice', sides))
            return rng.randrange(sides)

        @self.s.tool(version=1, randomness='stochastic', cache=False)
        async def coin(rng):
            calls.append(('coin',))
            return rng.random()
        g = self.s.group('g')
        async with self.s.run(g):
            self.assertEqual(await upper.aio('ab'), 'AB')
            d1, d2 = await dice(10 ** 9), await dice(10 ** 9)
            await coin()
        async with self.s.run(self.s.group('h')):
            self.assertEqual(await upper.aio('ab'), 'AB')              # global
            e1 = await dice(10 ** 9)
        async with self.s.run(g):
            f1 = await dice(10 ** 9)
            await coin()
        self.assertEqual(calls.count(('upper', 'ab')), 1)
        self.assertNotEqual(d1, d2)                                     # two samples in one run
        self.assertEqual(f1, d1)                                        # shared within g
        self.assertNotEqual(e1, d1)                                     # independent across groups
        self.assertEqual(calls.count(('dice', 10 ** 9)), 3)
        self.assertEqual(calls.count(('coin',)), 2)
        with self.assertRaises(SDKError):
            async with self.s.run(g):
                await dice(object())                                    # not JSON

    async def test_resume_rereads_its_claims(self):
        g = self.s.group('g')
        async with self.s.run(g, run_id='r-fixed') as r:
            _, a = await self.ask('gen', 'x', n=2)
        other = await Session.connect(self.rig.socket)
        other.register_llm('gen', randomness='stochastic', cache=True, pin={'temperature': 0.9},
                           max_tokens=64)
        try:
            async with other.run(g, run_id='r-fixed'):
                b = (await other.chat('gen', msgs('x'), n=3))['evalrt']['calls']
        finally:
            await other.aclose()
        self.assertEqual(sorted((c['index'], c['status']) for c in b),
                         [(1, 'own'), (2, 'own'), (3, 'generated')])

    async def test_run_cap(self):
        capped = await Session.connect(self.rig.socket, cap=2)
        capped.register_llm('gen', randomness='stochastic', cache=True, pin={'temperature': 0.9},
                            max_tokens=64)
        try:
            async with capped.run(capped.group()):
                await capped.chat('gen', msgs('a'))
                await capped.chat('gen', msgs('b'))
                with self.assertRaises(SDKError) as ctx:
                    await capped.chat('gen', msgs('c'))
            self.assertEqual(ctx.exception.kind, 'run_cap')
        finally:
            await capped.aclose()

    async def test_uncached_nodes_ignore_randomness(self):
        self.s.register_llm('u-none', cache=False)
        self.s.register_llm('u-det', cache=False, randomness='deterministic')   # not greedy: fine
        self.s.register_llm('u-sto', cache=False, randomness='stochastic')
        seen = []

        @self.s.tool(cache=False, randomness='deterministic')
        def noisy(rng):                              # an uncached tool may take rng whatever it says
            seen.append(rng.random())
            return len(seen)
        g = self.s.group('u')
        statuses = []
        for _ in range(2):
            async with self.s.run(g):
                for name in ('u-none', 'u-det', 'u-sto'):
                    _, c = await self.ask(name, 'same')
                    statuses.append(c[0]['status'])
                await noisy.aio()
        self.assertEqual(statuses, ['generated'] * 6)          # a private draw on every call
        self.assertEqual(len(seen), 2)

    @unittest.skip('static graphs are not in the package')
    async def test_static_graphs_run_without_interception(self):
        g4 = graph({'x': node(['task'], instruction='Answer.'),
                    'y': node(['task', 'answer'], 'answer', instruction='Check.', model='small')},
                   [('task', 'x', 0), ('task', 'y', 0), ('x', 'y', 1), ('y', 'output', 0)])
        grp = self.s.group('static')
        first = await self.s.execute(g4, [{'prompt': 'q1'}, {'prompt': 'q2'}], grp)  # no run open
        async with self.s.run(self.s.group('agent')) as agent:         # an agent run is not used
            again = await self.s.execute(g4, [{'prompt': 'q1'}], grp)
        self.assertNotEqual(again['run'], agent.id)
        self.assertEqual(len(self.s._programs), 1)                       # built once
        res1, res2 = first['results'], again['results']
        self.assertTrue(all(leaf['answer'] for r in res1 for leaf in r['leaves']))
        st = sorted(ex['status'] for ex in res2[0]['executions'] if 'key_hash' in ex)
        self.assertEqual(st, ['hit_existing', 'hit_existing'])          # the same group shares
        self.assertEqual(res2[0]['leaves'][0]['answer'], res1[0]['leaves'][0]['answer'])
        rows = [x for x in self.rows() if x.get('run') == again['run'] and x['ev'] == 'call']
        self.assertTrue(rows and all(x['group'] == 'static' for x in rows))
        grown = graph({'x': node(['task'], instruction='Answer.')},
                      [('task', 'x', 0), ('x', 'output', 0)])
        grown = dict(grown, templates={'t': node(['task'], instruction='More.')})
        with self.assertRaises(SDKError):
            await self.s.execute(grown, [{'prompt': 'q'}], grp)

    async def test_review_fixes(self):
        # an OpenAI-style async client whose create is a plain function returning a coroutine
        class AsyncCompletions:
            def create(self, **kw):
                async def never():
                    raise AssertionError('not routed')
                return never()

            def parse(self, **kw):
                raise AssertionError('parse must be refused, not sent')
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=AsyncCompletions()))
        self.s.wrap(client)
        g, search = self.s.group('g'), self.s.group('search')
        async with self.s.run(g):
            r = await client.chat.completions.create(model='gen', messages=msgs('x'),
                                                     stream=False, logprobs=None, tools=None,
                                                     stop='\n')
            with self.assertRaises(SDKError):
                client.chat.completions.parse(model='gen', messages=msgs('x'))
            named = await self.s.chat('gen', [{'role': 'user', 'content': 'x', 'name': 'bob'}])
            self.assertNotEqual(named['evalrt']['calls'][0]['key_hash'],    # a name is kept: it
                                r['evalrt']['calls'][0]['key_hash'])        # is another request
            with self.assertRaises(SDKError):                               # other keys are not
                await self.s.chat('gen', [{'role': 'user', 'content': 'x', 'colour': 'red'}])
        self.assertIsNone(r.choices[0].message.tool_calls)                  # OpenAI-shaped
        self.assertIsNone(r.usage.prompt_tokens_details)
        self.assertIsInstance(r.created, int)
        # a run opened inside another run's in_group block uses its own group
        async with self.s.run(g):
            with self.s.in_group(search):
                async with self.s.run(self.s.group('inner')) as inner:
                    _, c = await self.ask('gen', 'y')
        rows = [x for x in self.rows() if x.get('ev') == 'call' and x['run'] == inner.id]
        self.assertEqual(rows[0]['group'], 'inner')
        # concurrent identical deterministic calls in one run: generated once, the rest waited
        async with self.s.run(g) as r2:
            _, j = await self.ask('judge', 'concurrent', n=3)
        self.assertEqual(sorted(x['status'] for x in j), ['generated', 'waited', 'waited'])
        self.assertEqual(r2.generated, 1)
        # resuming a run in the same session re-reads its claims
        async with self.s.run(g, run_id='r-same') as first:
            _, a = await self.ask('gen', 'z', n=2)
        async with self.s.run(g, run_id='r-same'):
            _, b = await self.ask('gen', 'z', n=3)
        self.assertEqual(sorted((x['index'], x['status']) for x in b),
                         [(1, 'own'), (2, 'own'), (3, 'generated')])
        from copula.kernel.sdk import session as sdk_session
        async with self.s.run(g) as done:
            pass
        token = sdk_session._run.set(done)              # a task that outlived its run
        try:
            with self.assertRaises(SDKError) as ctx:
                await self.s.chat('gen', msgs('late'))
            self.assertEqual(ctx.exception.kind, 'no_run')
        finally:
            sdk_session._run.reset(token)

    async def test_tool_memo_copies_and_retries(self):
        @self.s.tool(version=1, randomness='deterministic', cache=True)
        def listing(n):
            return {'items': list(range(n))}

        started = asyncio.Event()

        @self.s.tool(version=1, randomness='deterministic', cache=True)
        async def slow(x):
            started.set()
            await asyncio.sleep(0.3)
            return x * 2
        async with self.s.run(self.s.group('a')):
            got = await listing.aio(2)
            got['items'].append('MUTATED')
        async with self.s.run(self.s.group('b')):
            self.assertEqual(await listing.aio(2), {'items': [0, 1]})
        g = self.s.group('c')

        async def first():
            async with self.s.run(g):
                await asyncio.wait_for(slow(3), 0.05)

        async def second():
            await started.wait()
            async with self.s.run(g):
                return await slow(3)
        res = await asyncio.gather(first(), second(), return_exceptions=True)
        self.assertIsInstance(res[0], asyncio.TimeoutError)
        self.assertEqual(res[1], 6)                     # the waiter retried, not cancelled

    async def test_model_errors_are_traced_then_raised(self):
        self.rig.model.body_fault = lambda body: 'request' if 'boom' in \
            body['messages'][-1]['content'] else None
        async with self.s.run(self.s.group('g')) as r:
            with self.assertRaises(SDKError) as ctx:
                await self.s.chat('gen', msgs('boom'), n=2)
        self.assertEqual(ctx.exception.kind, 'model_error')
        rows = [x for x in self.rows() if x.get('ev') == 'call' and x['run'] == r.id]
        self.assertEqual(len(rows), 2)
        self.assertEqual(r.generated, 2)

    @unittest.skip('static graphs are not in the package')
    async def test_graph_inputs_keep_their_ids(self):
        retrieve = {'kind': 'retrieve', 'metric': 'tfidf', 'k': 2, 'stratify': 'none',
                    'relative_to': 'query', 'label_filter': 'any', 'template': 'Q: {facts}',
                    'item_chars': 400, 'max_chars': 2000, 'inputs': [{'type': 'task', 'note': ''}]}
        g4 = graph({'ret': retrieve, 'x': node(['task', 'examples'], instruction='Answer.')},
                   [('task', 'ret', 0), ('task', 'x', 0), ('ret', 'x', 1), ('x', 'output', 0)])
        pool = [{'id': 'q1', 'facts': 'alpha beta', 'labels': ['A']},
                {'id': 'q2', 'facts': 'alpha gamma', 'labels': ['B']},
                {'id': 'q3', 'facts': 'delta', 'labels': ['C']}]
        out = (await self.s.execute(g4, [{'id': 'q1', 'prompt': 'alpha beta', 'pool': pool}],
                                    self.s.group('ids')))['results']
        ret = next(ex for ex in out[0]['executions'] if ex['node'] == 'ret')
        self.assertNotIn('q1', ret['retrieved'])
        self.assertEqual(out[0]['id'], 'q1')
        self.assertEqual(self.s.graph_bound(g4)['calls'], 1)

    async def test_trace_rows(self):
        async with self.s.run(self.s.group('g'), label='demo') as r:
            await self.ask('gen', 'x')
        rows = self.rows()
        self.assertEqual([x['ev'] for x in rows], ['run_start', 'call', 'run_end'])
        call = rows[1]
        self.assertEqual((call['run'], call['group'], call['node'], call['status'], call['index']),
                         (r.id, 'g', 'gen', 'generated', 1))

    async def test_sync_agents(self):
        socket = self.rig.socket

        def agent():
            s = Session.connect_sync(socket)
            s.register_llm('gen', randomness='stochastic', cache=True, pin={'temperature': 0.9},
                           max_tokens=64)

            @s.tool(version=1, randomness='deterministic', cache=True)
            def rev(text):
                return text[::-1]

            def create(**kw):
                raise AssertionError('not called')
            client = types.SimpleNamespace(chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)))
            s.wrap(client)
            try:
                with s.run_sync(s.group('sync')):
                    r1 = client.chat.completions.create(model='gen', messages=msgs('x'))
                    r2 = client.chat.completions.create(model='gen', messages=msgs('x'))
                    t = rev('abc')
                return r1, r2, t
            finally:
                s.close()
        r1, r2, t = await asyncio.to_thread(agent)
        self.assertEqual([r1['evalrt']['calls'][0]['index'], r2['evalrt']['calls'][0]['index']],
                         [1, 2])
        self.assertEqual(t, 'cba')


if __name__ == '__main__':
    unittest.main()
