"""The packaged system with real agent clients (reports/sysval_20261009): tool calling, structured
output, openai clients wrapped at their request method, and process-wide interception. Mock models,
no GPU; the openai tests skip when the package is missing:
    PYTHONPATH=src:. python3 -m unittest tests.kernel.test_evalrt_sdk_clients
"""
import asyncio
import concurrent.futures
import json
import os
import shutil
import tempfile
import unittest

from copula.kernel.sdk import SDKError, Session
from copula.kernel.testing.rig import Rig

try:
    import openai
except ImportError:                 # pragma: no cover
    openai = None

TOOLS = [{'type': 'function', 'function': {
    'name': 'add', 'description': 'add two integers',
    'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'},
                                                    'b': {'type': 'integer'}},
                   'required': ['a', 'b']}}}]
def _dead_url():
    """A local port nothing listens on (bound, then closed): a request that leaves the process
    fails at once. A fixed low port is not safe: some hosts run a discard service on port 9."""
    import socket
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return 'http://127.0.0.1:%d/v1' % probe.getsockname()[1]


DEAD = _dead_url()


def respond(body):
    """A scripted model: with tools and no tool result yet it calls `add`; after a tool result it
    answers with it; asked for JSON it returns JSON; otherwise the key's own content."""
    last = body['messages'][-1]
    if body.get('tools') and last['role'] == 'user':
        return {'content': '', 'finish_reason': 'tool_calls', 'tool_calls': [
            {'id': 'call_%d' % len(last['content']), 'type': 'function',
             'function': {'name': 'add', 'arguments': '{"a": 2, "b": 3}'}}]}
    if last['role'] == 'tool':
        return 'the sum is %s' % last['content']
    if body.get('response_format'):
        return json.dumps({'answer': len(last['content']), 'unit': 'chars'})
    return None


def msgs(text):
    return [{'role': 'user', 'content': text}]


class ClientsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.rig = await Rig('clients', respond=respond).start()
        self.dir = tempfile.mkdtemp(prefix='evalrt-clients-')
        self.trace = os.path.join(self.dir, 'trace.jsonl')
        self.s = await Session.connect(self.rig.socket, trace=self.trace)
        self.s.register_llm('gen', randomness='stochastic', cache=True, pin={'temperature': 0.9},
                            max_tokens=64)
        self.s.register_llm('judge', randomness='deterministic', cache=True,
                            pin={'temperature': 0.0}, max_tokens=16)

    async def asyncTearDown(self):
        await self.s.aclose()
        await self.rig.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def calls(self, run_id=None):
        with open(self.trace) as f:
            rows = [json.loads(line) for line in f]
        return [r for r in rows if r.get('ev') == 'call' and (run_id is None or r['run'] == run_id)]

    async def agent(self, g):
        """A tool-calling loop: the model calls `add`, the agent runs it, the model answers."""
        history = msgs('what is 2 + 3?')
        r = await self.s.chat('gen', history, tools=TOOLS, tool_choice='auto')
        message = r.choices[0].message
        self.assertEqual(r.choices[0].finish_reason, 'tool_calls')
        self.assertIsNone(message.content)
        history.append(message)                             # the response message itself
        call = message.tool_calls[0]
        args = json.loads(call.function.arguments)
        history.append({'role': 'tool', 'tool_call_id': call.id,
                        'content': str(args['a'] + args['b'])})
        final = await self.s.chat('gen', history, tools=TOOLS)
        return message, final.choices[0].message.content

    async def test_tool_calling_is_part_of_the_request(self):
        g = self.s.group('tools')
        async with self.s.run(g) as first:
            m1, a1 = await self.agent(g)
        async with self.s.run(g) as second:
            m2, a2 = await self.agent(g)
        self.assertEqual(a1, 'the sum is 5')
        self.assertEqual((m1, a1), (m2, a2))
        self.assertEqual([c['status'] for c in self.calls(first.id)], ['generated', 'generated'])
        self.assertEqual([c['status'] for c in self.calls(second.id)], ['hit', 'hit'])
        # the same messages without tools are another request (another pool)
        async with self.s.run(g):
            plain = await self.s.chat('gen', msgs('what is 2 + 3?'))
        self.assertTrue(plain.choices[0].message.content.startswith('mock:'))
        self.assertNotEqual(self.calls()[-1]['pool'], self.calls(first.id)[0]['pool'])
        # malformed tool fields and messages are refused before anything is sent
        async with self.s.run(g):
            for kw in ({'tools': [{'type': 'function'}]}, {'tool_choice': 'sometimes'},
                       {'response_format': {'type': 'xml'}}, {'parallel_tool_calls': 'yes'}):
                with self.assertRaises(SDKError):
                    await self.s.chat('gen', msgs('x'), **kw)
            with self.assertRaises(SDKError):
                await self.s.chat('gen', [{'role': 'tool', 'content': '5'}])
            empty = await self.s.chat('gen', msgs('x'), tools=[], response_format=None)
        self.assertTrue(empty.choices[0].message.content.startswith('mock:'))

    async def test_ambient_runs(self):
        g = self.s.group('amb')

        def worker():                               # a thread: no run context of its own
            return self.s.chat_sync('gen', msgs('from a thread'))
        async with self.s.run(g):
            with self.assertRaises(SDKError) as ctx:
                await asyncio.get_running_loop().run_in_executor(None, worker)
            self.assertEqual(ctx.exception.kind, 'no_run')
        async with self.s.run(g, ambient=True) as run:
            r = await asyncio.get_running_loop().run_in_executor(None, worker)
            with self.assertRaises(SDKError):
                async with self.s.run(g, ambient=True):
                    pass
        self.assertEqual(r['evalrt']['run'], run.id)
        with self.assertRaises(SDKError):                # ended with its run
            await asyncio.get_running_loop().run_in_executor(None, worker)
        async with self.s.run(g):
            with self.assertRaises(SDKError) as ctx:     # on the loop's own thread: would block it
                self.s.chat_sync('gen', msgs('x'))
        self.assertEqual(ctx.exception.kind, 'not_sync')


    async def test_http_endpoint(self):
        import urllib.request
        base = await self.s.serve_http()
        loop = asyncio.get_running_loop()

        def post(url, body):
            req = urllib.request.Request(url + '/chat/completions', method='POST',
                                         data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=10) as h:
                    return h.status, h.read().decode()
            except urllib.error.HTTPError as err:
                return err.code, err.read().decode()
        g, other = self.s.group('http'), self.s.group('http-other')
        async with self.s.run(g, ambient=True) as run:
            st, out = await loop.run_in_executor(None, post, base, {
                'model': 'gen', 'messages': msgs('over http'), 'n': 2})
            st2, out2 = await loop.run_in_executor(None, post, self.s.http_url(run, other), {
                'model': 'gen', 'messages': msgs('over http')})
            st3, sse = await loop.run_in_executor(None, post, self.s.http_url(run), {
                'model': 'gen', 'messages': msgs('over http'), 'stream': True,
                'stream_options': {'include_usage': True}})
            st4, bad = await loop.run_in_executor(None, post, base, {
                'model': 'nope', 'messages': msgs('x')})
        st5, closed = await loop.run_in_executor(None, post, self.s.http_url(run), {
            'model': 'gen', 'messages': msgs('x')})
        self.assertEqual((st, st2, st3, st4, st5), (200, 200, 200, 400, 400))
        body = json.loads(out)
        self.assertEqual(len(body['choices']), 2)
        self.assertNotIn('evalrt', body)
        events = [json.loads(line[6:]) for line in sse.split('\n')
                  if line.startswith('data: ') and line != 'data: [DONE]']
        self.assertTrue(sse.rstrip().endswith('data: [DONE]'))
        self.assertTrue(events[0]['choices'][0]['delta']['content'].startswith('mock:'))
        self.assertEqual(json.loads(bad)['error']['type'], 'unregistered')
        self.assertEqual(json.loads(closed)['error']['type'], 'no_run')
        rows = self.calls(run.id)
        self.assertEqual([(r['group'], r['index']) for r in rows],
                         [('http', 1), ('http', 2), ('http-other', 1), ('http', 3)])


    async def test_http_wire_and_errors(self):
        """Review fixes D1, M5: manager errors are final replies (502, x-should-retry: false);
        Expect: 100-continue and chunked trailers are handled."""
        base = await self.s.serve_http()
        host, port = base.split('//')[1].split('/')[0].split(':')
        g = self.s.group('wire')

        async def raw(req):
            r, w = await asyncio.open_connection(host, int(port))
            w.write(req)
            await w.drain()
            data = await asyncio.wait_for(r.read(), 10)              # Connection: close
            w.close()
            return data.decode()
        body = json.dumps({'model': 'gen', 'messages': msgs('wire')}).encode()
        async with self.s.run(g, ambient=True):
            cont = await raw(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: '
                             b'application/json\r\nExpect: 100-continue\r\nConnection: close'
                             b'\r\nContent-Length: %d\r\n\r\n' % len(body) + body)
            chunked = await raw(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n'
                                b'Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n'
                                + b'%x\r\n' % len(body) + body + b'\r\n0\r\nX-T: 1\r\n\r\n')
            self.rig.model.fault = lambda key, attempt: 'crash'
            failed = await raw(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nConnection: '
                               b'close\r\nContent-Length: %d\r\n\r\n' % len(body) + body)
            self.rig.model.fault = None
        self.assertTrue(cont.startswith('HTTP/1.1 100 Continue'))
        self.assertIn('HTTP/1.1 200', cont)
        self.assertIn('HTTP/1.1 200', chunked)
        self.assertTrue(failed.startswith('HTTP/1.1 502'))
        self.assertIn('x-should-retry: false', failed)

    async def test_failed_starts_and_ended_runs(self):
        """Review fixes M1, M2: a run whose start failed is not served; a draw on an ended
        attempt is refused by the manager."""
        g = self.s.group('starts')
        async with self.s.run(g, ambient=True, run_id='x') as x:
            with self.assertRaises(SDKError):
                async with self.s.run(g, ambient=True, run_id='x'):
                    pass
            self.assertIs(self.s._open['x'], x)
        self.assertNotIn('x', self.s._open)
        ticket = x.ticket_of(g)
        with self.assertRaises(Exception) as ctx:
            await self.s.manager.draw(ticket, 'gen', {'messages': msgs('late'), 'max_tokens': 8},
                                      'run', partition=g.partition, attempt=x.attempt)
        self.assertEqual(ctx.exception.kind, 'run_ended')


@unittest.skipIf(openai is None, 'the openai package is not installed')
class OpenAIClientsTest(ClientsTest):
    async def test_wrapped_openai_clients_route_every_chat_path(self):
        client = openai.AsyncOpenAI(api_key='x', base_url=DEAD, max_retries=0, timeout=10)
        early = client.chat.completions.with_raw_response   # made before the wrap: still routed
        self.s.wrap(client)
        g = self.s.group('oa')
        async with self.s.run(g) as run:
            r = await client.chat.completions.create(model='gen', messages=msgs('hello'), n=2)
            raw = await early.create(model='gen', messages=msgs('hello'))
            stream = await client.chat.completions.create(
                model='gen', messages=msgs('hello'), stream=True,
                stream_options={'include_usage': True})
            parts, usage = [], None
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    parts.append(chunk.choices[0].delta.content)
                usage = chunk.usage or usage
            tools = await client.chat.completions.create(model='gen', messages=msgs('2+3?'),
                                                         tools=TOOLS)
            if hasattr(client.chat.completions, 'parse'):
                from pydantic import BaseModel

                class Answer(BaseModel):
                    answer: int
                    unit: str
                parsed = await client.chat.completions.parse(model='judge',
                                                             messages=msgs('count me'),
                                                             response_format=Answer)
                self.assertEqual(parsed.choices[0].message.parsed, Answer(answer=8, unit='chars'))
            with self.assertRaises(SDKError):               # not a chat completion
                await client.completions.create(model='gen', prompt='x')
            with self.assertRaises(SDKError):
                await client.embeddings.create(model='gen', input='x')
            with self.assertRaises(SDKError):               # not registered: nothing bypasses
                await client.chat.completions.create(model='gpt-4o', messages=msgs('x'))
        self.assertIsInstance(r, openai.types.chat.ChatCompletion)
        self.assertEqual(len(r.choices), 2)
        third = raw.parse().choices[0].message.content      # the 3rd sample of the request
        self.assertTrue(third.startswith('mock:'))
        self.assertNotIn(third, [c.message.content for c in r.choices])
        streamed = ''.join(parts)                   # the 4th sample of the request in this run
        self.assertTrue(streamed.startswith('mock:'))
        self.assertNotIn(streamed, [c.message.content for c in r.choices])
        self.assertEqual(usage.completion_tokens, 8)
        self.assertEqual(tools.choices[0].message.tool_calls[0].function.name, 'add')
        rows = self.calls(run.id)
        self.assertEqual([c['index'] for c in rows[:4]], [1, 2, 3, 4])
        self.assertTrue(all(c['status'] == 'generated' for c in rows))

    async def test_review_fixes_openai(self):
        """D1 retries, D2 copies, D4 stream helpers, D5 parsed messages, M4 streaming responses,
        M8 pass-throughs."""
        from pydantic import BaseModel

        class Answer(BaseModel):
            answer: int
            unit: str
        client = openai.AsyncOpenAI(api_key='x', base_url=DEAD, max_retries=0, timeout=10)
        self.s.wrap(client)
        g = self.s.group('fixes')
        async with self.s.run(g) as run:
            copy = client.with_options(timeout=30)                       # D2
            r = await copy.chat.completions.create(model='gen', messages=msgs('copy'))
            async with client.chat.completions.stream(model='gen',       # D4
                                                       messages=msgs('helper')) as stream:
                async for _ in stream:
                    pass
                final = await stream.get_final_completion()
            p1 = await client.chat.completions.parse(model='judge', messages=msgs('count me'),
                                                     response_format=Answer)
            history = msgs('count me') + [p1.choices[0].message] + msgs('and again')    # D5
            p2 = await client.chat.completions.parse(model='judge', messages=history,
                                                     response_format=Answer)
            before = run.generated
            with self.assertRaises(SDKError):                             # M4
                async with client.chat.completions.with_streaming_response.create(
                        model='gen', messages=msgs('x')):
                    pass
            self.assertEqual(run.generated, before)
        self.assertTrue(r.choices[0].message.content.startswith('mock:'))
        self.assertTrue(final.choices[0].message.content.startswith('mock:'))
        self.assertIsNotNone(p2.choices[0].message.parsed)
        # D1: over HTTP a model failure is answered once; the openai client does not retry it
        base = await self.s.serve_http()
        calls0 = sum(self.rig.model.calls.values())
        self.rig.model.fault = lambda key, attempt: 'crash'
        try:
            async with self.s.run(g) as run2:
                c = openai.AsyncOpenAI(api_key='x', base_url=self.s.http_url(run2), max_retries=2)
                with self.assertRaises(openai.APIStatusError):
                    await c.chat.completions.create(model='gen', messages=msgs('fails'))
                await c.close()
        finally:
            self.rig.model.fault = None
        self.assertEqual(sum(self.rig.model.calls.values()) - calls0, 1)
        # M8: an unregistered model inside a run under intercept is counted
        with self.s.intercept():
            async with self.s.run(g, ambient=True):
                other = openai.AsyncOpenAI(api_key='x', base_url=DEAD, max_retries=0, timeout=10)
                with self.assertRaises(openai.APIConnectionError):
                    await other.chat.completions.create(model='gpt-4o', messages=msgs('q'))
        self.assertEqual(self.s.passthrough, 1)

    async def test_sync_client_from_a_thread(self):
        client = openai.OpenAI(api_key='x', base_url=DEAD, max_retries=0, timeout=10)
        self.s.wrap(client)
        g = self.s.group('oa-sync')
        async with self.s.run(g, ambient=True) as run:
            loop = asyncio.get_running_loop()
            r = await loop.run_in_executor(None, lambda: client.chat.completions.create(
                model='gen', messages=msgs('sync')))
            raw = await loop.run_in_executor(None, lambda: (
                client.chat.completions.with_raw_response.create(model='gen',
                                                                 messages=msgs('sync'))))
            chunks = await loop.run_in_executor(None, lambda: list(
                client.chat.completions.create(model='gen', messages=msgs('sync'), stream=True)))
        self.assertTrue(r.choices[0].message.content.startswith('mock:'))
        self.assertEqual(dict(raw.headers), {})
        self.assertEqual(chunks[-1].choices[0].finish_reason, 'stop')
        self.assertEqual([c['index'] for c in self.calls(run.id)], [1, 2, 3])

    async def test_openai_client_over_http(self):
        async with self.s.run(self.s.group('oa-http')) as run:
            await self.s.serve_http()
            client = openai.AsyncOpenAI(api_key='x', base_url=self.s.http_url(run), max_retries=0)
            r = await client.chat.completions.create(model='gen', messages=msgs('h'), tools=TOOLS)
            chunks = [c async for c in await client.chat.completions.create(
                model='gen', messages=msgs('h'), stream=True)]
            await client.close()
        self.assertEqual(r.choices[0].message.tool_calls[0].function.name, 'add')
        self.assertTrue(chunks[0].choices[0].delta.content.startswith('mock:'))
        self.assertEqual([c['index'] for c in self.calls(run.id)], [1, 1])

    async def test_intercept_routes_registered_calls_of_every_client(self):
        made_before = openai.OpenAI(api_key='x', base_url=DEAD, max_retries=0, timeout=10)
        before_raw = made_before.chat.completions.with_raw_response     # cached early
        g = self.s.group('icpt')
        pool = concurrent.futures.ThreadPoolExecutor(4)
        loop = asyncio.get_running_loop()
        with self.s.intercept():
            async with self.s.run(g, ambient=True) as run:
                # a framework's own clients, in its own threads, made before or after the patch
                rs = await asyncio.gather(*[loop.run_in_executor(pool, lambda i=i: (
                    before_raw.create(model='gen', messages=msgs('q%d' % i)).parse()))
                    for i in range(4)])
                later = openai.AsyncOpenAI(api_key='x', base_url=DEAD, max_retries=0, timeout=10)
                a = await later.chat.completions.create(model='judge', messages=msgs('q'))
                with self.assertRaises(openai.APIConnectionError):    # not registered: unchanged
                    await later.chat.completions.create(model='gpt-4o', messages=msgs('q'))
        pool.shutdown()
        self.assertEqual(len({r.choices[0].message.content for r in rs}), 4)
        rows = self.calls(run.id)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[-1]['node'], 'judge')
        self.assertTrue(a.choices[0].message.content.startswith('mock:'))
        with self.assertRaises(openai.APIConnectionError):            # uninstalled with the block
            made_before.chat.completions.create(model='gen', messages=msgs('q'))


if __name__ == '__main__':
    unittest.main()
