"""Copula: the Responses and Anthropic Messages protocols (src/copula/protocols.py;
docs/reference/http.md). Translation units, then the
proxy on the kernel's mock rig. No GPU, no SDKs:
    PYTHONPATH=src:. python3 -m unittest tests.test_copula_protocols
"""
import asyncio
import json
import os
import shutil
import socket
import tempfile
import unittest
import urllib.error
import urllib.request

from copula.kernel.sdk import Session
from copula.kernel.testing.rig import Rig

from copula import protocols as P
from copula.proxy import Proxy
from copula.registry import Registry
from tests.copula_fixtures.mock_scripts import tools_reply


class _FakeReg:
    def __init__(self):
        self.stored = {}

    def put_response(self, rid, context, response):
        self.stored[rid] = (context, response)

    def response_context(self, rid):
        got = self.stored.get(rid)
        return None if got is None else got[0]


def chat_data(content='hi', calls=None, finish='stop', cid='evalrt-abc'):
    msg = {'role': 'assistant', 'content': content}
    if calls:
        msg['tool_calls'] = calls
    return {'id': cid, 'created': 5, 'model': 'm',
            'choices': [{'index': 0, 'finish_reason': finish, 'message': msg}],
            'usage': {'prompt_tokens': 3, 'completion_tokens': 4, 'total_tokens': 7}}


CALL = {'id': 'call_1', 'type': 'function', 'function': {'name': 'look', 'arguments': '{"q": "x"}'}}


class AnthropicTranslationTest(unittest.TestCase):
    def test_request(self):
        req = {'model': 'm', 'max_tokens': 9, 'system': [{'type': 'text', 'text': 'a'},
                                                           {'type': 'text', 'text': 'b'}],
               'temperature': 0.5, 'top_k': 3, 'stop_sequences': ['X'], 'metadata': {'u': 1},
               'tools': [{'name': 'look', 'description': 'd', 'input_schema': {'type': 'object'}}],
               'tool_choice': {'type': 'any', 'disable_parallel_tool_use': True},
               'messages': [
                   {'role': 'user', 'content': [{'type': 'text', 'text': 'q1',
                                                 'cache_control': {'type': 'ephemeral'}}]},
                   {'role': 'assistant', 'content': [{'type': 'text', 'text': 'let me look'},
                                                     {'type': 'tool_use', 'id': 'toolu_1',
                                                      'name': 'look', 'input': {'q': 'x'}}]},
                   {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_1',
                                                 'content': [{'type': 'text', 'text': 'found'}]},
                                                {'type': 'text', 'text': 'go on'}]}]}
        c = P.AnthropicProto(req).chat
        self.assertEqual(c['messages'], [
            {'role': 'system', 'content': 'a\nb'}, {'role': 'user', 'content': 'q1'},
            {'role': 'assistant', 'content': 'let me look', 'tool_calls': [
                {'id': 'toolu_1', 'type': 'function',
                 'function': {'name': 'look', 'arguments': '{"q": "x"}'}}]},
            {'role': 'tool', 'tool_call_id': 'toolu_1', 'content': 'found'},
            {'role': 'user', 'content': 'go on'}])
        self.assertEqual((c['max_tokens'], c['top_k'], c['stop'], c['tool_choice'],
                          c['parallel_tool_calls']), (9, 3, ['X'], 'required', False))
        self.assertEqual(c['tools'][0]['function']['parameters'], {'type': 'object'})
        self.assertNotIn('metadata', c)

    def test_refusals(self):
        base = {'model': 'm', 'max_tokens': 5, 'messages': [{'role': 'user', 'content': 'x'}]}
        for bad in ({'max_tokens': None}, {'thinking': {'type': 'enabled', 'budget_tokens': 9}},
                    {'container': 'c'}, {'tools': [{'type': 'web_search_20250305', 'name': 'w'}]},
                    {'messages': [{'role': 'user', 'content': [{'type': 'image', 'source': {}}]}]},
                    {'messages': []},
                    {'messages': [{'role': 'user', 'content': 'x'},
                                  {'role': 'assistant', 'content': '{'}]}):        # prefill
            with self.assertRaises(P.ProtoError, msg=bad):
                P.AnthropicProto(dict(base, **bad))

    def test_render_and_events(self):
        p = P.AnthropicProto({'model': 'm', 'max_tokens': 5,
                              'messages': [{'role': 'user', 'content': 'x'}]})
        r = p.render(chat_data('', [CALL], 'tool_calls'))
        self.assertEqual(r['stop_reason'], 'tool_use')
        self.assertEqual(r['content'], [{'type': 'tool_use', 'id': 'call_1', 'name': 'look',
                                         'input': {'q': 'x'}}])
        self.assertEqual(p.render(chat_data('t', finish='length'))['stop_reason'], 'max_tokens')
        self.assertEqual(p.render(chat_data('', [CALL], 'stop'))['stop_reason'], 'tool_use')
        self.assertEqual(r['id'], p.render(chat_data('', [CALL], 'tool_calls'))['id'])
        types = [l[7:] for l in p.events(chat_data('hello')).splitlines() if l.startswith('event: ')]
        self.assertEqual(types, ['message_start', 'content_block_start', 'content_block_delta',
                                 'content_block_stop', 'message_delta', 'message_stop'])
        self.assertEqual(P.AnthropicProto.error(404, 'x', 'y')['error']['type'], 'not_found_error')


class ResponsesTranslationTest(unittest.TestCase):
    def test_items(self):
        msgs = P.responses_messages([
            {'role': 'developer', 'content': 'rules'},
            {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'a'},
                                                            {'type': 'input_text', 'text': 'b'}]},
            {'type': 'reasoning', 'summary': []},
            {'type': 'function_call', 'call_id': 'c1', 'name': 'f', 'arguments': '{}'},
            {'type': 'function_call', 'call_id': 'c2', 'name': 'g', 'arguments': '{}'},
            {'type': 'function_call_output', 'call_id': 'c1', 'output': 'r1'},
            {'type': 'function_call_output', 'call_id': 'c2', 'output': 'r2'}])
        self.assertEqual([m['role'] for m in msgs], ['system', 'user', 'assistant', 'tool', 'tool'])
        self.assertEqual(msgs[1]['content'], 'a\nb')
        self.assertEqual([c['id'] for c in msgs[2]['tool_calls']], ['c1', 'c2'])
        self.assertEqual(P.responses_messages('hi'), [{'role': 'user', 'content': 'hi'}])
        for bad in ([{'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'u'}]}],
                    [{'type': 'item_reference', 'id': 'x'}], 5):
            with self.assertRaises(P.ProtoError):
                P.responses_messages(bad)

    def test_request_chain_and_render(self):
        reg = _FakeReg()
        p1 = P.ResponsesProto({'model': 'm', 'input': 'q', 'instructions': 'sys',
                               'max_output_tokens': 7, 'store': False,
                               'tools': [{'type': 'function', 'name': 'f', 'parameters': {}}],
                               'tool_choice': {'type': 'function', 'name': 'f'},
                               'text': {'format': {'type': 'json_schema', 'name': 's',
                                                   'schema': {'type': 'object'}}}}, reg)
        c = p1.chat
        self.assertEqual(c['messages'], [{'role': 'system', 'content': 'sys'},
                                         {'role': 'user', 'content': 'q'}])
        self.assertEqual((c['max_tokens'], c['tool_choice']['function']['name'],
                          c['response_format']['type']), (7, 'f', 'json_schema'))
        r1 = p1.render(chat_data('', [CALL], 'tool_calls'))
        self.assertEqual([o['type'] for o in r1['output']], ['function_call'])
        self.assertEqual(r1['output'][0]['call_id'], 'call_1')
        p2 = P.ResponsesProto({'model': 'm', 'previous_response_id': r1['id'], 'input': [
            {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'ok'}]}, reg)
        self.assertEqual([m['role'] for m in p2.chat['messages']], ['user', 'assistant', 'tool'])
        r2 = p2.render(chat_data('done', finish='length', cid='evalrt-def'))
        self.assertEqual((r2['status'], r2['incomplete_details']), ('incomplete',
                                                                     {'reason': 'max_output_tokens'}))
        with self.assertRaises(P.ProtoError):
            P.ResponsesProto({'model': 'm', 'input': 'x', 'previous_response_id': 'resp_no'}, reg)
        for bad in ({'conversation': 'c'}, {'reasoning': {'effort': 'high'}}, {'foo': 1},
                    {'tools': [{'type': 'web_search'}]}, {'top_logprobs': 3},
                    {'truncation': 'auto'}, {'text': {'verbosity': 'low'}},
                    {'tool_choice': {'type': 'allowed_tools', 'tools': []}}):
            with self.assertRaises(P.ProtoError, msg=bad):
                P.ResponsesProto(dict({'model': 'm', 'input': 'x'}, **bad), reg)

    def test_ids_follow_the_chain_context(self):
        reg = _FakeReg()
        a = P.ResponsesProto({'model': 'm', 'instructions': 'X', 'input': 'hi'}, reg)
        b = P.ResponsesProto({'model': 'm', 'input': [{'role': 'system', 'content': 'X'},
                                                      {'role': 'user', 'content': 'hi'}]}, reg)
        self.assertEqual(a.chat['messages'], b.chat['messages'])   # one chat request ...
        ra, rb = a.render(chat_data()), b.render(chat_data())
        self.assertNotEqual(ra['id'], rb['id'])                     # ... two chains
        self.assertEqual(reg.stored[ra['id']][0][0], {'role': 'user', 'content': 'hi'})

    def test_events(self):
        p = P.ResponsesProto({'model': 'm', 'input': 'x', 'max_output_tokens': 3}, _FakeReg())
        start = json.loads(p.events(chat_data('h', finish='length')).split('data: ', 2)[1]
                           .split('\n', 1)[0])
        self.assertIsNone(start['response']['incomplete_details'])
        p = P.ResponsesProto({'model': 'm', 'input': 'x'}, _FakeReg())
        raw = p.events(chat_data('hello'))
        evs = [json.loads(l[6:]) for l in raw.splitlines() if l.startswith('data: ')]
        self.assertEqual([e['type'] for e in evs], [
            'response.created', 'response.in_progress', 'response.output_item.added',
            'response.content_part.added', 'response.output_text.delta',
            'response.output_text.done', 'response.content_part.done',
            'response.output_item.done', 'response.completed'])
        self.assertEqual([e['sequence_number'] for e in evs], list(range(len(evs))))
        self.assertEqual(evs[-1]['response']['output'][0]['content'][0]['text'], 'hello')


def post(url, path, body, headers=None):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers=dict({'Content-Type': 'application/json'}, **(headers or {})),
                                 method=('GET' if body is None else 'POST'))
    if body is None:
        req = urllib.request.Request(url + path, headers=headers or {}, method='GET')
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read().decode()
    except urllib.error.HTTPError as err:
        return err.code, {k.lower(): v for k, v in err.headers.items()}, err.read().decode()


class _ProtoBase(unittest.IsolatedAsyncioTestCase):
    latency = 0.0
    respond = None

    async def asyncSetUp(self):
        kw = {'latency': self.latency}
        if self.respond is not None:
            kw['respond'] = self.respond
        self.rig = await Rig('copula-proto-%s' % self.latency, **kw).start()
        self.dir = tempfile.mkdtemp(prefix='copula-proto-')
        self.reg = Registry(os.path.join(self.dir, 'registry.db'))
        self.s = await Session.connect(self.rig.socket)
        self.proxy = Proxy(self.s, self.reg, {'m': 'actor'}, keepalive=0.05)
        await self.proxy.start()

    async def asyncTearDown(self):
        await self.proxy.close()
        await self.s.aclose()
        await self.rig.close()
        self.reg.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def launch(self, version='v-1'):
        rec = self.reg.process_run('e', 'dev', 'search', None, None, version)
        return rec, self.proxy.url(rec)

    async def call(self, url, path, body, headers=None):
        loop = asyncio.get_running_loop()
        st, h, raw = await loop.run_in_executor(None, post, url, path, body, headers)
        try:
            return st, h, json.loads(raw)
        except ValueError:
            return st, h, raw



class ProxyProtocolsTest(_ProtoBase):
    async def test_chat_text_parts_and_developer_role(self):
        _, u1 = self.launch()
        plain = {'model': 'm', 'max_tokens': 16, 'temperature': 0.8,
                 'messages': [{'role': 'system', 'content': 'be brief'},
                              {'role': 'user', 'content': 'q'}]}
        parts = dict(plain, messages=[{'role': 'developer', 'content': 'be brief'},
                                      {'role': 'user', 'content': [{'type': 'text', 'text': 'q'}]}])
        st, h1, d1 = await self.call(u1, '/chat/completions', plain)
        self.assertEqual((st, h1['x-copula-status']), (200, 'generated'))
        _, u2 = self.launch('v-2')                               # an edit: shares slot 1
        st, h2, d2 = await self.call(u2, '/chat/completions', parts)
        self.assertEqual((st, h2['x-copula-status']), (200, 'hit'))     # the same request
        self.assertEqual(d2['choices'][0]['message']['content'],
                         d1['choices'][0]['message']['content'])
        two = dict(plain, messages=[{'role': 'user', 'content': [
            {'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}]}])
        joined = dict(plain, messages=[{'role': 'user', 'content': 'a\nb'}])
        _, h3, _ = await self.call(u1, '/chat/completions', two)
        _, h4, _ = await self.call(u2, '/chat/completions', joined)
        self.assertEqual((h3['x-copula-status'], h4['x-copula-status']), ('generated', 'hit'))
        image = dict(plain, messages=[{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}]}])
        st, _, d = await self.call(u1, '/chat/completions', image)
        self.assertEqual(st, 400)
        self.assertIn('text only', d['error']['message'])

    async def test_anthropic_shares_like_chat(self):
        body = {'model': 'm', 'max_tokens': 16, 'temperature': 0.8,
                'messages': [{'role': 'user', 'content': 'q'}]}
        _, u1 = self.launch()
        st, h1, d1 = await self.call(u1, '/messages', body)
        self.assertEqual((st, h1['x-copula-status'], d1['type']), (200, 'generated', 'message'))
        _, u2 = self.launch('v-2')                               # an edit: shares slot 1
        _, h2, d2 = await self.call(u2, '/messages', body)
        _, h3, raw = await self.call(u2, '/messages', dict(body, stream=True))
        self.assertEqual((h2['x-copula-status'], d2['content'], d2['id']),
                         ('hit', d1['content'], d1['id']))
        self.assertIn('event: message_stop', raw)                # a 2nd occurrence: new sample
        self.assertIn(': x-copula-status generated', raw)
        _, u3 = self.launch()                                    # a rerun: slot 2
        _, h4, d4 = await self.call(u3, '/messages', body)
        self.assertEqual(h4['x-copula-status'], 'generated')
        self.assertNotEqual(d4['content'], d1['content'])
        st, _, d = await self.call(u3.replace('/r/', '/r/x'), '/messages', body)
        self.assertEqual((st, d['type'], d['error']['type']), (401, 'error', 'authentication_error'))
        st, _, d = await self.call(u3, '/messages', dict(body, model='nope'))
        self.assertEqual((st, d['error']['type']), (404, 'not_found_error'))
        st, _, d = await self.call(u3, '/messages/count_tokens', body)
        self.assertEqual((st, d['type']), (404, 'error'))

    async def test_responses_chain_is_shared_after_an_edit(self):
        _, u1 = self.launch()
        _, _, r1 = await self.call(u1, '/responses', {'model': 'm', 'input': 'first',
                                                       'instructions': 'be brief'})
        _, _, r2 = await self.call(u1, '/responses', {'model': 'm', 'input': 'second',
                                                       'previous_response_id': r1['id']})
        _, u2 = self.launch('v-2')
        _, h3, r3 = await self.call(u2, '/responses', {'model': 'm', 'input': 'first',
                                                        'instructions': 'be brief'})
        _, h4, r4 = await self.call(u2, '/responses', {'model': 'm', 'input': 'second',
                                                        'previous_response_id': r3['id']})
        self.assertEqual((h3['x-copula-status'], h4['x-copula-status']), ('hit', 'hit'))
        self.assertEqual((r3['id'], r4['id']), (r1['id'], r2['id']))
        self.assertEqual(r4['output'][0]['content'][0]['text'],
                         r2['output'][0]['content'][0]['text'])
        st, _, got = await self.call(u2, '/responses/' + r2['id'], None)
        self.assertEqual((st, got['id']), (200, r2['id']))
        _, _, r5 = await self.call(u2, '/responses', {'model': 'm', 'input': 'private',
                                                       'store': False})
        st, _, _ = await self.call(u2, '/responses/' + r5['id'], None)
        self.assertEqual(st, 404)                                   # store: false is not listed
        st, _, d = await self.call(u2, '/responses', {'model': 'm', 'input': 'x',
                                                       'previous_response_id': 'resp_foreign'})
        self.assertEqual((st, d['error']['code']), (404, 'not_found'))
        _, _, raw = await self.call(u2, '/responses', {'model': 'm', 'input': 'first',
                                                        'instructions': 'be brief', 'stream': True})
        self.assertIn('event: response.completed', raw)
        self.assertNotIn('[DONE]', raw)


class ToolProtocolsTest(_ProtoBase):
    respond = staticmethod(lambda body: tools_reply(body, 'tok-0123456789'))

    async def test_anthropic_tool_round_trip(self):
        _, u = self.launch()
        tools = [{'name': 'look', 'description': 'd', 'input_schema': {'type': 'object'}}]
        msgs = [{'role': 'user', 'content': 'find x'}]
        _, _, d1 = await self.call(u, '/messages', {'model': 'm', 'max_tokens': 32,
                                                     'tools': tools, 'messages': msgs})
        self.assertEqual((d1['stop_reason'], d1['content'][0]['type']), ('tool_use', 'tool_use'))
        use = d1['content'][0]
        msgs += [{'role': 'assistant', 'content': d1['content']},
                 {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': use['id'],
                                               'content': 'x is 4'}]}]
        st, _, d2 = await self.call(u, '/messages', {'model': 'm', 'max_tokens': 32,
                                                      'tools': tools, 'messages': msgs})
        self.assertEqual((st, d2['stop_reason']), (200, 'end_turn'))
        self.assertTrue(d2['content'][0]['text'].startswith('final answer'))

    async def test_responses_tool_round_trip(self):
        _, u = self.launch()
        tools = [{'type': 'function', 'name': 'look', 'parameters': {'type': 'object'}}]
        _, _, r1 = await self.call(u, '/responses', {'model': 'm', 'input': 'find', 'tools': tools})
        call = r1['output'][0]
        self.assertEqual(call['type'], 'function_call')
        st, _, r2 = await self.call(u, '/responses', {
            'model': 'm', 'tools': tools, 'previous_response_id': r1['id'],
            'input': [{'type': 'function_call_output', 'call_id': call['call_id'], 'output': '4'}]})
        self.assertEqual((st, r2['output'][0]['type']), (200, 'message'))


class SlowProtocolsTest(_ProtoBase):
    latency = 0.4

    async def abandon(self, url, path, body, headers=None):
        host, port = self.proxy.base[len('http://'):].split(':')
        raw = json.dumps(body).encode()
        head = ['POST %s HTTP/1.1' % (url[len(self.proxy.base):] + path), 'Host: x',
                'Content-Type: application/json', 'Content-Length: %d' % len(raw)]
        head += ['%s: %s' % kv for kv in (headers or {}).items()]
        loop = asyncio.get_running_loop()

        def go():
            s = socket.create_connection((host, int(port)))
            s.sendall(('\r\n'.join(head) + '\r\n\r\n').encode() + raw)
            s.close()
        await loop.run_in_executor(None, go)

    async def test_anthropic_stream_pings_and_retry(self):
        _, u = self.launch()
        body = {'model': 'm', 'max_tokens': 16, 'messages': [{'role': 'user', 'content': 'p'}]}
        _, _, raw = await self.call(u, '/messages', dict(body, stream=True))
        self.assertLess(raw.index('event: ping'), raw.index('event: message_start'))
        await self.abandon(u, '/messages', body)
        await asyncio.sleep(0.1)
        _, h, _ = await self.call(u, '/messages', body, {'x-stainless-retry-count': '1'})
        self.assertIn('retry re-read', h['x-copula-status'])
        self.assertEqual(sum(self.rig.model.calls.values()), 2)



if __name__ == '__main__':
    unittest.main()
