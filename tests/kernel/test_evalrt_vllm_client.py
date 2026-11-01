"""evalrt's vLLM client against a fake OpenAI-compatible server (no GPU, no model):
    PYTHONPATH=src python3 -m unittest tests.kernel.test_evalrt_vllm_client
"""
import asyncio
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from copula.kernel.manager.flights import Manager, ManagerError
from copula.kernel.manager.vllm_client import VLLMClient, file_identity, server_identity
from copula.kernel.store.entries import EntryStore

DEFAULTS = {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 0.0,
            'frequency_penalty': 0.0, 'repetition_penalty': 1.1, 'stop': [],
            'chat_template_kwargs': {}}
BODY = {'messages': [{'role': 'user', 'content': '你好'}], 'max_tokens': 16}


class FakeServer:
    """Scripted replies: each entry of `script` is (status, json body, delay seconds)."""

    def __init__(self):
        self.script = []
        self.seen = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body):
                data = json.dumps(body).encode()
                try:
                    self.send_response(status)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass                    # the client timed out first: expected in one test

            def do_GET(self):
                if self.path == '/v1/models':
                    self._send(200, {'data': [{'id': 'tiny'}]})
                elif self.path == '/version':
                    self._send(200, {'version': '0.20.2'})
                else:
                    self._send(404, {})

            def do_POST(self):
                length = int(self.headers['Content-Length'])
                outer.seen.append(json.loads(self.rfile.read(length)))
                status, body, delay = outer.script.pop(0) if outer.script else (200, ok(), 0)
                time.sleep(delay)
                self._send(status, body)

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = 'http://127.0.0.1:%d' % self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def ok(content='答案', finish='stop'):
    return {'choices': [{'message': {'content': content, 'reasoning_content': None},
                         'finish_reason': finish}],
            'usage': {'prompt_tokens': 12, 'completion_tokens': 3}}


class VLLMClientTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = FakeServer()
        self.dir = tempfile.mkdtemp(prefix='evalrt-vllm-')
        self.client = VLLMClient(self.server.url, 'tiny', DEFAULTS, {'files': {'x': 'y'}},
                                 timeout=0.5)
        self.store = EntryStore(os.path.join(self.dir, 'entries.db'))
        self.manager = Manager(self.store, self.client, max_attempts=3, backoff_base=0.0)
        await self.manager.start()
        self.manager.register_ticket('T', 'exec/0', self.client.model_rev, {'c': 'ck'})

    async def asyncTearDown(self):
        await self.manager.stop()
        await self.client.aclose()
        self.store.close()
        self.server.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    async def ask(self, rep=1):
        return await self.manager.get_or_generate('T', 'c', 'node', rep, 0, BODY)

    async def test_every_parameter_is_sent_explicitly_and_no_seed(self):
        res = await self.ask()
        [sent] = self.server.seen
        self.assertEqual(sent['model'], 'tiny')
        for name, value in DEFAULTS.items():
            self.assertEqual(sent[name], value, name)
        self.assertNotIn('seed', sent)
        self.assertEqual(sent['messages'], BODY['messages'])
        self.assertEqual(res['sample'], {'content': '答案', 'reasoning': None,
                                         'finish_reason': 'stop', 'prompt_tokens': 12,
                                         'completion_tokens': 3})

    async def test_server_errors_are_retried_and_written_once(self):
        self.server.script = [(500, {'error': 'boom'}, 0), (503, {}, 0), (200, ok('x'), 0)]
        res = await self.ask()
        self.assertEqual((res['sample']['content'], res['attempts']), ('x', 3))
        self.assertEqual(self.store.count(), 1)

    async def test_request_determined_errors_become_error_samples(self):
        self.server.script = [(400, {'message': "This model's maximum context length is 32768"}, 0),
                              (400, {'message': 'top_k must be positive'}, 0)]
        first = await self.ask(1)
        second = await self.ask(2)
        self.assertEqual(first['sample']['error'], 'context_overflow')
        self.assertEqual(second['sample']['error'], 'http_400')
        self.assertEqual(len(self.server.seen), 2)                 # never retried

    async def test_abort_and_timeouts_are_transport_failures(self):
        self.server.script = [(200, ok('', 'abort'), 0), (200, ok('late'), 1.0),
                              (200, ok('fine'), 0)]
        res = await self.ask()
        self.assertEqual((res['sample']['content'], res['attempts']), ('fine', 3))

    async def test_exhausted_retries_write_nothing(self):
        self.server.script = [(502, {}, 0)] * 3
        with self.assertRaises(ManagerError) as ctx:
            await self.ask()
        self.assertEqual(ctx.exception.kind, 'flight_failed')
        self.assertEqual(self.store.count(), 0)

    async def test_replicas_share_one_model_rev_and_split_requests(self):
        other = FakeServer()
        try:
            both = VLLMClient(self.server.url + ',' + other.url, 'tiny', DEFAULTS,
                              {'files': {'x': 'y'}}, timeout=0.5)
            self.assertEqual(both.model_rev, self.client.model_rev)
            for _ in range(4):
                await asyncio.get_running_loop().run_in_executor(None, both._post, BODY)
            self.assertEqual((len(self.server.seen), len(other.seen)), (2, 2))
            await both.aclose()
        finally:
            other.close()

    def test_model_rev_covers_model_identity_and_defaults(self):
        base = VLLMClient('http://x', 'tiny', DEFAULTS, {'files': {'a': '1'}}).model_rev
        self.assertNotEqual(base, VLLMClient('http://x', 'other', DEFAULTS,
                                             {'files': {'a': '1'}}).model_rev)
        self.assertNotEqual(base, VLLMClient('http://x', 'tiny', dict(DEFAULTS, top_p=0.9),
                                             {'files': {'a': '1'}}).model_rev)
        self.assertNotEqual(base, VLLMClient('http://x', 'tiny', DEFAULTS,
                                             {'files': {'a': '2'}}).model_rev)
        self.assertEqual(base, VLLMClient('http://y', 'tiny', DEFAULTS,
                                          {'files': {'a': '1'}}).model_rev)   # the URL is not

    def test_identity_of_files_and_server(self):
        d = tempfile.mkdtemp(prefix='evalrt-model-')
        try:
            os.makedirs(os.path.join(d, 'blobs'))
            blob = 'a' * 64
            with open(os.path.join(d, 'blobs', blob), 'w') as handle:
                handle.write('weights')
            with open(os.path.join(d, 'config.json'), 'w') as handle:
                handle.write('{}')
            os.symlink(os.path.join(d, 'blobs', blob), os.path.join(d, 'model.safetensors'))
            files = file_identity(d)
            self.assertEqual(files['model.safetensors'], blob)      # the blob name, not rehashed
            self.assertEqual(len(files['config.json']), 64)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        self.assertEqual(server_identity(self.server.url),
                         {'models': ['tiny'], 'vllm_version': '0.20.2'})


if __name__ == '__main__':
    unittest.main()
