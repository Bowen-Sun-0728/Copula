"""A fake OpenAI-compatible upstream for tests: kind 'vllm' (owned_by vllm, /version, `root`) or
'api' (owned_by openai, strict: refuses parameters the OpenAI API does not have). It records every
chat request (payload and headers). The reply text names the seed and temperature it was sent."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

API_FIELDS = {'model', 'messages', 'temperature', 'top_p', 'n', 'stop', 'max_tokens',
              'max_completion_tokens', 'presence_penalty', 'frequency_penalty', 'seed', 'tools',
              'tool_choice', 'response_format', 'parallel_tool_calls', 'user', 'logit_bias'}


class FakeUpstream:
    def __init__(self, kind='vllm', models=('fake-model',), root=None, key=None, roots=None,
                 parents=None, completion_tokens=False, open_listing=False):
        self.kind, self.models, self.root, self.key = kind, list(models), root, key
        self.roots, self.parents = dict(roots or {}), dict(parents or {})
        self.completion_tokens = completion_tokens      # refuse max_tokens (reasoning models)
        self.open_listing = open_listing                # GET /v1/models needs no key
        self.requests = []
        self.lock = threading.Lock()
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _auth_ok(self):
                return outer.key is None or self.headers.get('Authorization') == 'Bearer ' + outer.key

            def do_GET(self):
                if not self._auth_ok() and not outer.open_listing:
                    return self._json(401, {'error': {'message': 'bad key'}})
                if self.path == '/v1/models':
                    owner = 'vllm' if outer.kind == 'vllm' else 'openai'
                    return self._json(200, {'object': 'list', 'data': [
                        {'id': m, 'object': 'model', 'owned_by': owner,
                         'root': outer.roots.get(m, outer.root or m),
                         'parent': outer.parents.get(m)} for m in outer.models]})
                if self.path == '/version' and outer.kind == 'vllm':
                    return self._json(200, {'version': '0.0.0-fake'})
                return self._json(404, {'error': {'message': 'no'}})

            def do_POST(self):
                n = int(self.headers.get('Content-Length') or 0)
                body = json.loads(self.rfile.read(n).decode())
                with outer.lock:
                    outer.requests.append({'body': body, 'auth': self.headers.get('Authorization')})
                    count = len(outer.requests)
                if not self._auth_ok():
                    return self._json(401, {'error': {'message': 'bad key'}})
                if outer.completion_tokens and 'max_tokens' in body:
                    return self._json(400, {'error': {'message': "Unsupported parameter: "
                                      "'max_tokens' is not supported with this model. Use "
                                      "'max_completion_tokens' instead."}})
                if outer.kind == 'api':
                    bad = sorted(set(body) - API_FIELDS)
                    if bad:
                        return self._json(400, {'error': {'message': 'Unrecognized request '
                                                          'arguments: %s' % bad}})
                text = 'reply %d seed=%s temperature=%s' % (count, body.get('seed'),
                                                             body.get('temperature'))
                return self._json(200, {
                    'id': 'cmpl-%d' % count, 'object': 'chat.completion', 'model': body['model'],
                    'choices': [{'index': 0, 'finish_reason': 'stop',
                                 'message': {'role': 'assistant', 'content': text}}],
                    'usage': {'prompt_tokens': 3, 'completion_tokens': 5, 'total_tokens': 8}})

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.url = 'http://127.0.0.1:%d/v1' % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def chats(self):
        with self.lock:
            return list(self.requests)
