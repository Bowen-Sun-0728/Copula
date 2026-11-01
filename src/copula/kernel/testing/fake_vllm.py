"""A stand-in for a vLLM OpenAI-compatible server, for tests and GPU-free smokes of the campaign's
metering (2026-10-02, PLAN A-T). No model: replies are short fixed texts.

What it imitates is what the metering depends on:
- a prefix cache per (`cache_salt`, prompt): the first request of a prompt under a salt has nothing
  cached and, as vLLM 0.20.2 does, its usage has no `prompt_tokens_details` at all; a repeat
  reports the cached whole blocks (`block` tokens each). `details=False` never reports the field (a
  server started without --enable-prompt-tokens-details);
- /v1/models, /version, and /metrics with the counters instmon and the cost checks read;
- start-up log lines with the KV-cache size and the block size (the watchdog's identity).

Tokens are counted as len(text) // 4. `script` entries (status, body) are served first, in order.

Fault injection (sysval E3/E4, 2026-10-09): `--fail-rate` answers a request with HTTP 503 and
`--reset-rate` drops its connection without a reply, each with that probability (seeded by
`--fault-seed`); `--tag` makes replies unique across replicas ('Answer: <tag>-<n>'); `--jitter` adds
a uniform extra latency.

    python -m evalrt.testing.fake_vllm --port P [--served NAME] [--seed N] [--no-details] \\
        [--kv-tokens N | --kv-file PATH] [--block 16] [--latency S] [--jitter S] [--tag T] \\
        [--fail-rate P] [--reset-rate P] [--fault-seed N]
"""
import argparse
import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeVLLM:
    def __init__(self, served='tiny', details=True, block=16, version='0.20.2', latency=0.0,
                 host='127.0.0.1', port=0, jitter=0.0, tag=None, fail_rate=0.0, reset_rate=0.0,
                 fault_seed=None):
        self.served = served
        self.details = details
        self.block = block
        self.version = version
        self.latency = latency
        self.jitter = jitter
        self.tag = tag
        self.fail_rate = fail_rate
        self.reset_rate = reset_rate
        self.rng = random.Random(fault_seed)
        self.payloads = []
        self.script = []
        self.cache = set()
        self.counters = {'requests': 0, 'prompt': 0, 'cached': 0, 'generation': 0, 'running': 0,
                         'queries': 0, 'hits': 0}
        self.lock = threading.Lock()
        self.httpd = ThreadingHTTPServer((host, port), self._handler())
        self.httpd.daemon_threads = True
        self.url = 'http://%s:%d' % (host, self.httpd.server_address[1])
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def metrics(self):
        c = self.counters
        labels = 'engine="0",model_name="%s"' % self.served
        lines = [('vllm:num_requests_running', '', c['running']),
                 ('vllm:num_requests_waiting', '', 0),
                 ('vllm:kv_cache_usage_perc', '', 0.0),
                 ('vllm:num_preemptions_total', '', 0),
                 ('vllm:prompt_tokens_total', '', c['prompt']),
                 ('vllm:prompt_tokens_by_source_total', ',source="local_cache_hit"', c['cached']),
                 ('vllm:prompt_tokens_by_source_total', ',source="local_compute"',
                  c['prompt'] - c['cached']),
                 ('vllm:prefix_cache_queries_total', '', c['queries']),
                 ('vllm:prefix_cache_hits_total', '', c['hits']),
                 ('vllm:generation_tokens_total', '', c['generation'])]
        return ''.join('%s{%s%s} %s\n' % (name, labels, extra, float(value))
                       for name, extra, value in lines)

    def reply(self, payload):
        """(status, body) for one chat completion."""
        with self.lock:
            self.payloads.append(payload)
            self.counters['requests'] += 1
            n = self.counters['requests']
            if self.script:
                return self.script.pop(0)
        text = ''.join(m.get('content') or '' for m in payload.get('messages', []))
        prompt = max(1, len(text) // 4)
        completion = int(payload.get('min_tokens') or 3)
        key = (payload.get('cache_salt'), text)
        with self.lock:
            hit = key in self.cache
            self.cache.add(key)
            cached = prompt - prompt % self.block if hit else 0
            c = self.counters
            c['prompt'] += prompt
            c['cached'] += cached
            c['generation'] += completion
            c['queries'] += prompt
            c['hits'] += cached
        usage = {'prompt_tokens': prompt, 'completion_tokens': completion,
                 'total_tokens': prompt + completion}
        if cached and self.details:
            usage['prompt_tokens_details'] = {'cached_tokens': cached}
        return 200, {'id': 'chatcmpl-%d' % n, 'object': 'chat.completion', 'usage': usage,
                     'choices': [{'index': 0, 'finish_reason': 'stop',
                                  'message': {'role': 'assistant', 'content': (
                                      'Answer: %d' % n if self.tag is None
                                      else 'Answer: %s-%d' % (self.tag, n))}}]}

    def _handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, data, kind='application/json'):
                try:
                    self.send_response(status)
                    self.send_header('Content-Type', kind)
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                if self.path == '/v1/models':
                    body = {'object': 'list', 'data': [{'id': outer.served, 'object': 'model'}]}
                elif self.path == '/version':
                    body = {'version': outer.version}
                elif self.path == '/metrics':
                    self._send(200, outer.metrics().encode(), 'text/plain')
                    return
                elif self.path == '/counters':
                    with outer.lock:
                        body = dict(outer.counters)
                else:
                    self._send(404, b'{}')
                    return
                self._send(200, json.dumps(body).encode())

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with outer.lock:
                    outer.counters['running'] += 1
                    fault = outer.rng.random()
                    extra = outer.rng.random() * outer.jitter
                try:
                    if outer.latency or extra:
                        time.sleep(outer.latency + extra)
                    if fault < outer.reset_rate:
                        with outer.lock:
                            outer.counters['resets'] = outer.counters.get('resets', 0) + 1
                        self.close_connection = True
                        self.connection.shutdown(2)
                        return
                    if fault < outer.reset_rate + outer.fail_rate:
                        with outer.lock:
                            outer.counters['failed'] = outer.counters.get('failed', 0) + 1
                        self._send(503, b'{"error": "injected"}')
                        return
                    status, body = outer.reply(payload)
                finally:
                    with outer.lock:
                        outer.counters['running'] -= 1
                self._send(status, json.dumps(body).encode())
        return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m evalrt.testing.fake_vllm',
                                 description=__doc__.split('\n\n')[0])
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--served', default='tiny')
    ap.add_argument('--seed', default=None, help='accepted and logged, like vLLM --seed')
    ap.add_argument('--no-details', action='store_true')
    ap.add_argument('--kv-tokens', type=int, default=720192)
    ap.add_argument('--kv-file', default=None, help='read the KV-cache size from this file')
    ap.add_argument('--block', type=int, default=16)
    ap.add_argument('--num-gpu-blocks-override', type=int, default=None,
                    help='accepted and logged, like vLLM (the KV size still comes from --kv-*)')
    ap.add_argument('--latency', type=float, default=0.0)
    ap.add_argument('--jitter', type=float, default=0.0)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--fail-rate', type=float, default=0.0)
    ap.add_argument('--reset-rate', type=float, default=0.0)
    ap.add_argument('--fault-seed', type=int, default=None)
    ap.add_argument('--startup', type=float, default=0.3, help='seconds before serving')
    args = ap.parse_args(argv)
    kv = args.kv_tokens
    if args.kv_file:
        with open(args.kv_file) as handle:
            kv = int(handle.read())
    print('INFO fake vLLM seed=%s num_gpu_blocks_override=%s'
          % (args.seed, args.num_gpu_blocks_override), flush=True)
    print('INFO [interface.py] Setting attention block size to %d tokens' % args.block,
          flush=True)
    print('INFO [kv_cache_utils.py] GPU KV cache size: {:,} tokens, Maximum concurrency'
          .format(kv), flush=True)
    time.sleep(args.startup)
    server = FakeVLLM(args.served, details=not args.no_details, block=args.block,
                      latency=args.latency, port=args.port, jitter=args.jitter, tag=args.tag,
                      fail_rate=args.fail_rate, reset_rate=args.reset_rate,
                      fault_seed=args.fault_seed)
    print('INFO serving on %s' % server.url, flush=True)
    server.httpd.serve_forever()


if __name__ == '__main__':
    sys.exit(main())
