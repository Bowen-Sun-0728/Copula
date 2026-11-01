"""The model client for a vLLM OpenAI-compatible server (RUNTIME_V0 section 13, build step 4).

It sends exactly the normalised body the manager hashed, plus the served model name. Every sampling
parameter is explicit, so the server's own defaults -- a model's generation_config.json, which vLLM
applies by default -- can never change a sample silently. No per-request seed is ever sent (section
4.2). The server itself must be started with a fresh random `--seed`: vLLM's default engine seed is
0, so every restart would replay one random stream and two "independent" runs could draw the same
samples.

Failures follow section 12.5. Connection errors, timeouts, HTTP 408/409/429 and 5xx are
TransportError: the manager retries them and writes nothing. A 4xx the request itself determines
(context too long, a parameter the server refuses) is RequestError and becomes an error sample. A
finish reason other than stop or length (abort) is TransportError. The timeout sits far above the
longest legitimate generation, so it cannot select samples by length.

`model_rev` hashes what the output distribution depends on besides the request: the served model,
its weight/tokenizer/chat-template files, the vLLM version, the serving settings that change numerics
(dtype, quantisation, KV-cache dtype) and the pinned default table.

Metering (campaign 2026-10-02, SCHEMA sections 2-4): `generate_with_meta` sends the ticket's
`cache_salt` as a top-level request field (vLLM folds it into the hash of the first KV block only, so
it isolates prefix caching without touching the body, the key or `model_rev`) and returns the
attempt's meta: timing, replica, HTTP status, reply id and token counts. vLLM 0.20.2 omits
`usage.prompt_tokens_details` when nothing was cached, so a missing cached count is 0, never an error
(PLAN D36); the manager's startup probe (`probe`) checks that the server reports the field at all.
`route='case'` pins a case to one replica (its prefix cache), failing over on retries.

Replica failover (sysval E4, 2026-10-09): a replica whose connection fails or times out is skipped
for a cooldown (`DOWN_S`, doubling while it keeps failing, at most `DOWN_MAX_S`) whenever another
replica is up, and a retry never goes back to the replica that just failed while another is up. A
dead replica therefore costs one failed attempt per cooldown instead of failing the flights whose
retries all landed on it. Which replica serves an attempt never depends on the output.

Scoring backends (models INTERFACE J4, 2026-10-09): `RewardClient` sends the request to vLLM's
`/pooling` endpoint of a reward model (a process reward model such as Qwen2.5-Math-PRM). Its input is
the single user message's text (`reward_input='raw'`: the graph's template renders the chat markup)
or the messages (`'chat'`). The sample's content is the JSON list of per-step scores -- the
positive-class probability at each step tag -- and `completion_tokens` is 0. Its pinned table has
temperature 0, so the manager memoises every scoring request (deterministic + cache).
"""
import asyncio
import functools
import hashlib
import http.client
import itertools
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from ..core.canon import digest
from ..core.request import check_defaults, normalize_request
from .model import ModelClient, RequestError, TransportError

RETRY_STATUS = (408, 409, 429)
DOWN_S = 2.0                    # a failed replica is skipped this long (doubling while it fails)
DOWN_MAX_S = 30.0
ROUTES = ('rr', 'case')
PROBE_TEXT = ('The quick brown fox jumps over the lazy dog while the five boxing wizards jump '
              'quickly and a sphinx of black quartz judges my vow. ')
MODEL_FILES = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
               'vocab.json', 'merges.txt', 'chat_template.jinja', 'chat_template.json',
               'model.safetensors.index.json', 'model.safetensors')


def file_identity(model_dir):
    """sha256 of the files that shape the distribution. A Hugging Face snapshot links each file to a
    blob named by its sha256, which is used as is instead of rehashing gigabytes of weights."""
    out = {}
    for name in MODEL_FILES:
        path = os.path.join(model_dir, name)
        if not os.path.exists(path):
            continue
        target = os.path.realpath(path)
        blob = os.path.basename(target)
        if os.path.islink(path) and len(blob) == 64 and all(c in '0123456789abcdef' for c in blob):
            out[name] = blob
            continue
        sha = hashlib.sha256()
        with open(target, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                sha.update(chunk)
        out[name] = sha.hexdigest()
    if 'config.json' not in out:
        raise ValueError('%s has no config.json' % model_dir)
    return out


def _get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as handle:
        return handle.read().decode('utf-8')


def server_identity(base_url, timeout=10.0):
    """What the running server reports: its model ids and its vLLM version."""
    models = json.loads(_get(base_url.rstrip('/') + '/v1/models', timeout))
    try:
        version = json.loads(_get(base_url.rstrip('/') + '/version', timeout)).get('version')
    except (urllib.error.URLError, OSError, ValueError):
        version = None
    return {'models': sorted(m['id'] for m in models.get('data', [])), 'vllm_version': version}


def usage_counts(reply):
    """Token counts of a reply: {prompt_tokens, completion_tokens, cached_tokens, cached_present};
    all None when the reply has no usage. A missing `prompt_tokens_details` means 0 cached tokens
    (vLLM 0.20.2 omits it when nothing was cached); `cached_present` says whether it was there."""
    usage = reply.get('usage') if isinstance(reply, dict) else None
    if not isinstance(usage, dict):
        return {'prompt_tokens': None, 'completion_tokens': None, 'cached_tokens': None,
                'cached_present': None}
    details = usage.get('prompt_tokens_details')

    def count(value):
        return value if type(value) is int else None
    return {'prompt_tokens': count(usage.get('prompt_tokens')),
            'completion_tokens': count(usage.get('completion_tokens')),
            'cached_tokens': int((details if isinstance(details, dict) else {}).get(
                'cached_tokens') or 0),
            'cached_present': isinstance(details, dict) and 'cached_tokens' in details}


def probe_text(chars):
    """A deterministic ASCII prompt of exactly `chars` characters (the startup probe)."""
    return (PROBE_TEXT * (chars // len(PROBE_TEXT) + 1))[:chars]


def _is_timeout(err):
    reason = getattr(err, 'reason', None)
    return isinstance(err, (socket.timeout, TimeoutError)) or isinstance(
        reason, (socket.timeout, TimeoutError))


def _failed(err, meta):
    err.meta = meta
    return err


class VLLMClient(ModelClient):
    kind = 'generate'
    endpoint = '/v1/chat/completions'

    """`base_url` may list several replicas of one server configuration, comma separated; requests
    go round robin (`route='rr'`) or by case (`route='case'`: the replica hashed from the case key,
    the next one on each retry). The replicas must report the same identity (checked by the
    caller), so the model revision, and with it every cache key, is that of one server. `workers`
    bounds the requests in flight from this process (the manager passes its --inflight)."""

    def __init__(self, base_url, served_model, defaults, identity, timeout=900.0, workers=64,
                 route='rr'):
        if route not in ROUTES:
            raise ValueError('route is one of %s' % (ROUTES,))
        self.urls = [u.strip().rstrip('/') for u in base_url.split(',') if u.strip()]
        self.base_url = self.urls[0]
        self._next = itertools.count()
        self.served_model = served_model
        self.defaults = check_defaults(defaults)
        self.identity = dict(identity)
        self.model_rev = 'vllm-' + digest({'served_model': served_model, 'identity': self.identity,
                                           'defaults': self.defaults})[:24]
        self.timeout = timeout
        self.workers = workers
        self.route = route
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='vllm')
        self._health = threading.Lock()
        self._down = {}                 # replica -> (skipped until, current cooldown)

    async def generate(self, body, key_hash):
        sample, _ = await asyncio.get_running_loop().run_in_executor(self._pool, self._post, body)
        return sample

    async def generate_with_meta(self, body, key_hash, ctx):
        ctx = ctx or {}
        call = functools.partial(self._post, body, ctx.get('salt'), ctx.get('route'),
                                 ctx.get('attempt') or 1, avoid=ctx.get('failed_replica'))
        return await asyncio.get_running_loop().run_in_executor(self._pool, call)

    async def aclose(self):
        self._pool.shutdown(wait=False)

    def payload(self, body, salt=None):
        """The request as sent: the normalised body, every parameter explicit, and the model; the
        cache salt only when there is one (salt None: today's bytes exactly)."""
        out = dict(body)
        out['model'] = self.served_model
        if salt is not None:
            out['cache_salt'] = salt
        return out

    def _up(self, avoid=None):
        """Replicas not in their cooldown, and not `avoid` (the one that just failed), when any
        such replica exists; otherwise every replica."""
        now = time.monotonic()
        with self._health:
            up = [r for r in range(len(self.urls))
                  if self._down.get(r, (0, 0))[0] <= now and r != avoid]
        return set(up) if up else set(range(len(self.urls)))

    def mark_down(self, replica):
        """Start (or, for a replica failing again after its cooldown, double) its cooldown. The
        concurrent failures of one outage do not compound."""
        now = time.monotonic()
        with self._health:
            until, cool = self._down.get(replica, (0, 0))
            if until > now:
                return
            cool = min(DOWN_MAX_S, cool * 2) if cool else DOWN_S
            self._down[replica] = (now + cool, cool)

    def mark_up(self, replica):
        with self._health:
            self._down.pop(replica, None)

    def replica_for(self, route=None, attempt=1, avoid=None):
        """Index of the replica an attempt goes to: round robin, or by case (the next one on each
        retry), over the replicas that are up (see `_up`)."""
        up = self._up(avoid)
        n = len(self.urls)
        if self.route == 'case' and route is not None:
            first = int(hashlib.sha256(route.encode('utf-8')).hexdigest()[:8], 16)
            for k in range(n):
                r = (first + attempt - 1 + k) % n
                if r in up:
                    return r
        for _ in range(n):
            r = next(self._next) % n
            if r in up:
                return r
        return next(self._next) % n

    def _post(self, body, salt=None, route=None, attempt=1, replica=None, avoid=None):
        """One HTTP attempt: (sample, meta). Failures raise with `.meta` set."""
        if replica is None:
            replica = self.replica_for(route, attempt, avoid)
        url = self.urls[replica]
        meta = {'t_send': None, 't_done': None, 'instance_url': url, 'replica': replica,
                'salt': salt, 'http_status': None, 'error': None, 'response_id': None,
                'prompt_tokens': None, 'completion_tokens': None, 'cached_tokens': None,
                'cached_present': None, 'finish_reason': None}
        data = json.dumps(self.payload(body, salt), ensure_ascii=False).encode('utf-8')
        request = urllib.request.Request(url + self.endpoint, data=data,
                                         method='POST',
                                         headers={'Content-Type': 'application/json'})
        meta['t_send'] = time.time()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                raw = handle.read()
                status = getattr(handle, 'status', 200)
        except urllib.error.HTTPError as err:
            meta.update(t_done=time.time(), http_status=err.code)
            detail = err.read()[:500].decode('utf-8', 'replace')
            if err.code in RETRY_STATUS or err.code >= 500:
                meta['error'] = 'http_%d' % err.code
                raise _failed(TransportError('HTTP %d: %s' % (err.code, detail)), meta)
            text = detail.lower()
            if 'context length' in text or 'maximum context' in text or 'too long' in text:
                meta['error'] = 'context_overflow'
                raise _failed(RequestError('context_overflow', detail), meta)
            meta['error'] = 'http_%d' % err.code
            raise _failed(RequestError('http_%d' % err.code, detail), meta)
        except (urllib.error.URLError, socket.timeout, OSError, http.client.HTTPException,
                ValueError) as err:
            meta.update(t_done=time.time(), error='timeout' if _is_timeout(err) else 'connection')
            self.mark_down(replica)
            raise _failed(TransportError('%s: %s' % (type(err).__name__, str(err)[:300])), meta)
        self.mark_up(replica)
        meta.update(t_done=time.time(), http_status=status)
        try:
            reply = json.loads(raw.decode('utf-8'))
        except ValueError as err:
            meta['error'] = 'bad_json'
            raise _failed(TransportError('%s: %s' % (type(err).__name__, str(err)[:300])), meta)
        try:
            sample = self.sample_of(reply)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as err:
            meta['error'] = 'bad_reply'
            raise _failed(TransportError('unexpected reply shape: %r' % (err,)), meta)
        meta.update(usage_counts(reply), response_id=reply.get('id'),
                    finish_reason=sample['finish_reason'])
        return sample, meta

    @staticmethod
    def sample_of(reply):
        """The sample in a chat-completions reply (raises on another shape)."""
        choice = reply['choices'][0]
        message = choice.get('message') or {}
        usage = reply.get('usage') or {}
        out = {'content': message.get('content') or '',
               'reasoning': message.get('reasoning_content') or message.get('reasoning'),
               'finish_reason': choice.get('finish_reason'),
               'prompt_tokens': int(usage.get('prompt_tokens', 0)),
               'completion_tokens': int(usage.get('completion_tokens', 0))}
        if message.get('tool_calls'):           # the request's tools, called (absent: as before)
            out['tool_calls'] = [{'id': c['id'], 'type': 'function',
                                  'function': {'name': c['function']['name'],
                                               'arguments': c['function'].get('arguments') or ''}}
                                 for c in message['tool_calls']]
        return out

    def probe(self, replica, chars=8000, salt=None):
        """Does the server report cached prompt tokens (`--enable-prompt-tokens-details`)? One
        deterministic prompt of `chars` characters, every default explicit, `max_tokens` 1, sent
        twice in a row under one fresh salt: the second reply must show a cached prefix.
        Returns (ok, normalised body, salt, [meta of each attempt])."""
        body = normalize_request({'messages': [{'role': 'user', 'content': probe_text(chars)}],
                                  'max_tokens': 1}, self.defaults)
        salt = salt or 'probe-' + os.urandom(16).hex()
        metas = []
        for attempt in (1, 2):
            try:
                _, meta = self._post(body, salt, attempt=attempt, replica=replica)
            except (TransportError, RequestError) as err:
                meta = err.meta
            metas.append(meta)
        last = metas[-1] or {}
        ok = bool(last.get('cached_present')) and (last.get('cached_tokens') or 0) >= 1
        return ok, body, salt, metas


REWARD_INPUTS = ('raw', 'chat')


def step_scores(reply):
    """Per-step scores of a `/pooling` reply: for each pooled position (a step tag), the
    probability of the positive class (index 1 of a [negative, positive] pair) or the value itself
    when the server returns one number per position."""
    data = reply['data'][0]['data']
    if not isinstance(data, list):
        raise ValueError('pooling data is not a list')
    out = []
    for item in data:
        if isinstance(item, list):
            out.append(float(item[1] if len(item) > 1 else item[0]))
        else:
            out.append(float(item))
    return out


class RewardClient(VLLMClient):
    """A reward model behind vLLM's `/pooling` endpoint (models INTERFACE J4). Same transport,
    retries, replicas and metering as `VLLMClient`; the sampling parameters of the normalised body
    are pinned (temperature 0) and not sent, since pooling does not sample."""
    kind = 'reward'
    endpoint = '/pooling'

    def __init__(self, base_url, served_model, defaults, identity, timeout=900.0, workers=64,
                 route='rr', reward_input='raw'):
        if reward_input not in REWARD_INPUTS:
            raise ValueError('reward_input is one of %s' % (REWARD_INPUTS,))
        super().__init__(base_url, served_model, defaults, identity, timeout=timeout,
                         workers=workers, route=route)
        if self.defaults['temperature'] != 0.0:
            raise ValueError('a reward model is pinned to temperature 0')
        self.reward_input = reward_input
        self.model_rev = 'reward-' + digest({'served_model': served_model,
                                             'identity': self.identity, 'defaults': self.defaults,
                                             'input': reward_input})[:24]

    def payload(self, body, salt=None):
        messages = body['messages']
        if self.reward_input == 'raw':
            if len(messages) != 1 or messages[0]['role'] != 'user':
                raise RequestError('reward_input', 'raw reward input is one user message')
            out = {'model': self.served_model, 'input': messages[0]['content']}
        else:
            out = {'model': self.served_model, 'messages': messages}
        return out                      # no cache salt: pooling requests do not take one

    @staticmethod
    def sample_of(reply):
        usage = reply.get('usage') or {}
        return {'content': json.dumps(step_scores(reply)), 'reasoning': None,
                'finish_reason': 'stop', 'prompt_tokens': int(usage.get('prompt_tokens', 0)),
                'completion_tokens': 0}

    def probe_scores(self, text='1 + 1 = 2.'):
        """Does every replica answer a scoring request with scores? A misconfigured server (one
        not serving a pooling model) would otherwise turn every request into an error sample stored
        under a revision that does not change once the server is fixed. Returns (ok, detail)."""
        messages = [{'role': 'user', 'content': text}]
        body = normalize_request({'messages': messages, 'max_tokens': 1}, self.defaults)
        for replica in range(len(self.urls)):
            try:
                self._post(body, replica=replica)
            except (TransportError, RequestError) as err:
                return False, '%s: %s' % (self.urls[replica], err)
        return True, ''
