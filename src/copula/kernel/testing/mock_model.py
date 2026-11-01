"""A stand-in for the model server: a sample is a function of (nonce, key). No GPU.

Because a sample depends only on its key, two schedules of the same jobs must produce identical
samples and identical call counts, which is what the contract tests check. Latency and faults are
injectable per key and per attempt; `calls` counts every attempt that reached the model.
"""
import asyncio
import time
from collections import Counter

from ..core.canon import canonical, digest
from ..manager.model import ModelClient, RequestError, TransportError

MOCK_DEFAULTS = {
    'temperature': 1.0, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0, 'presence_penalty': 0.0,
    'frequency_penalty': 0.0, 'repetition_penalty': 1.0, 'stop': [],
    'chat_template_kwargs': {'enable_thinking': False},
}

FAULTS = ('transport', 'timeout', 'request', 'malformed', 'crash')


class MockModel(ModelClient):
    """`latency` and `fault` are constants or callables of (key_hash, attempt); `body_fault`, a
    callable of the request body, picks faults by content (say, one node's prompt); `respond`, a
    callable of the body, scripts replies: text, a dict of sample fields (`tool_calls`,
    `finish_reason`, ...), or None, which falls back to the key's content.

    Fault kinds: 'transport' and 'timeout' raise TransportError, 'request' raises RequestError,
    'malformed' returns an invalid sample, 'crash' raises an unclassified exception. `bodies` keeps
    the last body sent under each key, so tests can compare requests byte for byte.

    `meta` (metering, SCHEMA section 3.1): None returns no meta (today's records); True a synthetic
    meta per attempt (`instance_url` mock://0, no cached tokens, the field absent as vLLM 0.20.2
    sends it); a callable (body, key_hash, ctx) -> dict that dict. `contexts` collects the
    (key_hash, ctx) of every attempt, the cache salt included.
    """

    def __init__(self, nonce='mock', latency=0.0, fault=None, defaults=None, body_fault=None,
                 respond=None, meta=None):
        self.nonce = nonce
        self.defaults = dict(MOCK_DEFAULTS if defaults is None else defaults)
        self.model_rev = 'mock-' + digest({'mock': nonce, 'defaults': self.defaults})[:16]
        self.latency = latency
        self.fault = fault
        self.body_fault = body_fault
        self.respond = respond          # callable(body) -> content or None: scripted replies
        self.calls = Counter()
        self.bodies = {}
        self.active = 0
        self.max_active = 0
        self.meta = meta
        self.contexts = []

    def sample_for(self, key_hash):
        """The content the mock returns for `key_hash` (what a test expects)."""
        return 'mock:' + digest({'nonce': self.nonce, 'key': key_hash})[:24]

    async def generate(self, body, key_hash):
        self.calls[key_hash] += 1
        self.bodies[key_hash] = body
        attempt = self.calls[key_hash]
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            delay = self.latency(key_hash, attempt) if callable(self.latency) else self.latency
            if delay:
                await asyncio.sleep(delay)
            kind = self.fault(key_hash, attempt) if callable(self.fault) else self.fault
            if kind is None and self.body_fault is not None:
                kind = self.body_fault(body)
        finally:
            self.active -= 1
        if kind in ('transport', 'timeout'):
            raise TransportError('mock %s' % kind)
        if kind == 'request':
            raise RequestError('context_overflow', 'mock request error')
        if kind == 'malformed':
            return {'content': None}
        if kind == 'crash':
            raise RuntimeError('mock client bug')
        content = self.respond(body) if self.respond is not None else None
        if content is None:
            content = self.sample_for(key_hash)
        sample = {'content': content, 'reasoning': None, 'finish_reason': 'stop',
                  'prompt_tokens': len(canonical(body)) // 4, 'completion_tokens': 8}
        if isinstance(content, dict):           # a scripted sample (tool calls, say)
            sample.update(content)
            sample.setdefault('content', '')
            if sample.get('content') is None:
                sample['content'] = ''
        return sample

    def _synthetic(self, key_hash, ctx, t_send, sample=None, error=None):
        sample = sample if isinstance(sample, dict) else {}
        attempt = ctx.get('attempt') or 1
        return {'t_send': t_send, 't_done': time.time(), 'instance_url': 'mock://0', 'replica': 0,
                'salt': ctx.get('salt'), 'http_status': None if error else 200, 'error': error,
                'response_id': None if error else 'mock-%s-%d' % (key_hash[:12], attempt),
                'prompt_tokens': None if error else sample.get('prompt_tokens'),
                'completion_tokens': None if error else sample.get('completion_tokens'),
                'cached_tokens': None if error else 0,
                'cached_present': None if error else False,
                'finish_reason': None if error else sample.get('finish_reason')}

    async def generate_with_meta(self, body, key_hash, ctx):
        ctx = dict(ctx or {})
        if self.contexts is not None:           # None: not kept (a long-running mock manager)
            self.contexts.append((key_hash, ctx))
        if self.meta is None:
            return (await self.generate(body, key_hash)), None
        t_send = time.time()
        try:
            sample = await self.generate(body, key_hash)
        except TransportError as err:
            if self.meta is True:
                err.meta = self._synthetic(key_hash, ctx, t_send,
                                           error='timeout' if 'timeout' in str(err) else 'connection')
            raise
        except RequestError as err:
            if self.meta is True:
                err.meta = self._synthetic(key_hash, ctx, t_send, error=err.kind)
            raise
        if callable(self.meta):
            return sample, self.meta(body, key_hash, ctx)
        return sample, self._synthetic(key_hash, ctx, t_send, sample)


MOCK_REWARD_DEFAULTS = dict(MOCK_DEFAULTS, temperature=0.0)
STEP_TAG = '<extra_0>'


class MockReward(MockModel):
    """A stand-in for a reward backend (models INTERFACE J4): one score per step tag in the input
    (at least one), each a function of (nonce, key, position), returned as a JSON list."""
    kind = 'reward'

    def __init__(self, nonce='mock-reward', **kw):
        kw.setdefault('defaults', MOCK_REWARD_DEFAULTS)
        super().__init__(nonce, **kw)
        self.model_rev = 'mockreward-' + digest({'mock': nonce, 'defaults': self.defaults})[:16]

    def scores_for(self, key_hash, body):
        text = ''.join(m['content'] for m in body['messages'])
        n = max(1, text.count(STEP_TAG))
        return [int(digest({'nonce': self.nonce, 'key': key_hash, 'i': i})[:4], 16) / 65535.0
                for i in range(n)]

    async def generate(self, body, key_hash):
        sample = await super().generate(body, key_hash)
        if isinstance(sample, dict) and isinstance(sample.get('content'), str) \
                and self.respond is None:
            import json
            sample = dict(sample, content=json.dumps(self.scores_for(key_hash, body)),
                          completion_tokens=0)
        return sample
