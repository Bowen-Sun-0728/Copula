"""The packaged system: runs, groups and registered calls (reports/evalrt_sdk_20261009/INTERFACE.md).

An agent is ordinary code. Its model and tool calls are registered nodes; each one declares
`randomness` (deterministic or stochastic) and `cache` (True or False) when it is registered (U4).
A call made inside a run blocks until the system has read the cache, waited for the single flight
that is generating the sample, or generated it and written it (U3):

    stochastic + cache      (group, pool, j): j counts this run's earlier calls to the pool in the
                            group, in arrival order (U7). Runs of one group share; another group
                            never does (U5, U6)
    deterministic + cache   (pool): one global value, never copied (U4)
    stochastic, no cache    a private draw on every call
    deterministic, no cache executed on every call

There are no cases and no jobs (U2): the smallest unit is a run. A group is an independent pool, and
a request belongs to exactly one group: the run's, or the one a block of code switches to. `cache` is
always declared; `randomness` only matters, and is only required, for a cached node.

A declared static graph (U1) is executed by the system itself (`Session.execute`): it is built once,
runs as its own run, and nothing is intercepted -- no agent, client wrapper or run context is
involved.

The manager (`evalrt.manager`) is the backend: a run's ticket in a group is `<run>/<group>`, its
consumer the run id and its one pseudo-case the run id, so the manager's ledger counts samples per
(group, run, pool). Each attempt of a run (a resume included) is its own manager scope, so a
resumed run re-reads the samples it claimed, in claim order.
"""
import asyncio
import collections
import contextlib
import contextvars
import functools
import inspect
import json
import random
import re
import threading
import time
import uuid

from ..core.canon import canonical, digest
from ..core.key import GROUP_PREFIX, is_greedy
from ..core.request import (EXTRA_FIELDS, FORBIDDEN, SAMPLING_PARAMS, RequestSpecError,
                            normalize_message, normalize_request)
from ..errors import EvalrtError
from ..manager.server import AsyncClient as ManagerClient

RANDOMNESS = ('deterministic', 'stochastic')
LATE_WAIT = 300.0               # s that aclose waits for the tools of cancelled calls to finish
NODE_NAME = re.compile(r'^[A-Za-z0-9_.:-]{1,64}$')
GROUP_NAME = re.compile(r'^[A-Za-z0-9_.-]{1,64}$')
CASE = 'run'                    # a run's one pseudo-case in the manager's ledger
# Request fields an OpenAI-style call may carry besides the sampling parameters: refused when set
# to a value the system cannot honour, accepted when unset (None, False or empty).
# Tool calling and structured output (`tools`, `tool_choice`, `response_format`,
# `parallel_tool_calls`) are part of the request (sysval build).
REFUSED = tuple(sorted(set(FORBIDDEN) - {'model', 'n'})) + (
    'functions', 'function_call', 'logit_bias', 'modalities', 'audio', 'prediction',
    'reasoning_effort', 'web_search_options', 'stream_options')
IGNORED = ('user', 'timeout', 'extra_headers', 'extra_query', 'metadata', 'store', 'service_tier',
           'prompt_cache_key', 'safety_identifier', 'idempotency_key')
RAW_HEADER = 'X-Stainless-Raw-Response'      # openai's with_raw_response marker

_run = contextvars.ContextVar('evalrt_sdk_run', default=None)
_group = contextvars.ContextVar('evalrt_sdk_group', default=None)


class SDKError(EvalrtError):
    """A refused registration or call."""


def _err(kind, detail):
    return SDKError(kind, detail)


class Group:
    """An independent pool of stochastic samples (U2). Deterministic cached nodes ignore it (U4)."""
    __slots__ = ('id',)

    def __init__(self, gid):
        self.id = gid

    @property
    def partition(self):
        return GROUP_PREFIX + self.id

    def __eq__(self, other):
        return isinstance(other, Group) and other.id == self.id

    def __hash__(self):
        return hash(('group', self.id))

    def __repr__(self):
        return 'Group(%r)' % self.id


class LLMNode:
    __slots__ = ('name', 'backend', 'randomness', 'cache', 'pin', 'max_tokens')

    def __init__(self, name, backend, randomness, cache, pin, max_tokens):
        self.name, self.backend, self.randomness = name, backend, randomness
        self.cache, self.pin, self.max_tokens = cache, pin, max_tokens


class ToolNode:
    __slots__ = ('name', 'version', 'fn', 'randomness', 'cache', 'rev', 'takes_rng', 'max_hold')

    def __init__(self, name, version, fn, randomness, cache, rev, takes_rng, max_hold=None):
        self.name, self.version, self.fn = name, version, fn
        self.randomness, self.cache, self.rev, self.takes_rng = randomness, cache, rev, takes_rng
        self.max_hold = max_hold        # s an identical call waits on one execution (None: any)


RENEW_MIN = 0.05                    # s: the shortest renewal interval, whatever the manager's TTL


def _stored(reply):
    """The stored value a computed-value write reports when another requester wrote first."""
    return reply.get('value') if isinstance(reply, dict) else None


class _ToolFailed(Exception):
    """A deterministic tool raised: the same arguments raise again, so waiters get it too."""

    def __init__(self, err):
        super().__init__(err)
        self.err = err


def _check_fields(randomness, cache):
    """`cache` is always declared; `randomness` matters only for a cached node (an uncached node
    executes on every call whatever it declares, so stochastic, deterministic or nothing is
    accepted and changes nothing)."""
    if cache not in (True, False):
        raise _err('invalid_node', 'cache must be True or False')
    if cache and randomness not in RANDOMNESS:
        raise _err('invalid_node', 'a cached node declares randomness: one of %s' % (RANDOMNESS,))
    if not cache and randomness not in (None,) + RANDOMNESS:
        raise _err('invalid_node', 'randomness is one of %s, or left out' % (RANDOMNESS,))


def _fn_source(fn):
    """The function's own source (its revision changes when its body does; helpers in other
    functions are covered by the version the user declares)."""
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return getattr(getattr(fn, '__code__', None), 'co_code', b'').hex()


def _is_openai(client):
    try:
        import openai
    except ImportError:
        return False
    return isinstance(client, (openai.OpenAI, openai.AsyncOpenAI))


def _openai_async():
    import openai
    return openai.AsyncOpenAI


def _unset(value):
    return value is None or value is False or (isinstance(value, (list, dict)) and not value)


class Namespace(dict):
    """A response with attribute access, as OpenAI client objects have (`r.choices[0].message`)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    @classmethod
    def of(cls, obj):
        if isinstance(obj, dict):
            return cls({k: cls.of(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.of(v) for v in obj]
        return obj

    def model_dump(self):
        return json.loads(json.dumps(self))


class _Refuse:
    """Stands in for a client entry point the system does not route (nothing bypasses it, N11)."""

    def __init__(self, what):
        self._what = what

    def __call__(self, *args, **kwargs):
        raise _err('unsupported', '%s is not routed through the system; use '
                   'chat.completions.create' % self._what)

    def __getattr__(self, name):
        if name.startswith('__'):           # introspection (functools.wraps, copy) sees nothing
            raise AttributeError(name)
        raise _err('unsupported', '%s.%s is not routed through the system' % (self._what, name))


# -- openai clients: what an intercepted chat-completions request returns -----------------------------

_INTERCEPTING = []                  # sessions routing every openai client of the process (intercept)
_ORIGINAL = {}                      # openai client class -> its own `request`


def _chat_body(options):
    """The JSON body of an openai chat-completions request (extra_body merged), or None for any
    other request."""
    if str(getattr(options, 'method', '')).lower() != 'post' or not str(
            getattr(options, 'url', '')).rstrip('/').endswith('chat/completions'):
        return None
    body = dict(getattr(options, 'json_data', None) or {})
    extra = getattr(options, 'extra_json', None)
    if extra:
        body.update(extra)
    return body


def _raw_mode(options):
    headers = getattr(options, 'headers', None)
    value = headers.get(RAW_HEADER) if isinstance(headers, dict) else None
    if value is not None and str(value).lower() == 'stream':
        raise _err('unsupported', 'with_streaming_response is not routed through the system; use '
                   'create(stream=True) or with_raw_response')
    return value is not None and str(value).lower() == 'true'


def _chunks(data, include_usage):
    """The completed response as stream chunks: per choice one delta with the whole message,
    then its finish reason; usage last when asked for (the sample is complete before it is
    returned, so a stream replays it)."""
    base = {'id': data['id'], 'object': 'chat.completion.chunk', 'created': data['created'],
            'model': data['model'], 'system_fingerprint': None, 'service_tier': None}
    out = []
    for ch in data['choices']:
        m = ch['message']
        delta = {'role': 'assistant', 'content': m['content']}
        if m.get('tool_calls'):
            delta['tool_calls'] = [dict(c, index=k) for k, c in enumerate(m['tool_calls'])]
        if m.get('reasoning_content'):
            delta['reasoning_content'] = m['reasoning_content']
        out.append(dict(base, choices=[{'index': ch['index'], 'delta': delta,
                                        'finish_reason': None, 'logprobs': None}]))
        out.append(dict(base, choices=[{'index': ch['index'], 'delta': {},
                                        'finish_reason': ch['finish_reason'], 'logprobs': None}]))
    if include_usage:
        out.append(dict(base, choices=[], usage=data['usage']))
    return out


class _HTTP:
    """What a response's `http_response` / a stream's `response` offers: no real HTTP happened."""
    status_code = 200
    reason_phrase = 'OK'

    def __init__(self):
        try:
            import httpx
            self.headers = httpx.Headers({})
        except ImportError:                 # pragma: no cover (openai depends on httpx)
            self.headers = {}

    def close(self):                        # openai's stream helpers close the response
        pass

    async def aclose(self):
        pass


class _RawResponse:
    """openai's `with_raw_response` result: `parse()` gives the response."""
    retries_taken = 0
    request_id = None
    status_code = 200

    def __init__(self, obj):
        self._obj = obj
        self.http_response = _HTTP()
        self.headers = self.http_response.headers

    def parse(self, *, to=None):
        return self._obj

    @property
    def text(self):
        return self._obj.model_dump_json()


class _Stream:
    """A sync openai stream over the chunks of a completed response."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.response = _HTTP()

    def __iter__(self):
        return self

    def __next__(self):
        if not self._chunks:
            raise StopIteration
        return self._chunks.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._chunks = []


class _AsyncStream(_Stream):
    """An async openai stream over the chunks of a completed response."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self):
        self._chunks = []

    async def close(self):          # openai's AsyncStream.close is a coroutine
        self._chunks = []


def _owner(model):
    """The intercepting session that registered `model` (the one whose run this is first), or
    None: the request is not a registered call and goes out unchanged."""
    run = _run.get()
    if run is not None and run.session in _INTERCEPTING and model in run.session.llm:
        return run.session
    owners = [s for s in _INTERCEPTING if model in s.llm]
    if len(owners) > 1:
        raise _err('ambiguous', 'several intercepting sessions registered %r; call it inside a '
                   'run of one of them' % (model,))
    return owners[0] if owners else None


def _passthrough(model):
    """An intercepted request for a model no session registered goes out unchanged; while a run
    is active it is counted and traced, so a framework that rewrites model names is seen."""
    run = _run.get()
    if run is not None and run.session in _INTERCEPTING:
        sessions = [(run.session, run)]
    else:
        sessions = [(s, s._ambient) for s in _INTERCEPTING if s._ambient is not None]
    for s, r in sessions:
        s.passthrough += 1
        s._row({'ev': 'passthrough', 'model': model, 'run': r.id})


def _install():
    """Patch openai's client classes once: every chat-completions request of every client --
    `create`, `with_raw_response`, `parse`, streams, and clients made by frameworks (litellm,
    DSPy, LangChain, ...) before or after this point -- passes through `_owner`."""
    if _ORIGINAL:
        return
    try:
        from openai import _base_client as base
    except ImportError:
        raise _err('unsupported', 'the openai package is not installed')
    sync_request, async_request = base.SyncAPIClient.request, base.AsyncAPIClient.request

    def request(client, cast_to, options, *args, **kwargs):
        body = _chat_body(options)
        session = None if body is None else _owner(body.get('model'))
        if session is None:
            if body is not None:
                _passthrough(body.get('model'))
            return sync_request(client, cast_to, options, *args, **kwargs)
        return session._serve_sync(cast_to, options, body, kwargs.get('stream', False))

    async def arequest(client, cast_to, options, *args, **kwargs):
        body = _chat_body(options)
        session = None if body is None else _owner(body.get('model'))
        if session is None:
            if body is not None:
                _passthrough(body.get('model'))
            return await async_request(client, cast_to, options, *args, **kwargs)
        return await session._serve_async(cast_to, options, body, kwargs.get('stream', False))
    _ORIGINAL[base.SyncAPIClient] = sync_request
    _ORIGINAL[base.AsyncAPIClient] = async_request
    base.SyncAPIClient.request = request
    base.AsyncAPIClient.request = arequest


def _uninstall():
    for cls, original in _ORIGINAL.items():
        cls.request = original
    _ORIGINAL.clear()


class Run:
    """One execution under the system: the consumer of every call it makes."""

    def __init__(self, session, rid, group, label=None):
        self.session = session
        self.id = rid
        self.group = group
        self.label = label
        self.attempt = 'a-' + uuid.uuid4().hex[:12]    # this attempt's manager scope
        self.started_at = time.time()
        self.tickets = {}               # group id -> registered ticket
        self.generated = 0              # samples and results this attempt generated (the cap)
        self.closed = False
        self.salt = None                # vLLM prefix-cache isolation of this run (claims C6)
        self._generated_keys = set()    # store entries this attempt generated (counted once)
        self._lock = None               # made in the session's loop (a sync run is made elsewhere)

    def ticket_of(self, group):
        return '%s/%s' % (self.id, group.id)

    @property
    def private(self):
        return 'fresh/' + self.id

    async def ensure(self, group):
        """Register this run's ticket in `group` (once)."""
        if group.id in self.tickets:
            return self.tickets[group.id]
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if group.id not in self.tickets:
                s = self.session
                ticket = self.ticket_of(group)
                extra = {} if self.salt is None else {'cache_salt': self.salt}
                await s.manager.call('register_pool_ticket', ticket=ticket,
                                     partitions=[group.partition, self.private],
                                     consumer=self.id, model_rev=s.models['actor']['model_rev'],
                                     case_keys={CASE: self.id},
                                     models={n: m['model_rev'] for n, m in s.models.items()
                                             if n != 'actor'}, **extra)
                self.tickets[group.id] = ticket
        return self.tickets[group.id]

    def mine(self, writer):
        """Whether a stored result was written by this run: by any of its tickets (one per group;
        a deterministic result generated in one group is read again from another) and in any of
        its attempts. Run ids hold no '/', so the ticket prefix names the run exactly."""
        return isinstance(writer, str) and writer.startswith(self.id + '/')

    def count(self, key, source, ticket, generated_by):
        """The status of one result, counting each entry this attempt generated once (two
        concurrent identical calls join one flight: the second waited)."""
        mine = generated_by == ticket or self.mine(generated_by)
        if source == 'generated' or (source == 'waited' and mine):
            if key in self._generated_keys:
                return 'waited'
            self._generated_keys.add(key)
            self.generated += 1
            return 'generated'
        if source == 'waited':
            return 'waited'
        return 'own' if mine else 'hit'


MARK = '\u2063fc-%s\u2063'       # a stand-in's marker (claims C2, C3): invisible, unique per stand-in
MARK_PREFIX = '\u2063fc-'


def _with_mark(text):
    """`text` with a unique marker at the end of every line of at least 24 characters (claims
    C3): text derived from such a line (a parsed problem, a quoted solution) carries the marker
    into later requests, while short values (a label, a score, a boxed answer, a short reply) are
    left untouched and untracked, so code that compares or parses them behaves as in a real run."""
    mark = MARK % uuid.uuid4().hex[:12]
    lines = text.split('\n')
    return '\n'.join(line + mark if len(line.strip()) >= 24 else line for line in lines)


def _text_of(body):
    return '\n'.join(m.get('content') or '' for m in body['messages']
                     if isinstance(m.get('content'), str))


class ForecastRun:
    """A run that only forecasts (claims C3): every registered call is answered from the store when
    the store holds the sample a real run would read, and is otherwise predicted to generate and
    answered with a stand-in. Nothing is claimed, flown or written. `reference` {node: [outputs
    in call order]} supplies stand-ins (the same node's output at the same occurrence,
    cyclically), each with a unique marker; a later call whose messages carry a marker is derived
    from a predicted generation and is predicted to generate too. `history` {node: {'completion':
    mean tokens, 'prompt_per_char': tokens per character}} gives the token estimates."""

    forecast = True

    def __init__(self, session, group, reference=None, history=None, label=None):
        self.session = session
        self.id = 'forecast-' + uuid.uuid4().hex[:12]
        self.group = group
        self.label = label
        self.closed = False
        self.reference = {k: list(v) for k, v in (reference or {}).items()}
        self.history = dict(history or {})
        self.numbers = collections.Counter()        # (group id, pool) -> draws so far
        self.occurrence = collections.Counter()     # node -> calls so far (stand-in choice)
        self.predicted = {}                         # deterministic address -> its stand-in
        self.rows = []

    def stand_in(self, node):
        ref = self.reference.get(node) or []
        k = self.occurrence[node]
        self.occurrence[node] += 1
        return _with_mark(ref[k % len(ref)] if ref else '')

    @staticmethod
    def tainted(body):
        return MARK_PREFIX in _text_of(body)

    def estimate(self, node, body):
        h = self.history.get(node) or {}
        chars = len(_text_of(body))
        c = h.get('completion')
        p = h.get('prompt_per_char')
        return {'completion': float(c) if c is not None else float(body.get('max_tokens') or 0),
                'prompt': chars * float(p) if p is not None else chars / 4.0,
                'no_history': c is None}

    def summary(self):
        out = {'calls': 0, 'completion_tokens': 0.0, 'prompt_tokens': 0.0, 'hits': 0,
               'no_history': 0, 'by_node': {}}
        for r in self.rows:
            n = out['by_node'].setdefault(r['node'], {'generate': 0, 'hit': 0,
                                                      'completion_tokens': 0.0})
            if r['status'] in ('hit', 'waited'):
                out['hits'] += 1
                n['hit'] += 1
                continue
            out['calls'] += 1
            out['completion_tokens'] += r['completion']
            out['prompt_tokens'] += r['prompt']
            out['no_history'] += r['no_history']
            n['generate'] += 1
            n['completion_tokens'] += r['completion']
        return out


class Session:
    """A connection to one manager, with its registry of nodes, groups and runs."""

    def __init__(self, manager, models, trace=None, cap=None):
        self.manager = manager
        self.models = models            # name -> {model_rev, type, defaults}
        self.llm = {}
        self.tools = {}
        self.cap = cap                  # per run attempt: generations allowed, or None
        self._trace = open(trace, 'a') if trace else None
        self._programs = {}             # (graph revision, config) -> built static program (U1)
        self._memo = {}                 # tool identity -> future (in-process dedup)
        self._releases = set()          # lease releases still being sent
        self._renewals = {}             # lease token -> the task renewing it while its tool runs
        self.lease_caps = None          # the manager's lease features (ping), None: an older one
        self._late = set()              # tools and writes of cancelled calls still running
        self._loop = None
        self._loop_thread = None        # the thread the session's loop runs in
        self._thread = None             # set when the session's loop runs in a background thread
        self._ambient = None            # the run of calls made where no run context is visible
        self._open = {}                 # run id -> its open attempt (the HTTP endpoint names runs)
        self.passthrough = 0            # intercepted chat requests for unregistered models
        self._http = None               # the HTTP endpoint's server, once started

    # -- connection -------------------------------------------------------------------------------

    @classmethod
    async def connect(cls, socket, trace=None, cap=None):
        manager = await ManagerClient.connect(socket)
        ping = await manager.call('ping')
        models = ping.get('models') or {'actor': {'model_rev': ping['model_rev'],
                                                  'type': 'generate', 'defaults': None}}
        session = cls(manager, models, trace=trace, cap=cap)
        session.lease_caps = ping.get('leases')
        session._loop = asyncio.get_running_loop()
        session._loop_thread = threading.current_thread()
        return session

    @classmethod
    def connect_sync(cls, socket, trace=None, cap=None):
        """A session for synchronous agent code: its event loop runs in a background thread."""
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, name='evalrt-sdk', daemon=True)
        thread.start()
        try:
            session = asyncio.run_coroutine_threadsafe(cls.connect(socket, trace, cap),
                                                       loop).result()
        except BaseException:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(5)
            loop.close()
            raise
        session._thread = thread
        return session

    def _submit(self, coro):
        """Run `coro` on the session's loop from synchronous code: from any thread but the loop's
        own (where waiting would block the loop that must serve the call)."""
        if self._loop is None or threading.current_thread() is self._loop_thread:
            coro.close()
            raise _err('not_sync', 'a synchronous call on the thread of the session\'s event loop '
                       'would block it: use the async calls, call from another thread, or open '
                       'the session with connect_sync()')
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _bridge(self, coro):
        """Run `coro` on the session's loop, awaited from whatever loop the caller is on."""
        try:
            here = asyncio.get_running_loop()
        except RuntimeError:
            here = None
        if here is self._loop:
            return await coro
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, self._loop))

    async def aclose(self):
        if self._http is not None:
            self._http.close()
        if self._late:                  # let cancelled calls' tools finish and write (bounded)
            await asyncio.wait(set(self._late), timeout=LATE_WAIT)
        if self._releases:
            await asyncio.gather(*self._releases, return_exceptions=True)
        renewals = list(self._renewals.values())     # tools still running past LATE_WAIT
        for task in renewals:
            task.cancel()
        if renewals:
            await asyncio.gather(*renewals, return_exceptions=True)
        await self.manager.close()
        if self._trace is not None:
            self._trace.close()

    def close(self):
        if self._thread is None:
            raise _err('not_sync', 'a session opened with connect() is closed with aclose()')
        self._submit(self.aclose())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)
        if not self._loop.is_running():
            self._loop.close()

    # -- registration (U4) ------------------------------------------------------------------------

    def register_llm(self, name, *, cache, randomness=None, backend='actor', pin=None,
                     max_tokens=1024):
        """An LLM node: calls whose `model` is `name` go to `backend` with `pin` fixed. `cache` is
        required, and so is `randomness` when the node is cached: a deterministic cached node must
        be greedy (temperature 0 or top_k 1), and a reward backend's cached nodes are deterministic.
        An uncached node draws privately on every call, whatever `randomness` says."""
        _check_fields(randomness, cache)
        self._check_name(name)
        model = self.models.get(backend)
        if model is None:
            raise _err('unknown_model', 'the stack has %s, not %r' % (sorted(self.models), backend))
        pin = dict(pin or {})
        unknown = sorted(set(pin) - set(SAMPLING_PARAMS))
        if unknown:
            raise _err('invalid_node', 'pin names %s, which are not sampling parameters' % unknown)
        if type(max_tokens) is not int or max_tokens < 1:
            raise _err('invalid_node', 'max_tokens must be a positive int')
        if model.get('defaults') is not None:
            try:
                probe = normalize_request(dict({'messages': [{'role': 'user', 'content': ''}],
                                                'max_tokens': 1}, **pin), model['defaults'])
            except ValueError as err:
                raise _err('invalid_node', str(err))
            if cache and randomness == 'deterministic' and not is_greedy(probe):
                raise _err('invalid_node', 'deterministic node %r must pin greedy decoding '
                           '(temperature 0 or top_k 1)' % name)
        if cache and model.get('type') == 'reward' and randomness != 'deterministic':
            raise _err('invalid_node', 'a reward backend scores deterministically')
        self.llm[name] = LLMNode(name, backend, randomness, cache, pin, max_tokens)
        return self.llm[name]

    def _check_name(self, name):
        if not isinstance(name, str) or not NODE_NAME.match(name):
            raise _err('invalid_node', 'node name must match %s' % NODE_NAME.pattern)
        if name in self.llm or name in self.tools:
            raise _err('invalid_node', 'node %r is already registered' % name)

    def tool(self, name=None, version=1, *, cache, randomness=None, max_hold=None):
        """Decorator: a tool node. `cache` is required, and so is `randomness` when the tool is
        cached: a stochastic cached tool takes an `rng` argument (seeded from its address), a
        deterministic cached one does not. Identical concurrent calls of a cached tool execute
        once, in any process: the others wait on the manager's lease and read the result. The
        executing process renews its lease while the tool runs, so a holder that dies or freezes
        holds up the others for at most the manager's lease TTL. `max_hold` (seconds, cached tools
        only) bounds how long a call waits on another's execution of the same call, in this
        process or another (the shortest bound declared by the executor or a waiter counts): past
        it, one waiter executes the call itself. Whoever executes, every caller gets the value
        stored first. An uncached tool executes on every call whatever it declares (given a fresh
        `rng` if it takes one). Arguments and results are JSON. The revision covers the function's
        own source and `version`: bump the version when a helper it calls changes; `max_hold` is
        not part of it."""
        _check_fields(randomness, cache)
        if max_hold is not None:
            if not (type(max_hold) in (int, float) and 0 < max_hold < float('inf')):
                raise _err('invalid_node', 'max_hold must be a number of seconds > 0, or None')
            if not cache:
                raise _err('invalid_node', 'max_hold bounds waits on a cached tool; this tool '
                           'is not cached')
            if not (self.lease_caps or {}).get('max_hold'):
                raise _err('invalid_node', 'max_hold needs a manager with lease expiry; this '
                           'manager would ignore it')

        def register(fn):
            tname = name or fn.__name__
            self._check_name(tname)
            takes_rng = 'rng' in inspect.signature(fn).parameters
            if cache and randomness == 'stochastic' and not takes_rng:
                raise _err('invalid_node', 'stochastic cached tool %r must take `rng`' % tname)
            if cache and randomness == 'deterministic' and takes_rng:
                raise _err('invalid_node', 'deterministic cached tool %r takes no `rng`' % tname)
            rev = digest({'name': tname, 'version': version, 'source': _fn_source(fn),
                          'qualname': getattr(fn, '__qualname__', '')})
            node = ToolNode(tname, version, fn, randomness, cache, rev, takes_rng, max_hold)
            self.tools[tname] = node
            if inspect.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def call(*args, **kwargs):
                    run, group = self._current()
                    return await self._bridge(self._tool(run, group, node, args, kwargs,
                                                         self._task()))
            else:
                @functools.wraps(fn)
                def call(*args, **kwargs):
                    run, group = self._current()
                    return self._submit(self._tool(run, group, node, args, kwargs, self._task()))

                async def aio(*args, **kwargs):
                    run, group = self._current()
                    return await self._bridge(self._tool(run, group, node, args, kwargs,
                                                         self._task()))
                call.aio = aio
            call.node = node
            return call
        return register

    # -- groups and runs (U2, U5, U6) ---------------------------------------------------------------

    def group(self, name=None):
        """A named group (the same name is the same pool), or a fresh one."""
        gid = name if name is not None else 'g-' + uuid.uuid4().hex[:16]
        if not isinstance(gid, str) or not GROUP_NAME.match(gid):
            raise _err('invalid_group', 'group name must match %s' % GROUP_NAME.pattern)
        return Group(gid)

    @contextlib.asynccontextmanager
    async def run(self, group, run_id=None, label=None, ambient=False, cache_salt=None):
        """One run in `group` (its default pool). `run_id` resumes an earlier run: it re-reads the
        samples it claimed, in claim order. A run's calls go to its own group, whatever block the
        run was opened in. `ambient=True` also gives it the calls made where no run context is
        visible -- threads a framework starts (a thread pool does not inherit context variables);
        one ambient run at a time per session. `cache_salt` isolates the run's requests in the
        server's prefix cache (vLLM `cache_salt`; it never enters a key); a resumed run passes the
        salt it started with."""
        r = self._new_run(group, run_id, label, cache_salt)
        self._set_ambient(r, ambient)
        try:
            await self._bridge(r.ensure(group))
        except BaseException:
            self._clear_ambient(r)
            raise
        self._open[r.id] = r
        self._row({'ev': 'run_start', 'run': r.id, 'attempt': r.attempt, 'group': group.id,
                   'label': label})
        tokens = (_run.set(r), _group.set(None))
        try:
            yield r
        finally:
            _group.reset(tokens[1])
            _run.reset(tokens[0])
            await self._bridge(self._end(r))

    @contextlib.contextmanager
    def run_sync(self, group, run_id=None, label=None, ambient=False, cache_salt=None):
        r = self._new_run(group, run_id, label, cache_salt)
        self._set_ambient(r, ambient)
        try:
            self._submit(r.ensure(group))
        except BaseException:
            self._clear_ambient(r)
            raise
        self._open[r.id] = r
        self._row({'ev': 'run_start', 'run': r.id, 'attempt': r.attempt, 'group': group.id,
                   'label': label})
        tokens = (_run.set(r), _group.set(None))
        try:
            yield r
        finally:
            _group.reset(tokens[1])
            _run.reset(tokens[0])
            self._submit(self._end(r))

    @contextlib.asynccontextmanager
    async def forecast_run(self, group, reference=None, history=None, label=None):
        """A forecast of an agent run in `group` (claims C3): the agent code runs as it would in a
        real run, but every registered LLM call reads the store without claiming (manager `peek`)
        at the number the real run would read (1 + its earlier calls to that pool in that group);
        a miss, a call whose messages carry a stand-in's marker, and every call of an uncached
        node is predicted to generate and returns a stand-in (ForecastRun); a deterministic
        request predicted earlier in the forecast is joined, as a real run joins its flight. Tools
        run as plain functions and write nothing. Yields the ForecastRun; `summary()` is the
        prediction. Nothing is written to the store and no model is called."""
        if not isinstance(group, Group):
            raise _err('invalid_group', 'a forecast needs a Group (session.group())')
        r = ForecastRun(self, group, reference, history, label)
        tokens = (_run.set(r), _group.set(None))
        try:
            yield r
        finally:
            _group.reset(tokens[1])
            _run.reset(tokens[0])
            r.closed = True

    def _set_ambient(self, r, ambient):
        if not ambient:
            return
        if self._ambient is not None:
            raise _err('ambient', 'run %s is already the ambient run' % self._ambient.id)
        self._ambient = r

    def _clear_ambient(self, r):
        if self._ambient is r:
            self._ambient = None

    async def _end(self, r):
        r.closed = True
        self._clear_ambient(r)
        if self._open.get(r.id) is r:
            del self._open[r.id]
        for ticket in list(r.tickets.values()):
            await self.manager.call('end_attempt', ticket=ticket, attempt=r.attempt)
        self._row({'ev': 'run_end', 'run': r.id, 'attempt': r.attempt, 'generated': r.generated,
                   'wall_s': round(time.time() - r.started_at, 3)})

    def _new_run(self, group, run_id, label, cache_salt=None):
        if not isinstance(group, Group):
            raise _err('invalid_group', 'a run needs a Group (session.group())')
        rid = run_id or 'r-' + uuid.uuid4().hex[:16]
        if not isinstance(rid, str) or not GROUP_NAME.match(rid):
            raise _err('invalid_run', 'run id must match %s' % GROUP_NAME.pattern)
        r = Run(self, rid, group, label)
        r.salt = cache_salt
        return r

    @contextlib.contextmanager
    def in_group(self, group):
        """The calls made in this block go to `group` (a request is in exactly one group, U5)."""
        if not isinstance(group, Group):
            raise _err('invalid_group', 'in_group needs a Group')
        token = _group.set(group)
        try:
            yield group
        finally:
            _group.reset(token)

    @staticmethod
    def context():
        """The caller's run context, for code that starts threads inside a run:
        `executor.submit(session.context().run, fn, ...)` (threads do not inherit it)."""
        return contextvars.copy_context()

    def _current(self):
        r = _run.get()
        if (r is None or r.session is not self) and self._ambient is not None:
            r = self._ambient                       # a thread without the run's context
        if r is None or r.session is not self:
            raise _err('no_run', 'a registered call was made outside a run of this session')
        if r.closed:
            raise _err('no_run', 'run %s has ended' % r.id)
        return r, _group.get() or r.group

    # -- LLM calls (U3) ---------------------------------------------------------------------------

    async def chat(self, model, messages, **params):
        """An OpenAI-style chat completion through the system: `model` is a registered node."""
        run, group = self._current()
        return Namespace.of(await self._bridge(self._llm(run, group, model, messages, params,
                                                         self._task())))

    def chat_sync(self, model, messages, **params):
        run, group = self._current()
        return Namespace.of(self._submit(self._llm(run, group, model, messages, params,
                                                   self._task())))

    def wrap(self, client, mode=None):
        """Route a client's chat completions through the system. An `openai` client (sync or
        async, Azure included) is routed at its request method, so `create`, `with_raw_response`,
        `parse` (structured output) and `create(stream=True)` all go through the system, while any
        other endpoint of it (legacy completions, responses, embeddings) refuses: the wrapped
        client is the system's (N11). Any other OpenAI-style object has its
        `chat.completions.create` replaced; `mode` ('async' or 'sync') says how it is called (by
        default async when `create` is a coroutine or a class is named Async*), and its `parse`,
        `stream`, `with_raw_response`, legacy completions and responses refuse."""
        if _is_openai(client):
            return self._wrap_openai(client)
        completions = client.chat.completions
        original = completions.create
        if mode is None:
            wrapped = getattr(original, '__wrapped__', None)
            mode = 'async' if (inspect.iscoroutinefunction(original)
                               or inspect.iscoroutinefunction(wrapped)
                               or type(client).__name__.startswith('Async')
                               or type(completions).__name__.startswith('Async')) else 'sync'
        if mode not in ('async', 'sync'):
            raise _err('invalid_call', "mode is 'async' or 'sync'")
        if mode == 'async':
            async def create(*args, model, messages, **params):
                if args:
                    raise _err('invalid_call', 'chat.completions.create takes keyword arguments')
                return await self.chat(model, messages, **params)
        else:
            def create(*args, model, messages, **params):
                if args:
                    raise _err('invalid_call', 'chat.completions.create takes keyword arguments')
                return self.chat_sync(model, messages, **params)
        create.original = original
        completions.create = create
        for attr in ('parse', 'stream', 'with_raw_response', 'with_streaming_response'):
            if hasattr(completions, attr):
                setattr(completions, attr, _Refuse('chat.completions.' + attr))
        for attr in ('completions', 'responses'):
            if getattr(client, attr, None) is not None:
                try:
                    setattr(client, attr, _Refuse(attr))
                except AttributeError:
                    pass
        return client

    def _wrap_openai(self, client):
        """The client becomes an instance of a subclass whose `request` routes through the
        system: every resource calls `self.request` at call time (resources made before the wrap
        are routed too), and `copy()` / `with_options()` build instances of the same subclass."""
        session = self
        base = type(client)
        if getattr(base, '_evalrt_session', None) is not None:
            base = base.__mro__[1]                      # wrapped before: re-wrap the original

        def refuse(options):
            raise _err('unsupported', '%s %s is not routed through the system; use chat '
                       'completions' % (options.method.upper(), options.url))
        if inspect.iscoroutinefunction(base.request) or isinstance(client, _openai_async()):
            async def request(self_, cast_to, options, *args, **kwargs):
                body = _chat_body(options)
                if body is None:
                    refuse(options)
                return await session._serve_async(cast_to, options, body,
                                                  kwargs.get('stream', False))
        else:
            def request(self_, cast_to, options, *args, **kwargs):
                body = _chat_body(options)
                if body is None:
                    refuse(options)
                return session._serve_sync(cast_to, options, body, kwargs.get('stream', False))
        wrapped = type(base.__name__, (base,), {'request': request, '_evalrt_session': session,
                                                '__module__': base.__module__})
        client.__class__ = wrapped
        return client

    @contextlib.contextmanager
    def intercept(self):
        """Route the registered calls of every `openai` client in this process, for the duration
        of the block: a chat-completions request whose model is a node registered here goes
        through the system (sync or async, raw, parsed or streamed); every other request goes out
        unchanged. This is how frameworks that build their own clients (litellm, DSPy, LangChain,
        LlamaIndex, ...) are run under the system without changing their code."""
        _install()
        _INTERCEPTING.append(self)
        try:
            yield self
        finally:
            _INTERCEPTING.remove(self)
            if not _INTERCEPTING:
                _uninstall()

    def _intercepted(self, body):
        body = dict(body)
        model, messages = body.pop('model', None), body.pop('messages', None)
        stream = bool(body.pop('stream', False))
        stream_options = body.pop('stream_options', None)
        return model, messages, body, stream, stream_options

    async def _serve_async(self, cast_to, options, body, stream):
        _raw_mode(options)                  # refuses with_streaming_response before drawing
        model, messages, params, streamed, stream_options = self._intercepted(body)
        run, group = self._current()
        result = await self._bridge(self._llm(run, group, model, messages, params, self._task()))
        return self._shape(cast_to, options, result, stream or streamed, stream_options, True)

    def _serve_sync(self, cast_to, options, body, stream):
        _raw_mode(options)
        model, messages, params, streamed, stream_options = self._intercepted(body)
        run, group = self._current()
        result = self._submit(self._llm(run, group, model, messages, params, self._task()))
        return self._shape(cast_to, options, result, stream or streamed, stream_options, False)

    @staticmethod
    def _shape(cast_to, options, result, stream, stream_options, is_async):
        """The response as the openai client would have returned it: a ChatCompletion (parsed
        by the request's parser for `parse`), a raw response, or a stream of chunks."""
        from openai.types.chat import ChatCompletion, ChatCompletionChunk
        data = {k: v for k, v in result.items() if k != 'evalrt'}
        raw = _raw_mode(options)
        if stream:
            include = isinstance(stream_options, dict) and bool(stream_options.get('include_usage'))
            chunks = [ChatCompletionChunk.model_validate(c) for c in _chunks(data, include)]
            out = _AsyncStream(chunks) if is_async else _Stream(chunks)
            return _RawResponse(out) if raw else out
        model = cast_to if isinstance(cast_to, type) and hasattr(cast_to, 'model_validate') \
            else ChatCompletion
        obj = model.model_validate(data)
        parser = getattr(options, 'post_parser', None)
        if callable(parser):
            obj = parser(obj)
        return _RawResponse(obj) if raw else obj

    # -- the HTTP endpoint ------------------------------------------------------------------------

    async def serve_http(self, host='127.0.0.1', port=0):
        """Serve the OpenAI chat-completions API over HTTP, for clients that send their own HTTP
        (any library or language): returns the base URL. `<base>/v1` routes to the ambient run;
        `<base>/runs/<run>/v1` to an open run of this session and `<base>/runs/<run>/groups/<g>/v1`
        to its calls in group g (a group switch). The model is a registered node; the semantics are
        those of `chat` (JSON replies, or server-sent events for `stream`)."""
        return await self._bridge(self._serve_http(host, port))

    def serve_http_sync(self, host='127.0.0.1', port=0):
        return self._submit(self._serve_http(host, port))

    def http_url(self, run=None, group=None):
        """The base URL (for an OpenAI client: append nothing, it ends in /v1) of a run's calls,
        optionally switched to `group`; without a run, the ambient run's."""
        if self._http is None:
            raise _err('no_http', 'call serve_http() first')
        host, port = self._http.sockets[0].getsockname()[:2]
        path = ''
        if run is not None:
            path += '/runs/%s' % (run.id if isinstance(run, Run) else run)
            if group is not None:
                path += '/groups/%s' % (group.id if isinstance(group, Group) else group)
        return 'http://%s:%d%s/v1' % (host, port, path)

    async def _serve_http(self, host, port):
        if self._http is None:
            self._http = await asyncio.start_server(self._http_conn, host, port)
        return self.http_url()

    async def _http_conn(self, reader, writer):
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                parts = line.decode('latin-1').split()
                if len(parts) != 3:
                    break
                method, path, version = parts
                headers = {}
                while True:
                    h = await reader.readline()
                    if h in (b'\r\n', b'\n', b''):
                        break
                    k, _, v = h.decode('latin-1').partition(':')
                    headers[k.strip().lower()] = v.strip()
                if headers.get('expect', '').lower() == '100-continue':
                    writer.write(b'HTTP/1.1 100 Continue\r\n\r\n')
                    await writer.drain()
                if headers.get('transfer-encoding', '').lower() == 'chunked':
                    body = b''
                    while True:
                        size = int((await reader.readline()).split(b';')[0], 16)
                        if size == 0:
                            while (await reader.readline()) not in (b'\r\n', b'\n', b''):
                                pass                            # trailers
                            break
                        body += await reader.readexactly(size)
                        await reader.readline()
                else:
                    body = await reader.readexactly(int(headers.get('content-length') or 0))
                status, kind, data = await self._http_answer(method, path, body)
                keep = version == 'HTTP/1.1' and headers.get('connection', '').lower() != 'close'
                # An error is final: the system has already retried what may be retried, and a
                # client retry could select samples by their outcome (openai honours this header).
                extra = '' if status == 200 else 'x-should-retry: false\r\n'
                writer.write(('HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %d\r\n'
                              '%sConnection: %s\r\n\r\n' % (
                                  status, 'OK' if status == 200 else 'Error', kind, len(data),
                                  extra, 'keep-alive' if keep else 'close')).encode('latin-1')
                             + data)
                await writer.drain()
                if not keep:
                    break
        except (asyncio.IncompleteReadError, ConnectionError, ValueError):
            pass
        finally:
            writer.close()

    async def _http_answer(self, method, path, body):
        """(status, content type, bytes) of one HTTP request."""
        m = re.match(r'^(?:/runs/([^/]+)(?:/groups/([^/]+))?)?/v1/(chat/completions|models)$',
                     path.split('?', 1)[0])
        if m is None:
            return self._http_error(404, 'not_found', 'no such endpoint: %s' % path)
        if m.group(3) == 'models':
            return 200, 'application/json', json.dumps({'object': 'list', 'data': [
                {'id': name, 'object': 'model', 'owned_by': 'evalrt'}
                for name in sorted(self.llm)]}).encode()
        if method != 'POST':
            return self._http_error(405, 'method', 'POST a chat completion')
        try:
            req = json.loads(body.decode('utf-8'))
            run = self._open.get(m.group(1)) if m.group(1) else self._ambient
            if run is None or run.closed:
                raise _err('no_run', 'no open run %s' % (m.group(1) or '(ambient)'))
            group = Group(m.group(2)) if m.group(2) else run.group
            if m.group(2) and not GROUP_NAME.match(m.group(2)):
                raise _err('invalid_group', 'group name must match %s' % GROUP_NAME.pattern)
            model, messages, params, stream, stream_options = self._intercepted(req)
            result = await self._llm(run, group, model, messages, params, 'http')
        except SDKError as err:
            return self._http_error(400, err.kind, err.detail)
        except EvalrtError as err:                  # the manager's: model or store failures
            return self._http_error(502, err.kind, err.detail)
        except (ValueError, TypeError, AttributeError) as err:
            return self._http_error(400, 'invalid_call', str(err))
        except Exception as err:                    # noqa: BLE001 -- always answer
            return self._http_error(500, 'internal', repr(err))
        data = {k: v for k, v in result.items() if k != 'evalrt'}
        if stream:
            include = isinstance(stream_options, dict) and bool(stream_options.get('include_usage'))
            events = ''.join('data: %s\n\n' % json.dumps(c) for c in _chunks(data, include))
            return 200, 'text/event-stream', (events + 'data: [DONE]\n\n').encode()
        return 200, 'application/json', json.dumps(data).encode()

    @staticmethod
    def _http_error(status, kind, detail):
        return status, 'application/json', json.dumps({'error': {
            'message': str(detail), 'type': kind, 'code': kind, 'param': None}}).encode()

    def openai_client(self):
        """An `openai.AsyncOpenAI` client (the openai package must be installed) whose calls go
        through the system only."""
        try:
            import openai
        except ImportError:
            raise _err('unsupported', 'the openai package is not installed; wrap your own client')
        return self.wrap(openai.AsyncOpenAI(api_key='evalrt', base_url='http://evalrt.invalid/v1'),
                         mode='async')

    def _body(self, node, messages, params):
        params = dict(params)
        extra = params.pop('extra_body', None) or {}
        for name in IGNORED:
            params.pop(name, None)
        if not isinstance(extra, dict):
            raise _err('invalid_call', 'extra_body is an object')
        params.update(extra)
        n = params.pop('n', None)
        n = 1 if n is None else n
        if type(n) is not int or n < 1:
            raise _err('invalid_call', 'n must be a positive int')
        refused = sorted(k for k in params if k in REFUSED and not _unset(params[k]))
        if refused:
            raise _err('invalid_call', 'the system does not take %s' % ', '.join(refused))
        for k in [k for k in params if k in REFUSED or params[k] is None
                  or (k in EXTRA_FIELDS and isinstance(params[k], (list, dict)) and not params[k])]:
            del params[k]
        max_tokens = params.pop('max_tokens', None)
        mct = params.pop('max_completion_tokens', None)
        max_tokens = max_tokens if max_tokens is not None else mct
        max_tokens = node.max_tokens if max_tokens is None else max_tokens
        if isinstance(params.get('stop'), str):
            params['stop'] = [params['stop']]
        unknown = sorted(set(params) - set(SAMPLING_PARAMS) - set(EXTRA_FIELDS))
        if unknown:
            raise _err('invalid_call', 'unknown request fields %s' % ', '.join(unknown))
        clash = sorted(k for k, v in params.items() if k in node.pin and v != node.pin[k])
        if clash:
            raise _err('invalid_call', 'node %s pins %s' % (node.name,
                                                            {k: node.pin[k] for k in clash}))
        if not isinstance(messages, (list, tuple)) or not messages:
            raise _err('invalid_call', 'messages is a non-empty list')
        clean = [self._message(m) for m in messages]
        body = dict({'messages': clean, 'max_tokens': max_tokens}, **dict(node.pin, **params))
        model = self.models[node.backend]
        if model.get('defaults') is not None:
            try:
                normalize_request(body, model['defaults'])
            except ValueError as err:
                raise _err('invalid_call', str(err))
        return body, n

    @staticmethod
    def _message(m):
        """One message as the system keys it. An OpenAI message object (a response's message
        appended to the history) is accepted: its response-only fields (refusal, annotations,
        audio, function_call, parsed) and its unset fields are dropped. Plain turns are
        {role, content} and may carry a `name`; an assistant turn may carry tool calls, a tool
        result is {role: tool, content, tool_call_id}; anything else (content parts, unknown
        fields) is refused."""
        if not isinstance(m, dict) and hasattr(m, 'model_dump'):
            m = m.model_dump()
        if not isinstance(m, dict):
            raise _err('invalid_call', 'a message is an object')
        m = {k: v for k, v in m.items()
             if v is not None and k not in ('refusal', 'annotations', 'audio', 'function_call',
                                            'parsed')}
        if isinstance(m.get('tool_calls'), (list, tuple)):
            m['tool_calls'] = [c.model_dump() if not isinstance(c, dict) and hasattr(
                c, 'model_dump') else c for c in m['tool_calls']]
        try:
            return normalize_message(m)
        except RequestSpecError as err:
            raise _err('invalid_call', 'messages: %s' % err)

    async def _llm(self, run, group, model, messages, params, task):
        node = self.llm.get(model)
        if node is None:
            raise _err('unregistered', 'model %r is not a registered node' % (model,))
        body, n = self._body(node, messages, params)
        if getattr(run, 'forecast', False):
            return await self._forecast_llm(run, group, node, body, n)
        ticket = await run.ensure(group)
        if node.cache:
            partition, randomness = group.partition, node.randomness
        else:                           # a private draw on every call
            partition, randomness = run.private, 'stochastic'
        self._check_cap(run)
        if run.closed:                      # a call that raced its run's end (thread, HTTP)
            raise _err('no_run', 'run %s has ended' % run.id)
        backend = None if node.backend == 'actor' else node.backend
        t0 = time.time()
        results = await asyncio.gather(*[
            self.manager.draw(ticket, node.name, body, CASE, partition=partition, model=backend,
                              randomness=randomness, attempt=run.attempt) for _ in range(n)])
        t1 = time.time()
        choices, calls, errors, pt, ct = [], [], [], 0, 0
        for i, res in enumerate(results):
            status = run.count(res['key_hash'], res['source'], ticket, res.get('generated_by'))
            s = res['sample']
            calls.append({'key_hash': res['key_hash'], 'index': res.get('index'),
                          'status': status})
            row = {'ev': 'call', 'run': run.id, 'group': group.id, 'node': node.name,
                   'kind': 'llm', 'backend': node.backend, 'randomness': node.randomness,
                   'cache': node.cache, 'pool': res.get('pool'), 'index': res.get('index'),
                   'key_hash': res['key_hash'], 'status': status, 't0': t0, 't1': t1,
                   'task': task}
            if 'error' in s:
                errors.append('%s: %s' % (s['error'], s.get('detail', '')))
                self._row(dict(row, error=s['error']))
                continue
            pt, ct = s['prompt_tokens'], ct + s['completion_tokens']
            message = {'role': 'assistant', 'content': s['content'], 'refusal': None,
                       'tool_calls': None, 'function_call': None, 'audio': None}
            if s.get('tool_calls'):
                message['tool_calls'] = s['tool_calls']
                if not s['content']:
                    message['content'] = None
            if s.get('reasoning'):
                message['reasoning_content'] = s['reasoning']
            choices.append({'index': i, 'finish_reason': s['finish_reason'], 'logprobs': None,
                            'message': message})
            self._row(dict(row, prompt_tokens=s['prompt_tokens'],
                           completion_tokens=s['completion_tokens']))
        if errors:
            raise _err('model_error', '; '.join(errors))
        return ({
            'id': 'evalrt-' + results[0]['key_hash'][:24], 'object': 'chat.completion',
            'created': int(t1), 'model': node.name, 'system_fingerprint': None,
            'service_tier': None, 'choices': choices,
            'usage': {'prompt_tokens': pt, 'completion_tokens': ct, 'total_tokens': pt + ct,
                      'prompt_tokens_details': None, 'completion_tokens_details': None},
            'evalrt': {'run': run.id, 'group': group.id, 'calls': calls}})

    async def _forecast_llm(self, run, group, node, body, n):
        backend = None if node.backend == 'actor' else node.backend
        choices = []
        pt = ct = 0
        tainted = run.tainted(body)
        stochastic = node.cache and node.randomness == 'stochastic'
        if stochastic:
            pool = (await self.manager.call('peek', body=body, partition=group.partition,
                                            model=backend, randomness='stochastic'))['pool']
        for i in range(n):
            status, sample, address = 'miss', None, None
            if not node.cache:
                status = 'uncached'
            elif node.randomness == 'deterministic':
                got = await self.manager.call('peek', body=body, partition=group.partition,
                                              model=backend, randomness='deterministic')
                address = got['key_hash']
                if address in run.predicted:              # predicted earlier in this run: joined
                    status = 'waited'
                elif tainted:
                    status = 'tainted'
                elif got['exists']:
                    sample = got['sample']
            else:
                key = (group.id, pool)
                run.numbers[key] += 1                     # the real run draws this number too
                if tainted:
                    status = 'tainted'
                else:
                    got = await self.manager.call('peek', body=body, partition=group.partition,
                                                  index=run.numbers[key], model=backend,
                                                  randomness='stochastic')
                    sample = got['sample'] if got['exists'] else None
            if status == 'waited':
                content = run.predicted[address]
                run.occurrence[node.name] += 1
                run.rows.append({'node': node.name, 'status': 'waited', 'completion': 0.0,
                                 'prompt': 0.0, 'no_history': False})
            elif sample is not None:
                status = 'hit'
                content = sample.get('content') or ''
                pt = sample.get('prompt_tokens') or 0
                ct += sample.get('completion_tokens') or 0
                run.occurrence[node.name] += 1
                run.rows.append({'node': node.name, 'status': 'hit', 'completion': 0.0,
                                 'prompt': 0.0, 'no_history': False})
            else:
                content = run.stand_in(node.name)
                if address is not None:
                    run.predicted[address] = content
                est = run.estimate(node.name, body)
                run.rows.append({'node': node.name, 'status': status,
                                 'completion': est['completion'], 'prompt': est['prompt'],
                                 'no_history': est['no_history']})
                pt, ct = int(est['prompt']), ct + int(est['completion'])
            choices.append({'index': i, 'finish_reason': 'stop', 'logprobs': None,
                            'message': {'role': 'assistant', 'content': content, 'refusal': None,
                                        'tool_calls': None, 'function_call': None,
                                        'audio': None}})
        return {'id': 'evalrt-forecast', 'object': 'chat.completion', 'created': int(time.time()),
                'model': node.name, 'system_fingerprint': None, 'service_tier': None,
                'choices': choices,
                'usage': {'prompt_tokens': pt, 'completion_tokens': ct, 'total_tokens': pt + ct,
                          'prompt_tokens_details': None, 'completion_tokens_details': None},
                'evalrt': {'run': run.id, 'group': group.id, 'forecast': True}}

    # -- tools ------------------------------------------------------------------------------------

    async def _tool(self, run, group, node, args, kwargs, task):
        kwargs = dict(kwargs)
        if 'rng' in kwargs:
            raise _err('invalid_call', 'the system passes `rng` to a stochastic tool')
        if getattr(run, 'forecast', False):          # a plain call: no memo, nothing written
            if node.takes_rng:
                kwargs['rng'] = random.Random(random.SystemRandom().getrandbits(64))
            return await self._execute(node, args, kwargs)
        try:
            ident = digest({'tool': node.name, 'rev': node.rev,
                            'args': json.loads(json.dumps([list(args), kwargs]))})
        except (TypeError, ValueError):
            raise _err('invalid_call', 'tool %s takes JSON arguments only' % node.name)
        t0 = time.time()
        index = None
        if not node.cache:
            if node.takes_rng:
                kwargs['rng'] = random.Random(random.SystemRandom().getrandbits(64))
            out, status = await self._execute(node, args, kwargs), 'executed'
        elif node.randomness == 'deterministic':
            out, status = await self._memo_tool(run, group, node, ident, args, kwargs)
        else:
            ticket = await run.ensure(group)
            got = await self._lease_call('kind_draw', ticket=ticket, case=CASE,
                                         partition=group.partition, identity=ident, role=None,
                                         attempt=run.attempt, **self._hold(node))
            index = got['index']
            if got['value'] is not None:
                out = json.loads(got['value'])
                status = self._read_status(run, got)
            else:
                lease = got.get('lease')         # None from a manager without leases
                kwargs['rng'] = random.Random(got['seed'])

                def put(text, index=index):
                    return self.manager.call('kind_put', ticket=ticket, case=CASE,
                                             partition=group.partition, identity=ident,
                                             index=index, value=text, kind=node.name,
                                             kind_rev=node.rev)
                done = self._late_row(run, group, node, ident, index,
                                      't:%s:%d' % (ident, index), ticket)
                err = None
                try:
                    out, _, first = await self._compute(run, node, args, kwargs, lease, put,
                                                        done)
                except _ToolFailed as failed:
                    err = failed.err
                if err is not None:
                    raise err                            # outside the handler: its own context
                status = run.count('t:%s:%d' % (ident, index), 'generated', ticket, ticket) \
                    if first else 'duplicate'
        self._row({'ev': 'call', 'run': run.id, 'group': group.id, 'node': node.name,
                   'kind': 'tool', 'randomness': node.randomness, 'cache': node.cache,
                   'identity': ident, 'index': index, 'status': status, 't0': t0,
                   't1': time.time(), 'task': task})
        return out

    async def _memo_tool(self, run, group, node, ident, args, kwargs):
        """Deterministic + cache: one global value (U4). Identical concurrent calls in this process
        compute it once; across processes the manager's lease does the same (2026-10-13). Every
        caller gets its own copy of the value. A failure that is not the tool's own (a cancel, the
        cap) makes the waiters try again; a failure of the tool's own is raised to this process's
        waiters, while waiters elsewhere take the lease over and compute it themselves. A waiter
        here waits at most the tool's `max_hold`, then asks the manager as another process would
        (the manager holds it to the same bound, so it reads the value or computes it itself); a
        value it obtains settles the pending call for every later caller here."""
        while True:
            pending = self._memo.get(ident)
            if pending is not None:
                done, _ = await asyncio.wait([pending], timeout=node.max_hold)  # never cancels it
                if not done:                             # only with max_hold: run past its bound
                    err = None
                    try:
                        text, status = await self._memo_fetch(run, group, node, ident, args,
                                                              kwargs)
                    except _ToolFailed as failed:
                        err = failed.err
                    if err is not None:
                        raise err
                    if self._memo.get(ident) is pending and not pending.done():
                        pending.set_result(('ok', text))     # later callers need not wait
                    return json.loads(text), status
                kind, value = pending.result()
                if kind == 'ok':
                    return json.loads(value), 'waited'
                if kind == 'raised':
                    raise value
                continue                                 # 'retry'
            future = asyncio.get_running_loop().create_future()
            self._memo[ident] = future
            try:
                text, status = await self._memo_fetch(run, group, node, ident, args, kwargs)
            except _ToolFailed as failed:
                if not future.done():                    # not settled by an overdue waiter
                    if self._memo.get(ident) is future:
                        del self._memo[ident]            # never memoised
                    future.set_result(('raised', failed.err))
                elif future.result()[0] == 'ok':         # an overdue waiter stored the value:
                    return json.loads(future.result()[1]), 'duplicate'  # this call gets it too;
                                                         # its own tool ran (and failed)
                err = failed.err
            except BaseException:
                if not future.done():
                    if self._memo.get(ident) is future:
                        del self._memo[ident]
                    future.set_result(('retry', None))
                raise
            else:
                if not future.done():
                    future.set_result(('ok', text))
                return json.loads(text), status
            raise err                                    # outside the handler: its own context

    async def _memo_fetch(self, run, group, node, ident, args, kwargs):
        """One look at the manager for a deterministic cached tool: the stored value, the value
        after waiting on another requester's lease, or this process computes it under its own
        lease. Returns (text, status); the tool's own failure comes back as _ToolFailed."""
        ticket = await run.ensure(group)
        got = await self._lease_call('memo_get', ticket=ticket, case=CASE, identity=ident,
                                     **self._hold(node))
        if got['value'] is not None:
            return got['value'], self._read_status(run, got)
        lease = got.get('lease')                     # None from a manager without leases

        def put(text):
            return self.manager.call('memo_put', ticket=ticket, case=CASE, identity=ident,
                                     value=text, kind=node.name, kind_rev=node.rev)
        done = self._late_row(run, group, node, ident, None, 't:' + ident, ticket)
        _, text, first = await self._compute(run, node, args, kwargs, lease, put, done)
        return text, (run.count('t:' + ident, 'generated', ticket, ticket) if first
                      else 'duplicate')

    @staticmethod
    def _hold(node):
        """The lease request's `max_hold`, only when the tool declares one (the request is
        otherwise byte-identical to a manager that has no lease expiry)."""
        return {} if node.max_hold is None else {'max_hold': node.max_hold}

    async def _lease_call(self, op, **kw):
        """`memo_get` or `kind_draw` asking for a lease. The manager's handler goes on when this
        call is cancelled (the rpc client only drops the reply), so a lease granted to a cancelled
        call would be held by nobody: a late reply that grants one releases it. This process
        promises to renew (to a manager that supports it); a lease granted with a `ttl` is renewed
        every third of it until it is written or released."""
        if self.lease_caps:
            kw['renew'] = True
        task = asyncio.ensure_future(self.manager.call(op, lease=True, **kw))
        try:
            got = await asyncio.shield(task)
        except asyncio.CancelledError:
            task.add_done_callback(self._late_reply)
            raise
        ttl = got.get('ttl')
        if got.get('lease') and type(ttl) in (int, float) and 0 < ttl < float('inf'):
            self._renewals[got['lease']] = asyncio.ensure_future(
                self._renew(got['lease'], max(ttl / 3.0, RENEW_MIN)))
        return got

    async def _renew(self, lease, every):
        """Keep `lease` alive while this process computes it. Stops when the manager says it has
        ended (written, released, expired) or the connection is gone (the manager dropped it);
        any other error is tried again at the next tick."""
        try:
            while True:
                await asyncio.sleep(every)
                try:
                    got = await self.manager.call('lease_renew', lease=lease)
                except EvalrtError as err:
                    if err.kind == 'disconnected':
                        return
                    continue
                except Exception:                    # noqa: BLE001 -- e.g. a malformed reply
                    continue
                if not isinstance(got, dict) or not got.get('renewed'):
                    return
        finally:
            if self._renewals.get(lease) is asyncio.current_task():
                del self._renewals[lease]

    def _unrenew(self, lease):
        """The lease has ended here (its value is written, or it is being released)."""
        task = self._renewals.pop(lease, None) if lease is not None else None
        if task is not None:
            task.cancel()

    def _late_reply(self, task):
        if not task.cancelled() and task.exception() is None:
            self._release(task.result().get('lease'))

    async def _compute(self, run, node, args, kwargs, lease, put, done):
        """Execute a cached tool under this process's lease (None: a manager without leases), then
        write its result with `put(text)`; returns (result, text, first). When another requester
        wrote first (an expired lease's late holder, or its taker), the stored value replaces this
        one and `first` is False, so every caller gets the one stored value. The tool's own
        exception comes back as _ToolFailed. Every failure releases the lease, and only once the
        tool has stopped. A cancel does not throw away work that cannot be stopped or is already
        done: a sync tool's thread, a finished result or a write in flight is still written when it
        ends (`done()` then traces it), so no waiter executes the same call again meanwhile."""
        try:
            self._check_cap(run)
        except BaseException:
            self._release(lease)
            raise
        task = asyncio.ensure_future(self._execute(node, args, kwargs))
        try:
            out = await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done() or not inspect.iscoroutinefunction(node.fn):
                self._late_task(task, functools.partial(self._finish_late, node, lease, put,
                                                        done))
            else:                                        # an async tool: stop it, then release
                task.cancel()
                self._late_task(task, lambda t: self._release(lease))
            raise
        except Exception as err:                         # the tool's own failure
            self._release(lease)
            raise _ToolFailed(err)
        except BaseException:
            self._release(lease)
            raise
        try:
            text = self._json(node, out)
        except BaseException:
            self._release(lease)
            raise
        write = asyncio.ensure_future(put(text))
        try:
            await asyncio.shield(write)
        except asyncio.CancelledError:
            self._late_task(write, functools.partial(self._written_late, lease, done))
            raise
        except BaseException:
            self._release(lease)
            raise
        self._unrenew(lease)                             # the write ended the lease
        stored = _stored(write.result())
        if stored is not None:                           # another requester wrote first
            return json.loads(stored), stored, False
        return out, text, True

    def _late_task(self, task, callback):
        """Run `callback(task)` once `task` ends; `aclose` waits for such tasks."""
        self._late.add(task)
        task.add_done_callback(self._late.discard)
        task.add_done_callback(callback)

    def _finish_late(self, node, lease, put, done, task):
        """The tool of a cancelled call has ended: write its result (the waiters read it) or
        release the lease."""
        try:
            text = self._json(node, task.result())
        except BaseException:                            # noqa: BLE001 -- raised or bad result
            self._release(lease)
            return
        self._late_task(asyncio.ensure_future(put(text)),
                        functools.partial(self._written_late, lease, done))

    def _written_late(self, lease, done, write):
        if write.cancelled() or write.exception() is not None:
            self._release(lease)
        else:
            self._unrenew(lease)
            done(first=_stored(write.result()) is None)

    def _late_row(self, run, group, node, ident, index, key, ticket):
        """Traces a result written after its call was cancelled: generated once, or a duplicate
        execution when another requester had written first."""
        def done(first=True):
            if first:
                run.count(key, 'generated', ticket, ticket)
            self._row({'ev': 'call', 'run': run.id, 'group': group.id, 'node': node.name,
                       'kind': 'tool', 'randomness': node.randomness, 'cache': node.cache,
                       'identity': ident, 'index': index,
                       'status': 'generated' if first else 'duplicate', 'late': True,
                       't0': None, 't1': time.time(), 'task': None})
        return done

    @staticmethod
    def _read_status(run, got):
        """A stored result: re-read by its own run (a resume), read after waiting on another
        requester's lease (`lease_waited`; `waited` is the wait on this process's own single
        flight), or a plain hit."""
        if run.mine(got['ticket']):
            return 'own'
        return 'lease_waited' if got.get('waited') else 'hit'

    def _release(self, lease):
        """Give a lease back without waiting on the reply (the caller may be cancelled): a waiter
        takes over. A closed connection releases it anyway."""
        if lease is None:
            return
        self._unrenew(lease)
        task = asyncio.ensure_future(self.manager.call('lease_release', lease=lease))
        self._releases.add(task)
        task.add_done_callback(self._released)

    def _released(self, task):
        self._releases.discard(task)
        if not task.cancelled():
            task.exception()                         # a closed connection released it already

    @staticmethod
    def _json(node, out):
        try:
            return json.dumps(out)
        except (TypeError, ValueError):
            raise _err('invalid_result', 'tool %s must return JSON' % node.name)

    @staticmethod
    async def _execute(node, args, kwargs):
        if inspect.iscoroutinefunction(node.fn):
            return await node.fn(*args, **kwargs)
        # A thread of its own: a sync tool may call other sync tools, which block on this loop,
        # so a bounded pool could deadlock.
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        ctx = contextvars.copy_context()

        def work():
            try:
                result = (True, ctx.run(node.fn, *args, **kwargs))
            except BaseException as err:    # noqa: BLE001
                result = (False, err)

            def deliver():
                if not future.done():
                    if result[0]:
                        future.set_result(result[1])
                    else:
                        future.set_exception(result[1])
            loop.call_soon_threadsafe(deliver)
        threading.Thread(target=work, name='evalrt-tool-%s' % node.name, daemon=True).start()
        return await future

    # -- static graphs (U1) -----------------------------------------------------------------------

    def program(self, graph, config=None):
        """The built program of a static v4 graph, built once per (revision, config)."""
        from harness_rsi_smoke import port_graph as pg
        from ..job.program import GraphProgram, ProgramError
        key = (pg.revision(graph), canonical(config or {}))
        prog = self._programs.get(key)
        if prog is None:
            try:
                prog = GraphProgram(graph, config)
            except ProgramError as err:
                raise _err(err.kind, err.detail)
            if getattr(prog, 'version', 3) != 4:
                raise _err('invalid_graph', 'static graphs are v4 port graphs')
            missing = [m for m in prog.models if m not in self.models]
            if missing:
                raise _err('unknown_model', 'the graph calls %s' % missing)
            self._programs[key] = prog
        return prog

    def graph_bound(self, graph, config=None):
        """Before running: the most model calls one input of a static graph makes, per node (a
        growing graph adds its declared bounds)."""
        from ..control.admission import dynamic_bounds
        prog = self.program(graph, config)
        design = prog.design(1, {})
        per_node = {n: design.size(n) for n in sorted(prog.stochastic)}
        dyn_calls, _, _ = dynamic_bounds(prog, design, 1, {})
        return {'calls': sum(per_node.values()) + int(dyn_calls), 'per_node': per_node,
                'dynamic_calls': int(dyn_calls)}

    async def execute(self, graph, inputs, group, run_id=None, label=None, config=None,
                      cache_salt=None):
        """Execute a declared static graph (U1). The system runs it itself, as its own run in
        `group`: nothing is intercepted, no agent run or context is consulted, and the program is
        built once per (revision, config). Each input is a dict: `prompt`, optional `id`, `case`,
        `pool` and the fields its registered kinds read. Returns {run, results}, results per input
        {id, leaves, executions}. `run_id` resumes an earlier run of the graph; `cache_salt` as
        for `run`."""
        return await self._bridge(self._execute_graph(graph, inputs, group, run_id, label,
                                                      config, cache_salt))

    def execute_sync(self, graph, inputs, group, run_id=None, label=None, config=None,
                     cache_salt=None):
        return self._submit(self._execute_graph(graph, inputs, group, run_id, label, config,
                                                cache_salt))

    async def forecast(self, graph, inputs, group, config=None, history=None):
        """Zero-call forecast of `execute(graph, inputs, group)` (claims C2). `group` is a Group or
        one Group per input. The program is walked as `execute` walks it, but every generate draw
        reads the store without claiming (manager `peek`) at the address the run would read (a new
        run's number for a canonical repeat is its role): a stored sample is a hit and is used
        downstream; anything else is predicted to generate, and its output is an unknown marker,
        so every node that reads it is predicted to generate as well (closure). Registered kinds
        run in process and write nothing (a stochastic cached kind is unknown). `history`
        {node: {'completion': mean tokens, 'prompt_per_char': tokens per character}} gives the
        estimates. Returns the prediction: calls, completion and prompt tokens, hits, per node and
        per input. Nothing is written and no model is called."""
        return await self._bridge(self._forecast_graph(graph, inputs, group, config, history))

    async def _forecast_graph(self, graph, inputs, group, config, history):
        from ..job.port_executor import PortCaseRun
        prog = self.program(graph, config)
        if prog.dynamic:
            raise _err('not_static', 'a graph with templates grows at run time: forecast it as '
                       'agent code (forecast_run)')
        groups = list(group) if isinstance(group, (list, tuple)) else [group] * len(inputs)
        if len(groups) != len(inputs) or not all(isinstance(g, Group) for g in groups):
            raise _err('invalid_group', 'forecast needs a Group or one Group per input')
        design = prog.design(1, {})
        uses = any(isinstance(v, dict) and v.get('kind') == 'generate'
                   and ('model' in v or 'sampling' in v)
                   for v in list(prog.graph['nodes'].values()) + list(prog.templates.values()))
        tables = {n: m['defaults'] for n, m in self.models.items()
                  if m.get('defaults') is not None} if uses else None
        est = ForecastRun(self, groups[0] if groups else None, None, history)
        ticket, started = 'forecast', time.time()
        predicted = {}                          # greedy addresses predicted in this forecast
        # Numbers this forecast's run would claim, per (partition, model, pool). `execute` runs all
        # its inputs as one consumer (the case is not in the key, N14), so inputs that send the
        # identical request claim successive numbers: the manager's rule (flights._claim) is the
        # draw's rank if this run has not claimed it, else the smallest unclaimed number (F1d).
        claimed = {}

        def number(partition, model, pool, rank):
            taken = claimed.setdefault((partition, model, pool), set())
            index = rank
            if index is None or index in taken:
                index = 1
                while index in taken:
                    index += 1
            taken.add(index)
            return index

        def make_draw(rec):
            async def draw(node, body, case_id, partition, role, rank, model=None):
                head = await self.manager.call('peek', body=body, partition=partition,
                                               index=None, model=model)
                index = rank if head['greedy'] else number(partition, model, head['pool'], rank)
                got = head if head['greedy'] else await self.manager.call(
                    'peek', body=body, partition=partition, index=index, model=model)
                if got['greedy'] and got['key_hash'] in predicted:
                    rec.append({'node': node, 'status': 'waited', 'completion': 0.0,
                                'prompt': 0.0, 'no_history': False})
                    return predicted[got['key_hash']]
                if MARK_PREFIX in _text_of(body):
                    status = 'tainted'
                else:
                    status = 'hit' if got['exists'] else ('inflight' if got['inflight']
                                                          else 'miss')
                if status == 'hit':
                    rec.append({'node': node, 'status': 'hit', 'completion': 0.0,
                                'prompt': 0.0, 'no_history': False})
                    return {'sample': got['sample'], 'key_hash': got['key_hash'],
                            'pool': got['pool'], 'index': got['index'], 'partition': partition,
                            'generated_by': 'store', 'created_at': 0.0, 'attempts': 0,
                            'greedy': got['greedy'], 'source': 'stored'}
                e = est.estimate(node, body)
                rec.append({'node': node, 'status': status, 'completion': e['completion'],
                            'prompt': e['prompt'], 'no_history': e['no_history']})
                uid = uuid.uuid4().hex[:12]
                out = {'sample': {'content': MARK % uid, 'prompt_tokens': int(e['prompt']),
                                  'completion_tokens': int(e['completion']),
                                  'finish_reason': 'stop'},
                       'key_hash': 'forecast:' + uid, 'pool': got['pool'], 'index': index,
                       'partition': partition, 'generated_by': ticket, 'created_at': started,
                       'attempts': 0, 'source': 'forecast'}
                if got['greedy']:
                    predicted[got['key_hash']] = dict(out, generated_by='forecast-joined',
                                                      created_at=0.0)
                return out
            return draw

        async def memo_get(case_id, ident, domain=None):
            return None, None

        async def memo_put(case_id, ident, value, kind, kind_rev, domain=None):
            return None

        async def kind_draw(case_id, partition, ident, role):
            return {'value': MARK % uuid.uuid4().hex[:12], 'ticket': 'forecast', 'index': 1,
                    'seed': 0}

        async def kind_put(case_id, partition, ident, index, value, kind, kind_rev):
            return None
        job_memo, runs, recs, ids = {}, [], [], set()
        for k, (inp, g) in enumerate(zip(inputs, groups)):
            case = dict(inp)
            case.setdefault('pool', [])
            case.setdefault('id', 'i%d' % k)
            if case['id'] in ids:
                raise _err('invalid_call', 'input ids must be distinct (%r)' % case['id'])
            ids.add(case['id'])
            rec = []
            recs.append((case['id'], rec))
            runs.append(PortCaseRun(prog, design, case, make_draw(rec), ticket, started,
                                    (lambda part: (lambda name: part))(g.partition),
                                    memo=(memo_get, memo_put), private='fresh/forecast',
                                    kind_draw=(kind_draw, kind_put), job_memo=job_memo,
                                    tables=tables))
        tasks = [t for r in runs for t in r.start()]

        def pending():
            return [t for r in runs for t in r.cells.values() if not t.done()]
        try:
            await asyncio.gather(*tasks)
            while pending():
                await asyncio.gather(*pending())
        finally:
            while pending():
                left = pending()
                for t in left:
                    t.cancel()
                await asyncio.gather(*left, return_exceptions=True)
        # closure: a generate node never reached below a predicted miss would generate too
        nodes = prog.graph['nodes']
        children = collections.defaultdict(set)
        for e in prog.graph.get('edges', []):
            children[e['from']].add(e['to'])

        def below(roots):
            seen, todo = set(), list(roots)
            while todo:
                for c in children.get(todo.pop(), ()):
                    if c not in seen:
                        seen.add(c)
                        todo.append(c)
            return seen
        out = {'calls': 0, 'completion_tokens': 0.0, 'prompt_tokens': 0.0, 'hits': 0,
               'no_history': 0, 'by_node': {}, 'per_input': []}
        for cid, rec in recs:
            reached = {r['node'] for r in rec}
            missed = {r['node'] for r in rec if r['status'] not in ('hit', 'waited')}
            for n in sorted(below(missed) - reached):
                if isinstance(nodes.get(n), dict) and nodes[n].get('kind') == 'generate':
                    h = (history or {}).get(n) or {}
                    c = h.get('completion')
                    cap = prog.config.get('actor_max_tokens')
                    mt = float(nodes[n].get('max_tokens') or 0)
                    rec.append({'node': n, 'status': 'closure',
                                'completion': float(c) if c is not None
                                else (min(mt, cap) if cap else mt),
                                'prompt': float(h.get('prompt') or 0.0),
                                'no_history': c is None})
            for r in rec:
                bn = out['by_node'].setdefault(r['node'], {'generate': 0, 'hit': 0,
                                                           'completion_tokens': 0.0})
                if r['status'] in ('hit', 'waited'):
                    out['hits'] += 1
                    bn['hit'] += 1
                else:
                    out['calls'] += 1
                    out['completion_tokens'] += r['completion']
                    out['prompt_tokens'] += r['prompt']
                    out['no_history'] += r['no_history']
                    bn['generate'] += 1
                    bn['completion_tokens'] += r['completion']
            out['per_input'].append({'id': cid, 'generate': sorted(
                r['node'] for r in rec if r['status'] not in ('hit', 'waited')),
                'hit': sorted(r['node'] for r in rec if r['status'] in ('hit', 'waited'))})
        return out

    async def _execute_graph(self, graph, inputs, group, run_id, label, config, cache_salt=None):
        prog = self.program(graph, config)
        if prog.dynamic:
            raise _err('not_static', 'a graph with templates grows at run time: write it as agent '
                       'code (a static graph declares every node)')
        run = self._new_run(group, run_id, label, cache_salt)
        await run.ensure(group)
        self._row({'ev': 'run_start', 'run': run.id, 'attempt': run.attempt, 'group': group.id,
                   'label': label, 'static': prog.graph_rev})
        try:
            results = await self._run_graph(run, group, graph, inputs, config, 'static')
        finally:
            await self._end(run)
        return {'run': run.id, 'results': results}

    async def _run_graph(self, run, group, graph, inputs, config, task):
        from ..job.port_executor import PortCaseRun
        prog = self.program(graph, config)
        ticket = await run.ensure(group)
        design = prog.design(1, {})
        uses = any(isinstance(v, dict) and v.get('kind') == 'generate'
                   and ('model' in v or 'sampling' in v)
                   for v in list(prog.graph['nodes'].values()) + list(prog.templates.values()))
        tables = {n: m['defaults'] for n, m in self.models.items()
                  if m.get('defaults') is not None} if uses else None

        async def draw(node, body, case_id, partition, role, rank, model=None):
            self._check_cap(run)
            t0 = time.time()
            res = await self.manager.draw(ticket, node, body, CASE, partition=partition,
                                          rank=rank, role=role, model=model,
                                          attempt=run.attempt)
            status = run.count(res['key_hash'], res['source'], ticket, res.get('generated_by'))
            s = res['sample']
            self._row({'ev': 'call', 'run': run.id, 'group': group.id, 'node': node,
                       'kind': 'graph', 'backend': model or 'actor', 'pool': res.get('pool'),
                       'index': res.get('index'), 'key_hash': res['key_hash'], 'status': status,
                       'prompt_tokens': s.get('prompt_tokens'),
                       'completion_tokens': s.get('completion_tokens'), 't0': t0,
                       't1': time.time(), 'task': task, 'input': case_id})
            return res

        async def memo_get(case_id, ident, domain=None):
            got = await self.manager.call('memo_get', ticket=ticket, case=CASE, identity=ident)
            return got['value'], got.get('ticket')

        async def memo_put(case_id, ident, value, kind, kind_rev, domain=None):
            await self.manager.call('memo_put', ticket=ticket, case=CASE, identity=ident,
                                    value=value, kind=kind, kind_rev=kind_rev)

        async def kind_draw(case_id, partition, ident, role):
            return await self.manager.call('kind_draw', ticket=ticket, case=CASE,
                                           partition=partition, identity=ident, role=role,
                                           attempt=run.attempt)

        async def kind_put(case_id, partition, ident, index, value, kind, kind_rev):
            await self.manager.call('kind_put', ticket=ticket, case=CASE, partition=partition,
                                    identity=ident, index=index, value=value, kind=kind,
                                    kind_rev=kind_rev)
        job_memo = {}
        runs, ids = [], set()
        for k, inp in enumerate(inputs):
            case = dict(inp)
            case.setdefault('pool', [])
            case.setdefault('id', 'i%d' % k)     # the caller's id: a case never retrieves itself
            if case['id'] in ids:
                raise _err('invalid_call', 'input ids must be distinct (%r)' % case['id'])
            ids.add(case['id'])
            runs.append(PortCaseRun(prog, design, case, draw, ticket, run.started_at,
                                    lambda name: group.partition, memo=(memo_get, memo_put),
                                    private=run.private, kind_draw=(kind_draw, kind_put),
                                    job_memo=job_memo, tables=tables))
        tasks = [t for r in runs for t in r.start()]

        def pending():
            return [t for r in runs for t in r.cells.values() if not t.done()]
        try:
            await asyncio.gather(*tasks)
            while pending():
                await asyncio.gather(*pending())
        finally:
            while pending():
                left = pending()
                for t in left:
                    t.cancel()
                await asyncio.gather(*left, return_exceptions=True)
        out = []
        for r in runs:
            executions = r.executions()
            for ex in executions:
                if ex.get('kind') is not None:        # registered kinds of the graph
                    self._row({'ev': 'call', 'run': run.id, 'group': group.id,
                               'node': ex['node'], 'kind': 'graph_kind', 'tool': ex['kind'],
                               'status': ex.get('status'), 'index': ex.get('index'),
                               'task': task, 'input': r.case_id})
            out.append({'id': r.case_id, 'leaves': r.leaves(), 'executions': executions})
        return out

    # -- helpers ----------------------------------------------------------------------------------

    def _check_cap(self, run):
        """The cap stops a run from issuing new calls once it generated that many results; a call
        already in flight (n draws, concurrent calls) may pass it."""
        if self.cap is not None and run.generated >= self.cap:
            raise _err('run_cap', 'run %s generated %d (the cap)' % (run.id, run.generated))

    @staticmethod
    def _task():
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        name = task.get_name() if task is not None else None
        return '%s/%s' % (threading.current_thread().name, name) if name else \
            threading.current_thread().name

    def _row(self, row):
        if self._trace is None:
            return
        row = dict(row, t=round(time.time(), 4))
        self._trace.write(json.dumps(row) + '\n')
        self._trace.flush()
