"""Upstream model servers of a Copula server (DESIGN.md §3.2 items 1, 6, 8; §10.2 item 3).

    --upstream URL [--served-model NAME] [--model-dir DIR] [--defaults JSON|@FILE]
                   [--identity-extra JSON|@FILE] [--engine-seed auto|address|off]
                   [--api-key-env VAR] [--model NAME[=SERVED] ...]

Without --upstream, `copula run` uses its own OPENAI_BASE_URL: the value the child would have used
without Copula. Credentials stay where they were: the server sends the key in $OPENAI_API_KEY of
its own environment (or --api-key-env), never one taken from a client.

Kinds, told apart by GET /models (`owned_by`):
- vllm, sglang (self-hosted). Every sampling parameter is sent explicitly, from a pinned table:
  --defaults, else the model's generation_config.json (what vLLM applies by default, so behaviour
  is native), else neutral values (a note says so). A LoRA adapter takes its base model's table.
  The identity is established (V5) only from model files the server itself loaded: --model-dir,
  or the local directory the server reports as `root` when the server runs on this host; it
  covers config, tokenizer and every weight file (shards too; a re-saved checkpoint is a new
  revision). A Hugging Face id or a remote server gives the sampling table from this host's cache
  but no established identity: greedy calls are then cached per group (W12). vLLM requests carry
  an engine seed derived from the sample address (an HMAC under the store secret, §3.2 item 6):
  a sample never depends on the server's --seed.
- api (any other OpenAI-compatible endpoint). temperature, top_p and the penalties are always sent
  (from the table, OpenAI's defaults, when the client set none), so a request with an explicit
  default and one without, which share a key, are sent alike; top_k, min_p, repetition_penalty
  and chat_template_kwargs only when not neutral. `max_tokens` becomes `max_completion_tokens`
  for models that want it. The identity is the endpoint and model name, not established.
Misconfiguration (HTTP 401, 403, 404, 405, 407) is retried and never stored as a sample.

Every model the upstream lists is a backend of its own (its own model_rev); `--model A=B` adds the
client-side name A for served model B, and `--model '*'` (or `*=B`) sends unknown names to the
default model.
"""
import hashlib
import hmac
import http.client
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

from .kernel.manager import vllm_client as vc
from .kernel.manager.model import RequestError, TransportError
from .kernel.testing.mock_model import MOCK_DEFAULTS

NEUTRAL = {'temperature': 1.0, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0, 'presence_penalty': 0.0,
           'frequency_penalty': 0.0, 'repetition_penalty': 1.0, 'stop': [],
           'chat_template_kwargs': {}}
# generation_config.json fields vLLM turns into default sampling parameters
GENERATION_FIELDS = ('temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty')
# an OpenAI-style API: these are always sent (from the table when the client set none), so an
# explicit value and an absent one, which share a key, are also sent alike; the others exist only
# on some servers and are sent only when they differ from neutral
API_DEFAULTS = dict(NEUTRAL)
API_OPTIONAL = ('top_k', 'min_p', 'repetition_penalty', 'chat_template_kwargs')
# a 4xx that says the server or the credentials are misconfigured: retried, never stored
CONFIG_STATUS = (401, 403, 404, 405, 407)
WEIGHT_EXT = ('.safetensors', '.bin', '.pt', '.pth', '.gguf')


class UpstreamError(Exception):
    pass


def json_arg(value):
    """A JSON value given inline or as @FILE (None stays None)."""
    if value is None:
        return None
    if isinstance(value, str) and value.startswith('@'):
        with open(value[1:], encoding='utf-8') as f:
            return json.load(f)
    return json.loads(value) if isinstance(value, str) else value


def openai_base(url):
    """The OpenAI base URL requests go to (`http://h:8000` -> `http://h:8000/v1`; a URL with a
    path, such as `.../v1beta/openai`, is kept as given)."""
    url = url.rstrip('/')
    return url + '/v1' if urllib.parse.urlparse(url).path in ('', '/') else url


def server_root(url):
    """Where a self-hosted server answers /version (the base without its trailing /v1)."""
    oa = openai_base(url)
    if oa.endswith('/v1'):
        return oa[:-3]
    p = urllib.parse.urlparse(oa)
    return '%s://%s' % (p.scheme, p.netloc)


base_of = server_root


def is_local(url):
    host = (urllib.parse.urlparse(openai_base(url)).hostname or '').lower()
    names = {'127.0.0.1', 'localhost', '::1', socket.gethostname().lower()}
    try:
        names.add(socket.getfqdn().lower())
    except OSError:
        pass
    return host in names


def _get_json(url, key=None, timeout=10.0):
    req = urllib.request.Request(url, headers={'Authorization': 'Bearer %s' % key} if key else {})
    with urllib.request.urlopen(req, timeout=timeout) as handle:
        return json.loads(handle.read().decode('utf-8'))


def describe(url, key=None, timeout=10.0, wait=0.0):
    """{kind, base, local, version, models: [{id, root, parent, owned_by}]} of the server at
    `url` (waits up to `wait` seconds for it to answer)."""
    oa = openai_base(url)
    deadline = time.time() + wait
    while True:
        try:
            listing = _get_json(oa + '/models', key, timeout)
            break
        except (urllib.error.URLError, OSError, ValueError) as err:
            if time.time() >= deadline:
                raise UpstreamError('the upstream %s does not answer GET /models: %s' % (oa, err))
            time.sleep(1.0)
    models = [m for m in listing.get('data', []) if isinstance(m, dict) and m.get('id')]
    owners = {str(m.get('owned_by', '')).lower() for m in models}
    kind = 'vllm' if 'vllm' in owners else 'sglang' if 'sglang' in owners else 'api'
    version = None
    if kind != 'api':
        for path in ('/version', '/get_server_info'):
            try:
                version = _get_json(server_root(url) + path, key, timeout).get('version')
                break
            except (urllib.error.URLError, OSError, ValueError, AttributeError):
                continue
    return {'kind': kind, 'base': oa, 'local': is_local(url), 'version': version,
            'models': [{'id': m['id'], 'root': m.get('root'), 'parent': m.get('parent'),
                        'owned_by': m.get('owned_by')} for m in models]}


def hf_snapshot(repo):
    """The local Hugging Face cache snapshot of `repo` (org/name), or None."""
    if not isinstance(repo, str) or repo.count('/') != 1 or os.path.isabs(repo):
        return None
    homes = [os.environ.get('HF_HUB_CACHE'),
             os.path.join(os.environ['HF_HOME'], 'hub') if os.environ.get('HF_HOME') else None,
             os.path.expanduser('~/.cache/huggingface/hub')]
    for hub in [h for h in homes if h]:
        d = os.path.join(hub, 'models--' + repo.replace('/', '--'))
        snaps = os.path.join(d, 'snapshots')
        if not os.path.isdir(snaps):
            continue
        try:
            with open(os.path.join(d, 'refs', 'main')) as f:
                rev = f.read().strip()
            if os.path.isdir(os.path.join(snaps, rev)):
                return os.path.join(snaps, rev)
        except OSError:
            pass
        found = sorted(os.listdir(snaps))
        if len(found) == 1:
            return os.path.join(snaps, found[0])
    return None


def locate(root, local):
    """(directory, exact) for the `root` a server reports: a local directory the server itself
    loaded (exact, when the server runs on this host), or a Hugging Face cache snapshot on this
    host (not exact: the server may have loaded another revision). (None, False) otherwise."""
    if isinstance(root, str) and os.path.isabs(root) and os.path.isdir(root):
        return root, bool(local)
    snap = hf_snapshot(root)
    return (snap, False) if snap else (None, False)


def generation_defaults(model_dir):
    """The pinned table vLLM would apply natively: neutral values overridden by the model's
    generation_config.json. (table, source)."""
    table = dict(NEUTRAL)
    path = os.path.join(model_dir, 'generation_config.json') if model_dir else None
    if not path or not os.path.isfile(path):
        return table, 'neutral'
    with open(path, encoding='utf-8') as f:
        gen = json.load(f)
    for name in GENERATION_FIELDS:
        if name in gen and gen[name] is not None:
            table[name] = int(gen[name]) if name == 'top_k' else float(gen[name])
    return table, 'generation_config'


def _file_id(path):
    """A Hugging Face blob name (the sha256 the cache names it by), else size and mtime."""
    target = os.path.realpath(path)
    blob = os.path.basename(target)
    if os.path.islink(path) and len(blob) == 64 and all(c in '0123456789abcdef' for c in blob):
        return blob
    st = os.stat(target)
    return 'stat:%d:%d' % (st.st_size, st.st_mtime_ns)


def model_identity(model_dir):
    """What the distribution depends on in a model directory: the kernel's file hashes plus every
    weight file (sharded checkpoints included), or None when there is no config.json."""
    try:
        ident = {'files': vc.file_identity(model_dir)}
    except (OSError, ValueError):
        return None
    ident['weights'] = {n: _file_id(os.path.join(model_dir, n))
                        for n in sorted(os.listdir(model_dir)) if n.endswith(WEIGHT_EXT)}
    return ident


def adapter_identity(adapter_dir):
    """The identity of a LoRA adapter directory, or None."""
    cfg = os.path.join(adapter_dir or '', 'adapter_config.json')
    if not adapter_dir or not os.path.isfile(cfg):
        return None
    out = {}
    for n in sorted(os.listdir(adapter_dir)):
        if n == 'adapter_config.json' or n.endswith(WEIGHT_EXT):
            p = os.path.join(adapter_dir, n)
            with open(p, 'rb') as f:
                out[n] = hashlib.sha256(f.read()).hexdigest() if n.endswith('.json') \
                    else _file_id(p)
    return out


def engine_seed(secret, key_hash):
    """The engine seed of one sample address: an HMAC under the store secret (31 bits)."""
    mac = hmac.new(secret.encode('utf-8') if isinstance(secret, str) else secret,
                   str(key_hash).encode('utf-8'), hashlib.sha256).hexdigest()
    return int(mac[:8], 16) & 0x7FFFFFFF


class _HTTP:
    """The kernel vLLM client's transport against an OpenAI base URL, with an Authorization
    header, a payload hook, and misconfiguration (401/403/404/405/407) retried, never stored."""
    api_key = None
    endpoint = '/chat/completions'

    def payload_for(self, body, key_hash, salt):
        return self.payload(body, salt)

    def retry_payload(self, status, detail, payload):
        """A corrected payload to send again at once after a 400, or None."""
        return None

    async def generate(self, body, key_hash):
        return (await self.generate_with_meta(body, key_hash, {}))[0]

    async def generate_with_meta(self, body, key_hash, ctx):
        import asyncio
        import functools
        ctx = ctx or {}
        call = functools.partial(self._send, body, key_hash, ctx.get('salt'), ctx.get('route'),
                                 ctx.get('attempt') or 1, ctx.get('failed_replica'))
        return await asyncio.get_running_loop().run_in_executor(self._pool, call)

    def _send(self, body, key_hash, salt, route, attempt, avoid):
        replica = self.replica_for(route, attempt, avoid)
        url = self.urls[replica]
        meta = {'t_send': None, 't_done': None, 'instance_url': url, 'replica': replica,
                'salt': salt, 'http_status': None, 'error': None, 'response_id': None,
                'prompt_tokens': None, 'completion_tokens': None, 'cached_tokens': None,
                'cached_present': None, 'finish_reason': None}
        payload = self.payload_for(body, key_hash, salt)
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer %s' % self.api_key
        for _ in range(2):
            request = urllib.request.Request(
                url + self.endpoint, data=json.dumps(payload, ensure_ascii=False).encode(),
                method='POST', headers=headers)
            meta['t_send'] = time.time()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                    raw = handle.read()
                    status = getattr(handle, 'status', 200)
                break
            except urllib.error.HTTPError as err:
                meta.update(t_done=time.time(), http_status=err.code)
                detail = err.read()[:500].decode('utf-8', 'replace')
                fixed = self.retry_payload(err.code, detail, payload) if err.code == 400 else None
                if fixed is not None:
                    payload = fixed
                    continue
                if err.code in vc.RETRY_STATUS or err.code in CONFIG_STATUS or err.code >= 500:
                    meta['error'] = 'http_%d' % err.code
                    raise vc._failed(TransportError('HTTP %d: %s' % (err.code, detail)), meta)
                text = detail.lower()
                if 'context length' in text or 'maximum context' in text or 'too long' in text:
                    meta['error'] = 'context_overflow'
                    raise vc._failed(RequestError('context_overflow', detail), meta)
                meta['error'] = 'http_%d' % err.code
                raise vc._failed(RequestError('http_%d' % err.code, detail), meta)
            except (urllib.error.URLError, socket.timeout, OSError, ValueError,
                    http.client.HTTPException) as err:
                meta.update(t_done=time.time(),
                            error='timeout' if vc._is_timeout(err) else 'connection')
                self.mark_down(replica)
                raise vc._failed(TransportError('%s: %s' % (type(err).__name__,
                                                            str(err)[:300])), meta)
        self.mark_up(replica)
        meta.update(t_done=time.time(), http_status=status)
        try:
            reply = json.loads(raw.decode('utf-8'))
            sample = self.sample_of(reply)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as err:
            meta['error'] = 'bad_reply'
            raise vc._failed(TransportError('unexpected reply: %r' % (err,)), meta)
        meta.update(vc.usage_counts(reply), response_id=reply.get('id'),
                    finish_reason=sample['finish_reason'])
        return sample, meta


class SelfHostedClient(_HTTP, vc.VLLMClient):
    """A vLLM or SGLang server: every parameter explicit; an engine seed per sample address."""

    def __init__(self, *args, secret=None, seed_engine=True, api_key=None, **kw):
        super().__init__(*args, **kw)
        self.secret = secret
        self.seed_engine = bool(seed_engine and secret)
        self.api_key = api_key

    def payload_for(self, body, key_hash, salt):
        out = self.payload(body, salt)
        if self.seed_engine:
            out['seed'] = engine_seed(self.secret, key_hash)
        return out


class ApiClient(_HTTP, vc.VLLMClient):
    """An OpenAI-compatible API: the OpenAI sampling parameters always sent; the others only
    when not neutral; max_tokens renamed when the model wants max_completion_tokens."""

    def __init__(self, *args, api_key=None, **kw):
        super().__init__(*args, **kw)
        self.api_key = api_key
        self.completion_tokens_name = 'max_tokens'

    def payload_for(self, body, key_hash, salt):
        out = {}
        for k, v in body.items():
            if (k in API_OPTIONAL and v == NEUTRAL[k]) or (k == 'stop' and not v):
                continue
            out[k] = v
        if 'max_tokens' in out and self.completion_tokens_name != 'max_tokens':
            out[self.completion_tokens_name] = out.pop('max_tokens')
        out['model'] = self.served_model
        return out

    def retry_payload(self, status, detail, payload):
        if 'max_completion_tokens' in detail and 'max_tokens' in payload:
            self.completion_tokens_name = 'max_completion_tokens'
            out = dict(payload)
            out['max_completion_tokens'] = out.pop('max_tokens')
            return out
        return None


def _slug(i):
    return 'm%d' % i


def build(args, secret, environ=None):
    """The upstream backends for the kernel manager from parsed options.

    Returns {actor, models: {kernel name: client}, names: {client name: kernel name},
    identity: {kernel name: bool}, notes: [str], describe: {...}, engine_seed}."""
    env = os.environ if environ is None else environ
    url = args.upstream
    key = env.get(args.api_key_env or 'OPENAI_API_KEY') or None
    if key == 'copula':                         # the launcher's placeholder, not a credential
        key = None
    try:
        info = describe(url, key, wait=getattr(args, 'wait_ready', 0.0) or 0.0)
    except UpstreamError:
        if not args.served_model:
            raise
        info = {'kind': 'api', 'base': openai_base(url), 'local': is_local(url),
                'version': None,                # no model listing: the named model only
                'models': [{'id': args.served_model, 'root': None, 'parent': None,
                            'owned_by': None}]}
    served = [m['id'] for m in info['models']]
    if args.served_model:
        if served and args.served_model not in served:
            raise UpstreamError('%s does not serve %r (it serves: %s)'
                                % (info['base'], args.served_model, ', '.join(served)))
        order = [args.served_model] + [m for m in served if m != args.served_model]
    else:
        order = served
    if not order:
        raise UpstreamError('%s lists no models; give --served-model' % info['base'])
    by_id = {m['id']: m for m in info['models']}
    defaults_arg = json_arg(args.defaults)
    extra = json_arg(args.identity_extra) or {}
    seed_mode = args.engine_seed
    notes = []
    if info['kind'] == 'api':
        if seed_mode == 'address':
            notes.append('an API upstream takes no engine seed (--engine-seed ignored)')
        seed_mode = 'off'
        notes.append('API upstream: temperature and top_p are always sent (1.0 when the client '
                     'sets none, as OpenAI applies); for a provider with other defaults pass '
                     '--defaults')
    elif seed_mode == 'auto':
        seed_mode = 'address' if info['kind'] == 'vllm' else 'off'
    clients, identity, names, sources, unknown = {}, {}, {}, {}, {}
    for i, sid in enumerate(order):
        kname = 'actor' if i == 0 else 'm%d' % i
        common = dict(timeout=args.timeout, workers=args.workers)
        if info['kind'] == 'api':
            table = dict(defaults_arg or API_DEFAULTS)
            client = ApiClient(info['base'], sid, table, {'endpoint': info['base'], 'model': sid},
                               api_key=key, **common)
            identity[kname] = False
        else:
            m = by_id.get(sid, {'root': None, 'parent': None})
            parent = by_id.get(m.get('parent')) if m.get('parent') else None
            base = parent or m
            if i == 0 and args.model_dir:
                mdir, exact = args.model_dir, True
            else:
                mdir, exact = locate(base.get('root'), info['local'])
            mid = model_identity(mdir) if mdir else None
            if defaults_arg is not None and i == 0:
                table, source = dict(defaults_arg), '--defaults'
            else:
                table, source = generation_defaults(mdir)
            ident = {'server': {'kind': info['kind'], 'version': info['version'], 'model': sid},
                     'extra': extra}
            if mid:
                ident['model'] = mid
            established = bool(mid and exact)
            if parent is not None:              # a LoRA adapter over `parent`
                adir, aexact = locate(m.get('root'), info['local'])
                aid = adapter_identity(adir)
                ident['adapter'] = aid
                established = established and bool(aid and aexact)
            identity[kname] = established
            sources[sid] = source
            if not established:
                unknown[sid] = ('no model files found' if not mid else
                                'model files from a cache snapshot or another host')
            client = SelfHostedClient(info['base'], sid, table, ident, secret=secret,
                                      seed_engine=seed_mode == 'address', api_key=key, **common)
        clients[kname] = client
        names[sid] = kname
    if sources:
        notes.append('sampling defaults: ' + '; '.join(
            '%s %s' % (sid, src) + (' (native vLLM would apply the model\'s generation_config: '
                                    'pass --model-dir)' if src == 'neutral' else '')
            for sid, src in sources.items()))
    for sid, why in unknown.items():
        notes.append('identity of %s not established (%s): greedy calls are cached per group '
                     '(W12); pass --model-dir to establish it' % (sid, why))
    for alias in args.model or []:
        name, sep, target = alias.partition('=')
        target = target if sep else order[0]
        if target not in names:
            raise UpstreamError('--model %s: the upstream does not serve %r' % (alias, target))
        names[name] = names[target]
    actor = clients.pop('actor')
    return {'actor': actor, 'models': clients, 'names': names, 'identity': identity,
            'notes': notes, 'describe': info, 'engine_seed': seed_mode}


def mock(args):
    """The mock backend (trials and tests): {actor, models, names, identity, notes}."""
    from .testing import ScriptedMock, load_script
    actor = ScriptedMock(args.mock_nonce, load_script(args.mock_script), latency=args.mock_latency)
    # without --model: any model name (a program's own names work unchanged); 'mock' is the default
    names = {n.partition('=')[0]: 'actor' for n in (args.model or ['mock', '*'])}
    return {'actor': actor, 'models': {}, 'names': names, 'identity': True,
            'notes': [], 'describe': {'kind': 'mock'}, 'engine_seed': 'off'}


__all__ = ['build', 'mock', 'describe', 'generation_defaults', 'engine_seed', 'UpstreamError',
           'MOCK_DEFAULTS', 'NEUTRAL']
