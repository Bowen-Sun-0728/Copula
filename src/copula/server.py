"""The Copula server of a project (docs/guides/project-server.md).

    copula server [--project DIR] [--listen HOST[:PORT]] [model-server options]

A server owns the project's sample store (the store's exclusive lock keeps it to one), the kernel
manager with its upstream backends, the experiment registry, the call log, and the HTTP endpoint
every run of the project is served from. It writes .copula/server.json (url, host, pid and a
control token; mode 0600) and removes it when it stops. `copula run` in the project finds it
there, so several commands share one store at once; a launcher on another host passes
--server URL and the token in $COPULA_SERVER_TOKEN. Runs are registered and finished by the
server, so the registry has one writer.

Without a live server, `copula run` starts the same server embedded for the duration of its
command; a second embedded launcher in the project then fails with a hint to start one.

Model server (one of): --upstream URL (default: $OPENAI_BASE_URL), --mock, --manager SOCK.
"""
import argparse
import asyncio
import json
import os
import secrets
import shutil
import signal
import socket
import sqlite3
import sys
import tempfile
import time
import urllib.error
import urllib.request

from . import upstream as up
from .calls import CallLog
from .registry import Registry

DISCOVERY = 'server.json'
BUSY = ('copula: another Copula process holds this project\'s store (.copula/entries.db). To run '
        'several commands at once, start one server in the project:  copula server  -- then '
        'every copula run connects to it.')
NO_BACKEND = ('copula: no model server. Set OPENAI_BASE_URL (the server your program already '
              'uses), or pass --upstream URL, or --mock for a trial with the mock model.')


def add_backend_options(p):
    g = p.add_argument_group('model server')
    g.add_argument('--upstream', help='OpenAI-compatible base URL to serve from '
                   '(default: $OPENAI_BASE_URL of this process)')
    g.add_argument('--served-model', help='the default model of the upstream')
    g.add_argument('--model-dir', help='files of the default model (identity, sampling defaults)')
    g.add_argument('--defaults', help='pinned sampling table: JSON or @FILE')
    g.add_argument('--identity-extra', help='serving settings in the identity: JSON or @FILE')
    g.add_argument('--engine-seed', choices=('auto', 'address', 'off'), default='auto',
                   help='engine seed per sample address (auto: on for vLLM)')
    g.add_argument('--api-key-env', help='environment variable with the upstream key '
                   '(default OPENAI_API_KEY)')
    g.add_argument('--model', action='append', default=[], metavar='NAME[=SERVED]',
                   help="client-side model names; '*' takes any name")
    g.add_argument('--mock', action='store_true', help='serve the mock model (trials, tests)')
    g.add_argument('--mock-nonce', default='copula-mock',
                   help='another nonce gives the mock model other replies')
    g.add_argument('--mock-latency', type=float, default=0.0, help='mock seconds per reply')
    g.add_argument('--mock-script', help='FILE.py:FUNC or MODULE:FUNC scripting the mock replies')
    g.add_argument('--manager', help='socket of a running kernel manager')
    g.add_argument('--no-identity', action='store_true',
                   help='treat the upstream identity as not established (W12)')
    g.add_argument('--keepalive', type=float, default=15.0,
                   help='seconds between keep-alives of a buffered stream')
    g.add_argument('--max-tokens', type=int, default=1024,
                   help='max_tokens of requests that set none')
    g.add_argument('--timeout', type=float, default=900.0, help='upstream request timeout, s')
    g.add_argument('--workers', type=int, default=64, help='upstream requests in flight')
    g.add_argument('--inflight', type=int, default=64, help='generations in flight per model')
    g.add_argument('--wait-ready', type=float, default=0.0, help='wait for the upstream, s')
    g.add_argument('--max-attempts', type=int, default=6,
                   help='attempts of a generation the upstream fails (transport errors, timeouts, '
                   '5xx, 408, 409, 429, 401/403/404/405/407)')
    g.add_argument('--backoff', type=float, default=0.5, help='base retry backoff, s')
    g.add_argument('--trace', action='store_true', help='write the kernel trace .copula/trace.jsonl')


def explicit_backend(a):
    return bool(a.upstream or a.mock or a.manager or a.mock_script)


def backend_overrides(a):
    """The model-server options set to something other than their default: a live server
    decides them, so a launcher that connects to one must not set them."""
    p = argparse.ArgumentParser()
    add_backend_options(p)
    defaults = vars(p.parse_args([]))
    return sorted('--' + k.replace('_', '-') for k, v in defaults.items()
                  if getattr(a, k, v) != v)


def private_state(state):
    """.copula/ holds the run-token secret, every prompt and every sample: only its owner may
    read it."""
    os.makedirs(state, mode=0o700, exist_ok=True)
    try:
        if os.stat(state).st_uid == os.getuid():
            os.chmod(state, 0o700)
    except OSError:
        pass


def _sock_path(state):
    p = os.path.join(state, 'm.sock')
    if len(p.encode()) < 100:
        return p, None
    d = tempfile.mkdtemp(prefix='copula-')
    return os.path.join(d, 'm.sock'), d


def _listen(addr):
    if not addr:
        return '127.0.0.1', 0
    if addr.startswith('['):                    # [::1]:8000
        host, _, rest = addr[1:].partition(']')
        return host, int(rest[1:]) if rest.startswith(':') else 0
    if addr.count(':') > 1:                     # a bare IPv6 address
        return addr, 0
    host, sep, port = addr.rpartition(':')
    if not sep:
        return addr, 0
    return host or '0.0.0.0', int(port)


class Server:
    """A server in this process: embedded (one command) or `copula server` (discover=True)."""

    def __init__(self, project, opts, environ=None, discover=False, listen=None):
        self.project = os.path.realpath(project)
        self.state = os.path.join(self.project, '.copula')
        self.opts = opts
        self.env = dict(os.environ if environ is None else environ)
        self.discover = discover
        self.host, self.port = _listen(listen)
        self.token = secrets.token_hex(16)
        self.notes = []
        self.proxy = None
        self._close = []

    async def start(self):
        private_state(self.state)
        try:
            await self._start()
        except BaseException:
            await self.close()
            raise
        return self

    async def _start(self):
        a = self.opts
        from .kernel.sdk import Session
        from .proxy import Proxy
        if a.manager:
            sock = a.manager
            names = {n.partition('=')[0]: 'actor' for n in (a.model or ['actor'])}
            identity, info = True, {'kind': 'manager', 'socket': sock}
        else:
            sock, info, names, identity = await self._manager()
        self.reg = Registry(os.path.join(self.state, 'registry.db'))
        self._close.append(self.reg.close)
        self.calls = CallLog(os.path.join(self.state, 'calls.db'))
        self._close.append(self.calls.close)
        session = await Session.connect(
            sock, trace=os.path.join(self.state, 'trace.jsonl') if a.trace else None)
        self._close.append(session.aclose)
        if a.no_identity:
            identity = False
        default = next((n for n, k in names.items() if k == 'actor' and n != '*'), None)
        self.info = {'upstream': info, 'notes': self.notes, 'project': self.project,
                     'default_model': default}
        self.proxy = Proxy(session, self.reg, names, identity=identity, keepalive=a.keepalive,
                           default_max_tokens=a.max_tokens, calls=self.calls,
                           control_token=self.token, info=self.info)
        await self.proxy.start(self.host, self.port)
        self._close.append(self.proxy.close)
        if self.host in ('0.0.0.0', '::', ''):
            self.proxy.base = 'http://%s:%d' % (socket.gethostname(),
                                                self.proxy.server.sockets[0].getsockname()[1])
        if self.discover:
            self._write_discovery()

    async def _manager(self):
        a = self.opts
        from .kernel.manager.flights import Manager
        from .kernel.manager.server import serve
        from .kernel.store.entries import EntryStore, StoreError
        try:
            store = EntryStore(os.path.join(self.state, 'entries.db'), batched=True)
        except (StoreError, sqlite3.OperationalError) as err:
            if 'locked' in str(err):
                raise SystemExit(BUSY)
            raise
        self._close.append(store.close)
        if a.mock or a.mock_script:
            b = up.mock(a)
        else:
            url = a.upstream or self.env.get('OPENAI_BASE_URL')
            if not url:
                raise SystemExit(NO_BACKEND)
            a.upstream = url
            try:
                b = up.build(a, store.secret, self.env)
            except up.UpstreamError as err:
                raise SystemExit('copula: %s' % err)
        self.notes += b['notes']
        clients = [b['actor']] + list(b['models'].values())
        for c in clients:
            self._close.append(c.aclose)
        manager = Manager(store, b['actor'], inflight_cap=a.inflight, models=b['models'],
                          max_attempts=a.max_attempts, backoff_base=a.backoff,
                          backoff_cap=max(a.backoff, 30.0))
        await manager.start()

        async def stop_manager():
            await manager.drain()
            await manager.stop()
        self._close.append(stop_manager)
        sock, tmpdir = _sock_path(self.state)
        if tmpdir:
            self._close.append(lambda: shutil.rmtree(tmpdir, ignore_errors=True))
        if os.path.exists(sock):
            os.unlink(sock)
        ksrv = await serve(manager, sock)

        async def stop_ksrv():
            ksrv.close()
            await ksrv.wait_closed()
            if os.path.exists(sock):
                os.unlink(sock)
        self._close.append(stop_ksrv)
        info = dict(b['describe'], engine_seed=b['engine_seed'])
        info.pop('models', None)
        info['models'] = sorted(b['names'])
        return sock, info, b['names'], b['identity']

    def _write_discovery(self):
        path = os.path.join(self.state, DISCOVERY)
        data = {'url': self.proxy.base, 'host': socket.gethostname(), 'pid': os.getpid(),
                'token': self.token, 'started': time.time(), 'upstream': self.info['upstream']}
        tmp = '%s.%d' % (path, os.getpid())
        if os.path.lexists(tmp):
            os.unlink(tmp)
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0),
                     0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, sort_keys=True)
        os.replace(tmp, path)

        def remove():
            try:
                with open(path) as f:
                    if json.load(f).get('token') == self.token:
                        os.unlink(path)
            except (OSError, ValueError):
                pass
        self._close.append(remove)

    # the launcher's interface (Remote has the same)
    async def paths(self, spec):
        return self.proxy.version_paths(spec)

    async def register(self, spec):
        return self.proxy.register_run(spec)

    async def end_run(self, run_id, reads, ok):
        return await self.proxy.finish_run(run_id, reads, ok)

    def describe(self):
        return 'embedded server (%s)' % _upstream_text(self.info['upstream'])

    async def close(self):
        while self._close:
            fn = self._close.pop()
            try:
                res = fn()
                if asyncio.iscoroutine(res):
                    await res
            except Exception as err:            # noqa: BLE001 -- close the rest anyway
                sys.stderr.write('copula: cleanup: %r\n' % (err,))


def _upstream_text(info):
    info = info or {}
    kind = info.get('kind', '?')
    if kind == 'mock':
        return 'mock model'
    if kind == 'manager':
        return 'manager %s' % info.get('socket')
    return '%s %s%s' % (kind, info.get('base', ''),
                        ', engine seed per address' if info.get('engine_seed') == 'address' else '')


class Remote:
    """A live server reached over HTTP (same host via .copula/server.json, or --server)."""

    def __init__(self, url, token, info=None):
        self.url = url.rstrip('/')
        self.token = token
        self.info = info or {}

    def _call(self, method, path, data=None, timeout=60.0):
        body = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(self.url + path, data=body, method=method, headers={
            'Authorization': 'Bearer %s' % self.token, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as handle:
                return json.loads(handle.read().decode('utf-8'))
        except urllib.error.HTTPError as err:
            try:
                detail = json.loads(err.read().decode('utf-8'))['error']['message']
            except Exception:                   # noqa: BLE001
                detail = str(err)
            raise SystemExit('copula: the server refused %s %s: %s' % (method, path, detail))

    async def _acall(self, *args, **kw):
        return await asyncio.get_running_loop().run_in_executor(None, lambda: self._call(*args,
                                                                                         **kw))

    def health(self, timeout=3.0):
        return self._call('GET', '/copula/v1/health', timeout=timeout)

    async def paths(self, spec):
        return (await self._acall('POST', '/copula/v1/paths', spec))['paths']

    async def register(self, spec):
        out = await self._acall('POST', '/copula/v1/runs', spec)
        return out['run'], out['url']

    async def end_run(self, run_id, reads, ok):
        return await self._acall('POST', '/copula/v1/runs/%s/end' % run_id,
                                 {'reads': reads, 'ok': ok}, timeout=600.0)

    def describe(self):
        return 'server %s (%s)' % (self.url, _upstream_text(self.info.get('upstream')))

    async def close(self):
        pass


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def discover(state):
    """The live server of the project, or None (a stale discovery file of a dead server on this
    host is removed)."""
    path = os.path.join(state, DISCOVERY)
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    try:
        remote = Remote(data['url'], data['token'])
    except (KeyError, TypeError, AttributeError):
        return None
    try:
        remote.info = remote.health()
        return remote
    except (SystemExit, OSError, ValueError, urllib.error.URLError):
        if data.get('host') == socket.gethostname() and not _alive(int(data.get('pid', 0))):
            try:
                os.unlink(path)
            except OSError:
                pass
        return None


async def serve_forever(a):
    srv = Server(a.project, a, discover=True, listen=a.listen)
    existing = discover(srv.state)
    if existing is not None:
        raise SystemExit('copula: a server already runs for this project: %s' % existing.url)
    await srv.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    sys.stderr.write('copula: server %s for %s (%s)\n' % (srv.proxy.base, srv.project,
                                                         _upstream_text(srv.info['upstream'])))
    for note in srv.notes:
        sys.stderr.write('copula:   note: %s\n' % note)
    sys.stderr.flush()
    print(json.dumps({'event': 'ready', 'url': srv.proxy.base, 'pid': os.getpid()}), flush=True)
    try:
        await stop.wait()
    finally:
        await srv.close()
    return 0
