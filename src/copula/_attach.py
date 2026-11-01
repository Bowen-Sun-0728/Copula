"""`copula.attach()`: framework integration inside the agent's process (DESIGN.md §10.3; O5, O8).

Under `copula run --exp E` the launcher's sitecustomize calls attach() before the program starts,
so a Python program needs no change. attach():

1. copies the caller's context into worker threads (ThreadPoolExecutor.submit, which asyncio's
   run_in_executor also uses), as OpenTelemetry's threading instrumentation does;
2. adds the context's X-Copula-* headers (client.headers) to every request sent to the Copula
   server through httpx, httpx2 or http.client (urllib, urllib3, requests); requests to any other
   host are not touched;
3. installs framework adapters when the framework is imported (now or later):
       dspy   units from module calls, candidates and seeds from optimizers, holdout evaluations,
              its response cache turned off (adapters/dspy_adapter.py)
       gepa   one decision scope per proposal, proposer role for reflection (adapters/gepa_adapter.py)
4. reports what it did once, on stderr ("copula: attach: ...").

Standard library only; frameworks are imported by the program, never by attach.
"""
import concurrent.futures
import contextvars
import functools
import importlib.abc
import os
import sys
import threading
import urllib.parse

from . import client

_state = {'attached': False, 'base': None, 'host': None, 'port': None, 'notes': [], 'finder': None}
_lock = threading.Lock()


def note(msg):
    """Report one attach decision (deduplicated)."""
    with _lock:
        if msg in _state['notes']:
            return
        _state['notes'].append(msg)
    if not os.environ.get('COPULA_QUIET'):
        sys.stderr.write('copula: attach: %s\n' % msg)
        sys.stderr.flush()


def notes():
    return list(_state['notes'])


def attach(exp=None, url=None):
    """Attach this process to its Copula run (idempotent). `exp` must agree with the launcher's
    --exp; `url` defaults to $COPULA_URL, which `copula run` sets."""
    url = url or os.environ.get('COPULA_URL')
    if not url:
        raise RuntimeError('copula.attach() needs a Copula server: run the program under '
                           '`copula run --exp NAME -- ...`')
    launched = os.environ.get('COPULA_EXP') or None
    if exp is not None and exp != launched:
        raise ValueError('attach(exp=%r) disagrees with the launcher (--exp %r); the experiment '
                         'is chosen by `copula run --exp`' % (exp, launched))
    with _lock:
        if _state['attached']:
            return
        _state['attached'] = True
    u = urllib.parse.urlsplit(url)
    _state.update(base='%s://%s/' % (u.scheme, u.netloc), host=u.hostname,
                  port=u.port or (443 if u.scheme == 'https' else 80))
    _install_threads()
    _install_http_client()
    finder = _PostImport()
    _state['finder'] = finder
    for name in ('httpx', 'httpx2'):
        finder.on(name, _patch_httpx)
    from .adapters import dspy_adapter, gepa_adapter
    finder.on('dspy', dspy_adapter.install)
    finder.on('gepa', gepa_adapter.install)
    sys.meta_path.insert(0, finder)
    note('attached to %s (threads copy context; headers on httpx, httpx2, http.client)'
         % _state['base'])


# -- threads ----------------------------------------------------------------------------------

def _install_threads():
    cls = concurrent.futures.ThreadPoolExecutor
    if getattr(cls.submit, '_copula', False):
        return
    orig = cls.submit

    @functools.wraps(orig)
    def submit(self, fn, /, *args, **kwargs):
        return orig(self, contextvars.copy_context().run, fn, *args, **kwargs)
    submit._copula = True
    cls.submit = submit


# -- headers ----------------------------------------------------------------------------------

def _ours(url):
    return _state['base'] is not None and str(url).startswith(_state['base'])


def _patch_httpx(mod):
    def add(request):
        if _ours(request.url):
            for k, v in client.headers().items():
                if k not in request.headers:
                    request.headers[k] = v
    for cls_name, is_async in (('Client', False), ('AsyncClient', True)):
        cls = getattr(mod, cls_name, None)
        if cls is None or getattr(cls.send, '_copula', False):
            continue
        orig = cls.send
        if is_async:
            async def send(self, request, *a, _orig=orig, **kw):
                add(request)
                return await _orig(self, request, *a, **kw)
        else:
            def send(self, request, *a, _orig=orig, **kw):
                add(request)
                return _orig(self, request, *a, **kw)
        functools.update_wrapper(send, orig)
        send._copula = True
        cls.send = send


def _install_http_client():
    import http.client
    cls = http.client.HTTPConnection
    if getattr(cls.endheaders, '_copula', False):
        return
    orig = cls.endheaders

    @functools.wraps(orig)
    def endheaders(self, message_body=None, **kw):
        if self.host == _state['host'] and self.port == _state['port']:
            buf = getattr(self, '_buffer', [])
            if not any(line[:9].lower() == b'x-copula-' for line in buf):
                for k, v in client.headers().items():
                    self.putheader(k, v)
        return orig(self, message_body, **kw)
    endheaders._copula = True
    cls.endheaders = endheaders


# -- post-import hooks ------------------------------------------------------------------------

class _PostImport(importlib.abc.MetaPathFinder):
    """Runs hooks right after a top-level module is executed (or now, if it is imported).

    The hooks of a name are taken only when its module has executed successfully, so a
    find_spec probe or a failed import keeps them; a nested lookup of the same name (a finder
    after this one that calls find_spec itself) is passed through; and the module's loader is
    wrapped, never changed, since a loader may serve other modules too."""

    def __init__(self):
        self.hooks = {}
        self._busy = set()
        self._lock = threading.Lock()

    def on(self, name, fn):
        mod = sys.modules.get(name)
        if mod is not None:
            _run(fn, mod)
        else:
            self.hooks.setdefault(name, []).append(fn)

    def find_spec(self, name, path=None, target=None):
        with self._lock:
            if name not in self.hooks or name in self._busy:
                return None
            self._busy.add(name)
        try:
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, 'find_spec'):
                    continue
                spec = finder.find_spec(name, path, target)
                if spec is not None:
                    break
            else:
                return None
        finally:
            with self._lock:
                self._busy.discard(name)
        if spec.loader is None or not hasattr(spec.loader, 'exec_module'):
            return spec
        spec.loader = _HookedLoader(spec.loader, name, self)
        return spec


class _HookedLoader(importlib.abc.Loader):
    """A loader that runs the finder's hooks for `name` after the real loader's exec_module."""

    def __init__(self, loader, name, finder):
        self._loader, self._name, self._finder = loader, name, finder

    def create_module(self, spec):
        create = getattr(self._loader, 'create_module', None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self._loader.exec_module(module)
        with self._finder._lock:
            hooks = self._finder.hooks.pop(self._name, [])
        for fn in hooks:
            _run(fn, module)

    def __getattr__(self, attr):
        return getattr(self._loader, attr)


def _run(fn, mod):
    try:
        fn(mod)
    except Exception as err:                    # noqa: BLE001 -- never break the program
        note('adapter %s failed for %s: %r (the program runs without it)'
             % (getattr(fn, '__module__', fn), getattr(mod, '__name__', mod), err))
