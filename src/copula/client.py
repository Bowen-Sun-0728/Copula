"""In-process helpers for code running under `copula run` (DESIGN.md §10.3; preliminary).

    import copula
    with copula.run(variant='strict', case=ex['id']):     # one unit (§4.1)
        r = copula.chat([{'role': 'user', 'content': q}], temperature=0.7)
    with copula.decision():                               # a decision scope (§4.2)
        ...
    with copula.holdout():                                # units opened inside are holdout units
        ...

The context travels as X-Copula-* headers. `headers()` gives them for any client; `copula.attach()`
(attach.py) adds them to every request to the Copula server made through httpx, httpx2 or
http.client, and copies the context into worker threads. Standard library only.

Context fields: `coords` (inherited coordinates), `unit` (a unit is open: its coordinates are sent),
`ordinal`, `decision`, `role`, `phase`, and adapter flags.
"""
import contextlib
import contextvars
import json
import os
import threading
import urllib.parse
import urllib.request
import uuid

_ctx = contextvars.ContextVar('copula_ctx', default=None)
_ordinals = {}
_lock = threading.Lock()


def current():
    return _ctx.get() or {}


def _push(**fields):
    """Set context fields; returns the token for _pop."""
    return _ctx.set(dict(current(), **fields))


def _pop(token):
    _ctx.reset(token)


def _open_unit(coords, **fields):
    """Open a unit with `coords` merged over the inherited ones; returns a token. The k-th unit
    with the same coordinates in this process gets ordinal k, so it reads new samples; a resumed
    process counts again from 1 and re-reads its units."""
    cur = current()
    merged = dict(cur.get('coords', {}))
    merged.update(coords)
    key = json.dumps([merged, cur.get('phase')], sort_keys=True, default=str)
    with _lock:
        _ordinals[key] = _ordinals.get(key, 0) + 1
        ordinal = _ordinals[key]
    return _ctx.set(dict(cur, coords=merged, unit=True, ordinal=ordinal, **fields))


@contextlib.contextmanager
def run(**coords):
    """A unit with these coordinates (merged over the enclosing context's)."""
    token = _open_unit(coords)
    try:
        yield
    finally:
        _pop(token)


@contextlib.contextmanager
def decision(name=None):
    """A decision scope: the units opened inside read one slot when they can (§4.2)."""
    token = _push(decision=name or 'd-' + uuid.uuid4().hex[:16])
    try:
        yield
    finally:
        _pop(token)


@contextlib.contextmanager
def role(name):
    """Calls made inside carry this role (routed by the plan's routes, §4.4)."""
    token = _push(role=name)
    try:
        yield
    finally:
        _pop(token)


@contextlib.contextmanager
def holdout():
    """Units opened inside are holdout units: a new group space, cases must be new (§4.2)."""
    token = _push(phase='holdout')
    try:
        yield
    finally:
        _pop(token)


def headers(role=None):
    """The X-Copula headers of the current context (values percent-encoded: ASCII only)."""
    q = urllib.parse.quote
    cur = current()
    out = {}
    if cur.get('unit') and cur.get('coords'):
        out['X-Copula-Coords'] = ';'.join(
            '%s=%s' % (q(str(k)), q(str(v))) for k, v in sorted(cur['coords'].items()))
        out['X-Copula-Unit'] = str(cur.get('ordinal', 1))
    if cur.get('phase'):
        out['X-Copula-Phase'] = cur['phase']           # without a unit: one unit of that phase
        if cur.get('phase_inferred'):
            out['X-Copula-Phase-Source'] = 'inferred'
    if cur.get('decision'):
        out['X-Copula-Decision'] = q(str(cur['decision']), safe='')
    role = role or cur.get('role')
    if role:
        out['X-Copula-Role'] = q(str(role), safe='')
    return out


def chat(messages, model=None, base_url=None, role=None, timeout=600, extra_headers=None,
         **params):
    """A chat completion through the Copula server named by OPENAI_BASE_URL. Returns the
    response dict, with the x-copula response headers under `_copula`."""
    base = base_url or os.environ['OPENAI_BASE_URL']
    body = dict(params, model=model or os.environ.get('COPULA_MODEL'), messages=messages)
    hdrs = {'Content-Type': 'application/json',
            'Authorization': 'Bearer %s' % os.environ.get('OPENAI_API_KEY', 'copula')}
    hdrs.update(headers(role))
    hdrs.update(extra_headers or {})
    req = urllib.request.Request(base.rstrip('/') + '/chat/completions',
                                 data=json.dumps(body).encode(), headers=hdrs, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        data['_copula'] = {k.lower(): v for k, v in resp.headers.items()
                           if k.lower().startswith('x-copula')}
    return data
