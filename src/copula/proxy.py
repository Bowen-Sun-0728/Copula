"""The model endpoint of a Copula server (DESIGN.md §10.5; preliminary).

    POST <base>[/r/<token>]/v1/chat/completions    OpenAI Chat Completions
    POST <base>[/r/<token>]/v1/responses           OpenAI Responses (protocols.ResponsesProto)
    GET  <base>[/r/<token>]/v1/responses/<id>      a response this server issued
    POST <base>[/r/<token>]/v1/messages            Anthropic Messages (protocols.AnthropicProto)
    GET  <base>[/r/<token>]/v1/models
    GET  <base>/copula/v1/health                    control (Authorization: Bearer <server token>)
    POST <base>/copula/v1/runs                      register a process run and serve it:
                                                    {experiment, recipe, phase, coords, overrides,
                                                    routes} -> {run, url}
    POST <base>/copula/v1/runs/<run>/end            end it: {reads, ok} -> {stats, notes, units,
                                                    counts, why}

The run is named by X-Copula-Run: <token>, or by the path token for clients without header
options (§10.2). Request headers (all optional):
    X-Copula-Coords: k=v;k=v    a unit of the run with these coordinates (§4.1)
    X-Copula-Unit: <n>          the unit's ordinal among units with the same coordinates
    X-Copula-Phase: holdout     a holdout unit (§4.2); without coordinates, one holdout unit
    X-Copula-Phase-Source: inferred   an adapter's guess: falls back to search on case overlap
    X-Copula-Decision: <id>     a decision scope (§4.2)
    X-Copula-Role: <role>       routed by the plan's routes (§4.4)
    x-stainless-retry-count, Idempotency-Key: retry signals (§3.2 item 3)

Response headers: x-copula-status, x-copula-rank, x-copula-group, x-copula-run, x-copula-rule.
Streams are buffered (change V3): keep-alives while the sample is generated (an SSE comment, or
Anthropic's ping event), then the protocol's events; the x-copula fields come as a final SSE
comment.

A proxy serves only the process runs it was given (`url`), and refuses a run once it has ended:
a new attempt of an ended run would re-read samples its client already received (C1).

Retries (§3.2 item 3): a request carrying a retry signal re-reads an occurrence of the same request
in the same run whose response was not delivered and whose client connection is gone; any other
request is a new occurrence. An occurrence is forgotten once a response (sample or error) was
written to its client.

Every request becomes one chat call of an automatically registered node (attrs.node_name) in a
run of the kernel SDK session, made without context variables, as the SDK's own HTTP endpoint
does. Each answered occurrence is written to the call log (calls.CallLog) when there is one.
"""
import asyncio
import collections
import hmac
import json
import os
import re
import time
import urllib.parse

from .kernel.errors import EvalrtError
from .kernel.sdk.session import Group, SDKError

from . import protocols
from .attrs import infer, node_name, node_spec
from .coords import PHASES
from .registry import HoldoutOverlap, check_route, h, seed_group

PATH = re.compile(r'^(?:/r/([^/]+))?/v1/(chat/completions|models|responses|messages|'
                  r'responses/[A-Za-z0-9_-]+)$')
RETRY_TTL = 600.0
SWEEP_EVERY = 256


class Occurrence:
    """One served occurrence of a request in a run, with the connection that owns its delivery."""

    def __init__(self, future, reader, writer, rkey, idem_key):
        self.future = future
        self.reader = reader
        self.writer = writer
        self.rkey = rkey
        self.idem_key = idem_key
        self.delivered = False
        self.created = time.time()

    def dead(self):
        return self.writer.is_closing() or self.reader.at_eof()

    def failed(self):
        return self.future.done() and (self.future.cancelled() or
                                       self.future.exception() is not None)


class HTTPError(Exception):
    def __init__(self, status, kind, detail):
        super().__init__(detail)
        self.status, self.kind, self.detail = status, kind, detail


class Proxy:
    def __init__(self, session, registry, models, identity=True, keepalive=15.0, routes=None,
                 default_max_tokens=1024, calls=None, control_token=None, info=None):
        """`models`: the model names clients may send -> the session backend serving them
        ('actor' or a named backend; the name '*' takes any other name). `identity`: whether the
        upstream identity is established (a manager-started backend: True), per backend name or
        for all. `routes`: role -> rule, for runs started without their own. `calls`: a CallLog.
        `control_token`: the bearer token of the /copula/v1 control endpoints (None: none)."""
        self.s = session
        self.reg = registry
        self.models = dict(models)
        self.identity = identity
        self.keepalive = keepalive
        self.routes = {role: check_route(rule) for role, rule in (routes or {}).items()}
        self.default_max_tokens = default_max_tokens
        self.calls = calls
        self.control_token = control_token
        self.info = dict(info or {})
        self.run_routes = {}            # process run -> its plan's routes
        self.allowed = set()            # process runs this proxy serves
        self.ended = set()              # runs whose attempt ended: refused from then on
        self.closing = False
        self.runs = {}                  # run id -> open kernel Run
        self.parent_of = {}             # open unit run id -> its process run
        self._opening = {}              # run id -> future of its opening
        self.stats = collections.defaultdict(collections.Counter)   # process run -> statuses
        self.notes = collections.defaultdict(collections.Counter)   # process run -> notes
        self._occ = collections.defaultdict(list)   # request key -> undelivered occurrences
        self._idem = {}                 # (run, idempotency key) -> occurrence
        self._writers = set()
        self._requests = 0
        self.server = None
        self.base = None

    # -- lifecycle ---------------------------------------------------------------------------

    async def start(self, host='127.0.0.1', port=0):
        self.server = await asyncio.start_server(self._conn, host, port)
        h_, p = self.server.sockets[0].getsockname()[:2]
        self.base = 'http://%s:%d' % (h_, p)
        return self.base

    def url(self, rec, routes=None):
        """The base URL of process run `rec`; from now on this proxy serves it, with `routes`
        (role -> rule) when given."""
        if routes is not None:
            self.run_routes[rec['run_id']] = {r: check_route(v) for r, v in routes.items()}
        self.allowed.add(rec['run_id'])
        return '%s/r/%s/v1' % (self.base, self.reg.token(rec['run_id']))

    def version_paths(self, spec):
        """The files a launcher digests before registering (registry.version_paths)."""
        if not spec.get('experiment'):
            return []
        from .coords import Classes
        compare = Classes(spec.get('recipe') or 'dev', spec.get('overrides') or {}).split(
            spec.get('coords') or {})['compare']
        return self.reg.version_paths(spec['experiment'], compare)

    def register_run(self, spec):
        """Register a launched process as one run (registry.process_run) and serve it:
        (record, url). `spec`: experiment, recipe, phase, coords, overrides, routes, owner
        {host, pid}, and digests {path: digest or None} taken by the launcher (a path it did
        not digest counts as unknown, the safe direction)."""
        digests = spec.get('digests')
        kw = {} if digests is None else {'digest_now': lambda p: digests.get(p)}
        rec = self.reg.process_run(spec.get('experiment'), spec.get('recipe') or 'dev',
                                   spec.get('phase') or 'search', spec.get('coords') or {},
                                   spec.get('overrides') or {}, owner=spec.get('owner'), **kw)
        return rec, self.url(rec, spec.get('routes') or None)

    async def finish_run(self, run_id, reads=None, ok=True):
        """End a process run and record its read set: {stats, notes, units, counts, why}."""
        res = await self.end_run(run_id)
        self.reg.finish(run_id, reads or {}, bool(ok))
        res['units'] = self.reg.count_units(run_id)
        if self.calls is not None:
            from .calls import run_summary
            counts, top = await asyncio.get_running_loop().run_in_executor(
                None, run_summary, self.calls.path, self.reg.path, run_id)
            res['counts'], res['why'] = counts, top
        return res

    async def end_run(self, run_id):
        """End a process run: {stats, notes} of its calls (forgotten here afterwards)."""
        await self.end([run_id])
        if self.calls is not None:
            self.calls.flush()
        self.run_routes.pop(run_id, None)
        self.allowed.discard(run_id)
        return {'stats': dict(self.stats.pop(run_id, {})),
                'notes': dict(self.notes.pop(run_id, {}))}

    async def end(self, run_ids=None):
        """End the attempts of the given runs and of their units (all when None)."""
        ids = set(self.runs if run_ids is None else run_ids)
        if run_ids is not None:
            ids |= {rid for rid in self.runs if self.parent_of.get(rid) in ids}
        self.ended |= ids
        for rid in list(ids):
            r = self.runs.pop(rid, None)
            self.parent_of.pop(rid, None)
            if r is not None and not r.closed:
                await self.s._end(r)
        for key in [k for k in self._occ if k[0] in ids]:
            for occ in self._occ.pop(key):
                self._idem.pop(occ.idem_key, None)

    async def close(self):
        self.closing = True
        if self.server is not None:
            self.server.close()
        await self.end()
        if self.calls is not None:
            self.calls.flush()
        for w in list(self._writers):
            try:
                w.close()
            except Exception:                   # noqa: BLE001
                pass
        if self.server is not None:
            await self.server.wait_closed()

    def ensure_node(self, backend, attrs):
        name = node_name(backend, attrs)
        if name not in self.s.llm:
            self.s.register_llm(name, max_tokens=self.default_max_tokens,
                                **node_spec(backend, attrs))
        return name

    async def open_run(self, rec):
        """The open evalrt run of registry record `rec` (opened once, without context vars)."""
        rid = rec['run_id']
        if rid in self.ended:
            raise HTTPError(410, 'run_ended', 'run %s has ended' % rid)
        if rid in self.runs:
            return self.runs[rid]
        fut = self._opening.get(rid)
        if fut is None:
            fut = self._opening[rid] = asyncio.ensure_future(self._open(rec))
        try:
            return await asyncio.shield(fut)
        finally:
            if fut.done():
                self._opening.pop(rid, None)

    async def _open(self, rec):
        group = Group(rec['grp'])
        r = self.s._new_run(group, rec['run_id'], rec['rule'])
        await r.ensure(group)
        self.s._open[r.id] = r
        self.s._row({'ev': 'run_start', 'run': r.id, 'attempt': r.attempt, 'group': group.id,
                     'label': rec['rule'], 'copula': {k: rec[k] for k in (
                         'experiment', 'phase', 'granularity', 'coords', 'version', 'slot')}})
        if rec['run_id'] in self.ended or (rec['parent'] and rec['parent'] in self.ended):
            await self.s._end(r)                # its run ended while it was opening
            self.ended.add(rec['run_id'])
            raise HTTPError(410, 'run_ended', 'run %s has ended' % rec['run_id'])
        self.runs[r.id] = r
        if rec['parent']:
            self.parent_of[r.id] = rec['parent']
        return r

    # -- HTTP --------------------------------------------------------------------------------

    async def _conn(self, reader, writer):
        self._writers.add(writer)
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
                    hl = await reader.readline()
                    if hl in (b'\r\n', b'\n', b''):
                        break
                    k, _, v = hl.decode('latin-1').partition(':')
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
                                pass
                            break
                        body += await reader.readexactly(size)
                        await reader.readline()
                else:
                    body = await reader.readexactly(int(headers.get('content-length') or 0))
                keep = version == 'HTTP/1.1' and headers.get('connection', '').lower() != 'close'
                ok = await self._request(method, path, headers, body, reader, writer, keep)
                if not ok or not keep:
                    break
        except (asyncio.IncompleteReadError, ConnectionError, ValueError):
            pass
        finally:
            self._writers.discard(writer)
            try:
                writer.close()
            except Exception:                   # noqa: BLE001
                pass

    async def _send(self, writer, status, payload, keep, extra=None, kind='application/json'):
        head = ['HTTP/1.1 %d %s' % (status, 'OK' if status == 200 else 'Error'),
                'Content-Type: %s' % kind, 'Content-Length: %d' % len(payload),
                'Connection: %s' % ('keep-alive' if keep else 'close')]
        if status != 200:
            head.append('x-should-retry: false')
        head += ['%s: %s' % (k, _hval(v)) for k, v in (extra or {}).items()]
        writer.write(('\r\n'.join(head) + '\r\n\r\n').encode('latin-1') + payload)
        await writer.drain()

    @staticmethod
    def _error_body(kind, detail):
        return json.dumps({'error': {'message': str(detail), 'type': kind, 'code': kind,
                                     'param': None}}).encode()

    async def _request(self, method, path, headers, body, reader, writer, keep):
        """Answer one request; False when the connection is unusable afterwards."""
        m = PATH.match(path.split('?', 1)[0])
        kind = m.group(2) if m else None

        async def fail(status, code, detail):
            body = json.dumps(protocols.error_for(path, status, code, detail)).encode()
            await self._send(writer, status, body, keep)
            return True
        try:
            if path.startswith('/copula/v1/'):
                return await self._control(method, path, headers, body, writer, keep)
            if self.closing:
                raise HTTPError(503, 'closing', 'the Copula server is shutting down')
            if m is None:
                raise HTTPError(404, 'not_found', 'no such endpoint: %s' % path)
            token = headers.get('x-copula-run') or m.group(1)      # headers before the path
            rec = self.reg.run_of(token) if token else None
            if rec is None or rec['run_id'] not in self.allowed:
                raise HTTPError(401, 'no_run', 'no valid Copula run token for this server')
            if rec['run_id'] in self.ended or rec['status'] != 'running':
                raise HTTPError(410, 'run_ended', 'run %s has ended' % rec['run_id'])
            if kind == 'models':
                data = {'object': 'list', 'data': [{'id': n, 'object': 'model',
                                                    'owned_by': 'copula'} for n in self.models]}
                await self._send(writer, 200, json.dumps(data).encode(), keep)
                return True
            if kind.startswith('responses/'):
                if method != 'GET':
                    raise HTTPError(405, 'method', 'GET a stored response')
                resp = self.reg.response(kind.split('/', 1)[1])
                if resp is None or resp.get('store') is False:
                    raise HTTPError(404, 'not_found', 'no response %s on this server' % kind)
                await self._send(writer, 200, json.dumps(resp).encode(), keep)
                return True
            if method != 'POST':
                raise HTTPError(405, 'method', 'POST a request')
            req = json.loads(body.decode('utf-8'))
            if not isinstance(req, dict):
                raise HTTPError(400, 'invalid_call', 'the request body is a JSON object')
            proto = protocols.make(kind, req, self.reg)
            return await self._chat(rec, proto, headers, reader, writer, keep)
        except HTTPError as err:
            return await fail(err.status, err.kind, err.detail)
        except protocols.ProtoError as err:
            return await fail(err.status, err.kind, err.detail)
        except SDKError as err:
            return await fail(400, err.kind, err.detail)
        except EvalrtError as err:
            return await fail(502, err.kind, err.detail)
        except (ValueError, TypeError) as err:
            return await fail(400, 'invalid_call', err)
        except (ConnectionError, asyncio.IncompleteReadError):
            return False
        except Exception as err:                # noqa: BLE001 -- always answer
            try:
                return await fail(500, 'internal', repr(err))
            except (ConnectionError, RuntimeError):
                return False

    async def _control(self, method, path, headers, body, writer, keep):
        """The control endpoints of `copula run` against a running server."""
        auth = headers.get('authorization', '')
        if not self.control_token or not hmac.compare_digest(
                auth.encode(), ('Bearer ' + self.control_token).encode()):
            raise HTTPError(401, 'unauthorized', 'a control call needs the server token')
        parts = path.split('?', 1)[0].split('/')[3:]
        if parts == ['health'] and method == 'GET':
            data = dict(self.info, ok=True, pid=os.getpid(), base=self.base,
                        models=sorted(self.models), closing=self.closing)
        elif parts == ['paths'] and method == 'POST':
            data = {'paths': self.version_paths(json.loads(body.decode('utf-8') or '{}'))}
        elif parts == ['runs'] and method == 'POST':
            if self.closing:
                raise HTTPError(503, 'closing', 'the Copula server is shutting down')
            try:
                rec, url = self.register_run(json.loads(body.decode('utf-8') or '{}'))
            except ValueError as err:
                raise HTTPError(400, 'invalid_call', err)
            data = {'run': rec, 'url': url}
        elif len(parts) == 3 and parts[0] == 'runs' and parts[2] == 'end' and method == 'POST':
            rec = self.reg.get(parts[1])
            if rec is None or rec['parent'] is not None:
                raise HTTPError(404, 'no_run', 'no process run %s' % parts[1])
            req = json.loads(body.decode('utf-8') or '{}')
            data = await self.finish_run(rec['run_id'], req.get('reads'), req.get('ok', True))
        else:
            raise HTTPError(404, 'not_found', 'no control endpoint %s %s' % (method, path))
        await self._send(writer, 200, json.dumps(data).encode(), keep)
        return True

    # -- chat --------------------------------------------------------------------------------

    def _unit(self, rec, headers):
        """The run a request belongs to: a unit (coordinates, or a phase alone) or the process
        run. An inferred holdout (an adapter's guess) whose cases were used in the search phase
        falls back to the search phase with a W5b note; a declared one is an error (W5a)."""
        raw = headers.get('x-copula-coords')
        phase = headers.get('x-copula-phase') or None
        if not raw and not phase:
            return rec
        if phase is not None and phase not in PHASES:
            raise HTTPError(400, 'invalid_call', 'X-Copula-Phase is one of %s' % ', '.join(PHASES))
        coords = {}
        for part in (raw or '').split(';'):
            if part.strip():
                k, _, v = part.partition('=')
                coords[urllib.parse.unquote(k.strip())] = urllib.parse.unquote(v.strip())
        ordinal = int(headers.get('x-copula-unit') or 1)
        decision = _unq(headers.get('x-copula-decision'))
        try:
            return self.reg.unit(rec, coords, ordinal, decision, phase)
        except HoldoutOverlap as err:
            if headers.get('x-copula-phase-source') != 'inferred':
                raise HTTPError(409, 'holdout_overlap', err)
            self.notes[rec['run_id']]['W5b: an inferred holdout evaluation used search cases; '
                                      'scored in the search phase (optimistic)'] += 1
            return self.reg.unit(rec, coords, ordinal, decision, 'search')

    def _identity(self, backend):
        if isinstance(self.identity, dict):
            return bool(self.identity.get(backend, False))
        return bool(self.identity)

    def _forget(self, occ):
        lst = self._occ.get(occ.rkey)
        if lst is not None:
            if occ in lst:
                lst.remove(occ)
            if not lst:
                self._occ.pop(occ.rkey, None)
        if occ.idem_key is not None and self._idem.get(occ.idem_key) is occ:
            del self._idem[occ.idem_key]

    def _sweep(self):
        now = time.time()
        for occ in [o for lst in self._occ.values() for o in lst]:
            if occ.delivered or occ.failed() or now - occ.created > RETRY_TTL:
                self._forget(occ)

    def _reclaim(self, rkey, idem_key):
        """An undelivered occurrence of `rkey` whose client is gone, for a signalled retry."""
        cands = []
        if idem_key is not None and idem_key in self._idem:
            cands.append(self._idem[idem_key])
        cands += self._occ.get(rkey, [])
        now = time.time()
        for occ in cands:                       # the earliest undelivered occurrence first
            if (occ.rkey == rkey and not occ.delivered and not occ.failed()
                    and now - occ.created < RETRY_TTL and occ.dead()):
                return occ
        return None

    async def _chat(self, rec, proto, headers, reader, writer, keep):
        req = proto.chat
        self._requests += 1
        if self._requests % SWEEP_EVERY == 0:
            self._sweep()
        unit = self._unit(rec, headers)
        proc = unit['parent'] or unit['run_id']
        model = req.get('model')
        backend = self.models.get(model, self.models.get('*'))
        if backend is None:
            raise HTTPError(404, 'model_not_found', 'model %r is not served by this Copula '
                            'server (serves: %s)' % (model, ', '.join(self.models)))
        attrs = infer(req, self._identity(backend))
        role = _unq(headers.get('x-copula-role'))
        rule = self.run_routes.get(proc, self.routes).get(role) if role else None
        if rule == 'fresh':
            attrs = attrs._replace(cache=False, reason='route fresh')
        for note in attrs.notes:
            self.notes[proc][note] += 1
        base_group = self.reg.route(unit, role, rule) if rule else unit['grp']
        name = self.ensure_node(backend, attrs)
        model_, messages, params, stream, stream_options = self.s._intercepted(attrs.body)
        n = params.pop('n', None)
        n = 1 if n is None else n
        if type(n) is not int or n < 1:
            raise HTTPError(400, 'invalid_call', 'n must be a positive int')
        if 'max_tokens' not in params and 'max_completion_tokens' not in params:
            params['max_tokens'] = self.default_max_tokens
        run = await self.open_run(unit)
        if attrs.seed is not None:              # change V2: children keyed seed + i
            calls = [(seed_group(base_group, attrs.seed + i if isinstance(attrs.seed, int)
                                 else '%s+%d' % (attrs.seed, i)), 1) for i in range(n)]
        else:
            calls = [(base_group, n)]
        retry = _int(headers.get('x-stainless-retry-count')) > 0
        idem = headers.get('idempotency-key')
        idem_key = (run.id, idem) if idem else None
        rkey = (run.id, base_group, name, h(attrs.body, attrs.seed, n))
        occ = None
        if retry or (idem_key is not None and idem_key in self._idem):
            occ = self._reclaim(rkey, idem_key)
        reread = occ is not None
        if occ is None:
            fut = asyncio.ensure_future(self._calls(run, calls, name, messages, params, model))
            occ = Occurrence(fut, reader, writer, rkey, idem_key)
            self._occ[rkey].append(occ)
            if idem_key is not None:
                self._idem[idem_key] = occ
        else:
            occ.reader, occ.writer = reader, writer     # the retry now owns the delivery
            self.notes[proc]['retry re-read an undelivered occurrence'] += 1
        if self.calls is not None and not reread:
            digest = h(name, messages, params)
            occ.log = ({'exp': unit['experiment'], 'proc': proc, 'run': unit['run_id'],
                        'ordinal': _int(headers.get('x-copula-unit')) or 1 if unit['parent']
                        else None, 'grp': base_group, 'node': name, 'model': model,
                        'body': digest, 'n': n,
                        'seed': None if attrs.seed is None else str(attrs.seed), 'role': role,
                        'reason': attrs.reason, 'proto': proto.name,
                        'rev': (self.s.models.get(backend) or {}).get('model_rev')},
                       digest, {'model': model, 'messages': messages, 'params': params})
        info = {'x-copula-run': unit['run_id'], 'x-copula-group': base_group,
                'x-copula-rule': '%s; %s; %s' % (unit['rule'], attrs.reason, proto.name)}
        if stream:
            return await self._stream(occ, writer, keep, info, proto, proc, reread)
        return await self._plain(occ, writer, keep, info, proto, proc, reread)

    async def _calls(self, run, calls, name, messages, params, model):
        parts = await asyncio.gather(*[self.s._llm(run, Group(g), name, messages,
                                                   dict(params, n=n), 'copula')
                                       for g, n in calls])
        data = dict(parts[0])
        choices, statuses, ranks = [], [], []
        pt = ct = 0
        for p in parts:
            for ch in p['choices']:
                choices.append(dict(ch, index=len(choices)))
            for c in p['evalrt']['calls']:
                statuses.append(c['status'])
                ranks.append('-' if c['index'] is None else str(c['index']))
            pt = p['usage']['prompt_tokens']
            ct += p['usage']['completion_tokens']
        data.update(choices=choices, model=model,
                    usage=dict(parts[0]['usage'], completion_tokens=ct, total_tokens=pt + ct))
        data.pop('evalrt', None)
        return data, statuses, ranks

    def _outcome(self, occ, proc, reread):
        """(status code, data or (kind, detail), statuses, ranks) of a finished occurrence; the
        statuses are counted once per occurrence (a retry's re-read is not counted again)."""
        try:
            data, statuses, ranks = occ.future.result()
        except SDKError as err:
            self._log(occ, [], [], err.kind)
            return 400, (err.kind, err.detail), [], []
        except EvalrtError as err:
            self._log(occ, [], [], err.kind)
            return 502, (err.kind, err.detail), [], []
        except Exception as err:                # noqa: BLE001 -- always answer
            self._log(occ, [], [], 'internal')
            return 500, ('internal', repr(err)), [], []
        if not getattr(occ, 'counted', False):
            occ.counted = True
            self.stats[proc].update(statuses)
            self._log(occ, statuses, ranks, None)
        return 200, data, statuses, ranks

    def _log(self, occ, statuses, ranks, error):
        """One call-log row per occurrence (a retry's re-read is not logged again)."""
        entry = getattr(occ, 'log', None)
        if entry is None or self.calls is None:
            return
        occ.log = None
        row, digest, body = entry
        row = dict(row, t=time.time(), statuses=','.join(statuses), ranks=','.join(ranks),
                   reread=0, error=error)
        try:
            self.calls.add(row, digest, body)
        except Exception:                       # noqa: BLE001 -- diagnostics never fail a call
            pass

    def _delivered(self, occ):
        occ.delivered = True
        self._forget(occ)

    @staticmethod
    def _status_line(statuses, ranks, reread):
        return ','.join(statuses) + (' (retry re-read)' if reread else ''), ','.join(ranks)

    async def _plain(self, occ, writer, keep, info, proto, proc, reread):
        await asyncio.wait({occ.future})
        status, data, statuses, ranks = self._outcome(occ, proc, reread)
        if occ.writer is not writer or occ.dead():
            return False                        # adopted by a retry, or the client is gone
        try:
            if status != 200:
                body = json.dumps(proto.error(status, *data)).encode()
                await self._send(writer, status, body, keep, info)
            else:
                st, rk = self._status_line(statuses, ranks, reread)
                extra = dict(info, **{'x-copula-status': st, 'x-copula-rank': rk})
                await self._send(writer, 200, json.dumps(proto.render(data)).encode(), keep, extra)
        except (ConnectionError, RuntimeError):
            return False                        # undelivered: a signalled retry may re-read it
        self._delivered(occ)
        return True

    async def _stream(self, occ, writer, keep, info, proto, proc, reread):
        head = ['HTTP/1.1 200 OK', 'Content-Type: text/event-stream', 'Cache-Control: no-cache',
                'Transfer-Encoding: chunked', 'Connection: %s' % ('keep-alive' if keep else 'close')]
        head += ['%s: %s' % (k, _hval(v)) for k, v in info.items()]
        try:
            writer.write(('\r\n'.join(head) + '\r\n\r\n').encode('latin-1'))
            await writer.drain()
            while not occ.future.done():
                done, _ = await asyncio.wait({occ.future}, timeout=self.keepalive)
                if not done:
                    writer.write(_chunk(proto.keepalive))
                    await writer.drain()
            status, data, statuses, ranks = self._outcome(occ, proc, reread)
            if occ.writer is not writer or occ.dead():
                return False
            if status != 200:
                writer.write(_chunk(proto.stream_error(status, *data).encode()))
            else:
                st, rk = self._status_line(statuses, ranks, reread)
                try:
                    events = proto.events(data) + ': x-copula-status %s; x-copula-rank %s\n\n' % (
                        st, rk)
                except Exception as err:            # noqa: BLE001 -- the head is already sent
                    events = proto.stream_error(500, 'internal', repr(err))
                writer.write(_chunk(events.encode()))
            writer.write((_chunk(proto.done) if proto.done else b'') + b'0\r\n\r\n')
            await writer.drain()
        except (ConnectionError, RuntimeError):
            return False
        self._delivered(occ)
        return True


def _chunk(b):
    return b'%x\r\n' % len(b) + b + b'\r\n'


def _hval(v):
    """A header value: printable ASCII only (no CR/LF injection, no encoding error)."""
    return ''.join(c if ' ' <= c <= '~' else '?' for c in str(v))


def _unq(v):
    return None if v is None else urllib.parse.unquote(v)


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0
