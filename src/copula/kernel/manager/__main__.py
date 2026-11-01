"""Run one cache manager.

    python -m evalrt.manager --store PATH --socket PATH --mock NONCE
    python -m evalrt.manager --store PATH --socket PATH --vllm URL --served-model NAME \
        --model-dir DIR --defaults JSON [--identity-extra JSON] [--inflight N] [--route case] \
        [--gen-ledger PATH [--gen-bodies PATH]] [--expect-prompt-tokens-details] \
        [--models JSON | @FILE]

Several backends (models INTERFACE J3, 2026-10-09): the flags above define the actor; `--models`
adds named backends, `{name: {"vllm": URL[,URL...], "served_model", "model_dir", "defaults",
"identity_extra", "type": "generate"|"reward", "route", "workers", "reward_input"}}`, each with its
own `model_rev`. A graph node names one with `model`. `--mock-models a,b:reward` adds mock backends
to `--mock`. The in-flight cap applies to each backend.

With `--vllm` the manager waits for the server, then derives `model_rev` from the served model,
the model's files, the vLLM version, `--identity-extra` (dtype, quantisation, KV-cache dtype) and the
pinned default table (`--defaults`, every sampling parameter). The store is locked before the socket
is bound, so a second manager on the same store fails at start. SIGTERM or SIGINT drains: stop
accepting, let sent flights finish and be written, close the store. SIGKILL is a crash: unwritten
flights are lost and nothing partial is ever written.

Metering (campaign 2026-10-02, SCHEMA section 4): `--gen-ledger` appends one row per HTTP attempt
(and a `start` row, written before the socket is bound, and a `stop` row); `--gen-bodies` keeps each
distinct prompt once. The vLLM client runs `--workers` requests at once (default `--inflight`).
`--expect-prompt-tokens-details` probes every replica at start -- one prompt twice under a fresh
salt; the second reply must report cached prompt tokens -- and exits with
`prompt_tokens_details_missing` before binding the socket when one does not.
"""
import argparse
import asyncio
import json
import os
import signal
import socket
import sys
import time

from .. import rpc
from ..core.canon import digest
from ..rpc import claim_socket
from ..store.entries import EntryStore
from ..testing.mock_model import MockModel, MockReward
from .flights import LEASE_TTL, MODEL_TYPES, Manager
from .ledger import GenLedger, manager_id, meta_fields
from .server import serve
from .vllm_client import REWARD_INPUTS, ROUTES, RewardClient, VLLMClient, file_identity, \
    server_identity


def _json_arg(value):
    """A JSON value given inline or as @file."""
    if value is None:
        return None
    if value.startswith('@'):
        with open(value[1:]) as handle:
            return json.load(handle)
    return json.loads(value)


async def _server(urls, served_model, wait_ready):
    """The identity the replicas in `urls` report, once they are all up and identical."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait_ready
    servers = []
    for url in [u.strip() for u in urls.split(',') if u.strip()]:
        while True:
            try:
                servers.append(server_identity(url))
                break
            except (OSError, ValueError):
                if loop.time() > deadline:
                    raise SystemExit('vLLM at %s not ready after %.0f s' % (url, wait_ready))
                await asyncio.sleep(2.0)
    server = servers[0]
    if any(other != server for other in servers[1:]):
        raise SystemExit('the replicas in %s differ: %s' % (urls, servers))
    if served_model not in server['models']:
        raise SystemExit('%s serves %s, not %s' % (urls, server['models'], served_model))
    return server


async def _vllm(args):
    server = await _server(args.vllm, args.served_model, args.wait_ready)
    identity = {'server': server, 'files': file_identity(args.model_dir),
                'extra': _json_arg(args.identity_extra) or {}}
    return VLLMClient(args.vllm, args.served_model, _json_arg(args.defaults), identity,
                      timeout=args.timeout, workers=args.workers or args.inflight,
                      route=args.route)


BACKEND_FIELDS = {'vllm', 'served_model', 'model_dir', 'defaults', 'identity_extra', 'type',
                  'route', 'workers', 'reward_input'}


async def _backends(args):
    """{name: client} of the named backends in `--models` (real) or `--mock-models` (mock)."""
    out = {}
    if args.mock_models:
        for item in [x.strip() for x in args.mock_models.split(',') if x.strip()]:
            name, _, kind = item.partition(':')
            if kind not in ('', 'generate', 'reward'):
                raise SystemExit('--mock-models: %r is not name or name:reward' % item)
            if name in out:
                raise SystemExit('--mock-models names %r twice' % name)
            nonce = '%s:%s' % (args.mock, name)
            client = MockReward(nonce, latency=args.latency) if kind == 'reward' else \
                MockModel(nonce, latency=args.latency, meta=True if args.mock_meta else None)
            client.contexts = None
            out[name] = client
    table = _json_arg(args.models) or {}
    if not isinstance(table, dict):
        raise SystemExit('--models is {name: backend}')
    for name, spec in sorted(table.items()):
        if name in out:
            raise SystemExit('--models and --mock-models both name %r' % name)
        if not isinstance(spec, dict) or set(spec) - BACKEND_FIELDS or not {
                'vllm', 'served_model', 'model_dir', 'defaults'} <= set(spec):
            raise SystemExit('--models %s: fields are %s (vllm, served_model, model_dir, defaults '
                             'required)' % (name, sorted(BACKEND_FIELDS)))
        kind = spec.get('type', 'generate')
        if kind not in MODEL_TYPES:
            raise SystemExit('--models %s: type is one of %s' % (name, MODEL_TYPES))
        server = await _server(spec['vllm'], spec['served_model'], args.wait_ready)
        identity = {'server': server, 'files': file_identity(spec['model_dir']),
                    'extra': spec.get('identity_extra') or {}}
        defaults = spec['defaults']
        if isinstance(defaults, str):
            defaults = _json_arg(defaults)
        common = dict(timeout=args.timeout, workers=spec.get('workers') or args.workers
                      or args.inflight, route=spec.get('route', 'rr'))
        try:
            if kind == 'reward':
                reward_input = spec.get('reward_input', 'raw')
                if reward_input not in REWARD_INPUTS:
                    raise SystemExit('--models %s: reward_input is one of %s'
                                     % (name, REWARD_INPUTS))
                out[name] = RewardClient(spec['vllm'], spec['served_model'], defaults, identity,
                                         reward_input=reward_input, **common)
                ok, detail = await asyncio.get_running_loop().run_in_executor(
                    None, out[name].probe_scores)
                if not ok:
                    raise SystemExit('--models %s: the reward server does not score: %s'
                                     % (name, detail))
            else:
                out[name] = VLLMClient(spec['vllm'], spec['served_model'], defaults, identity,
                                       **common)
        except (ValueError, TypeError) as err:            # a bad pinned table, route, ...
            raise SystemExit('--models %s: %s' % (name, err))
    return out


async def _probe(model, args, ledger):
    """The startup probe of every replica (rows `ev = "probe"`). True when all pass."""
    loop = asyncio.get_running_loop()
    passed = True
    for replica in range(len(model.urls)):
        ok, body, salt, metas = await loop.run_in_executor(None, model.probe, replica,
                                                           args.probe_chars)
        passed = passed and ok
        if ledger is None:
            continue
        prompt_digest = digest(body['messages'])
        ledger.body(prompt_digest, body['messages'])
        for attempt, meta in enumerate(metas, 1):
            meta = meta or {}
            ledger.row('probe', aid='%s/probe/%d/%d' % (ledger.mgr, replica, attempt), seq=None,
                       attempt=attempt, key_hash=None, pool=None, index=None, partition=None,
                       case_key=None, owner_ticket=None, node='probe', cache_salt=salt,
                       route=None, replica=replica, instance_url=model.urls[replica],
                       prompt_digest=prompt_digest, max_tokens=body['max_tokens'],
                       t_created=None, t_dispatch=None,
                       outcome='ok' if meta.get('error') is None and meta.get('t_done')
                       else 'transport_error', n=1, passed=ok, **meta_fields(meta))
        ledger.flush()
    return passed


async def _run(args, argv):
    if bool(args.mock) == bool(args.vllm):
        raise SystemExit('give exactly one of --mock and --vllm')
    if args.expect_prompt_tokens_details and not args.vllm:
        raise SystemExit('--expect-prompt-tokens-details probes a vLLM server: it needs --vllm')
    if args.mock_models and not args.mock:
        raise SystemExit('--mock-models adds mock backends to --mock')
    start_t = time.time()
    mgr = manager_id(start_t)
    store = EntryStore(args.store, batched=not args.no_group_commit)
    if args.mock:
        model = MockModel(args.mock, latency=args.latency, meta=True if args.mock_meta else None)
        model.contexts = None           # a long-running mock keeps no per-attempt history
    else:
        model = await _vllm(args)
    backends = await _backends(args)
    workers = getattr(model, 'workers', None)
    ledger = GenLedger(args.gen_ledger, args.gen_bodies, args.ledger_flush, mgr=mgr) \
        if args.gen_ledger else None
    if ledger is not None:
        ledger.row('start', t=start_t, pid=os.getpid(), host=socket.gethostname(),
                   model_rev=model.model_rev, served_model=getattr(model, 'served_model', None),
                   urls=getattr(model, 'urls', None), defaults=model.defaults,
                   identity=getattr(model, 'identity', None), workers=workers,
                   inflight=args.inflight, max_attempts=args.max_attempts,
                   backoff_base=args.backoff, backoff_cap=args.backoff_cap,
                   route=getattr(model, 'route', 'rr'),
                   expect_prompt_tokens_details=args.expect_prompt_tokens_details,
                   store=os.path.abspath(args.store), socket=os.path.abspath(args.socket),
                   argv=list(argv),
                   models={name: {'model_rev': c.model_rev, 'type': getattr(c, 'kind', 'generate'),
                                  'served_model': getattr(c, 'served_model', None),
                                  'urls': getattr(c, 'urls', None), 'defaults': c.defaults,
                                  'identity': getattr(c, 'identity', None)}
                           for name, c in sorted(backends.items())})
        ledger.flush()
    probe = 'off'
    if args.expect_prompt_tokens_details:
        if not await _probe(model, args, ledger):
            if ledger is not None:
                ledger.close()
            await model.aclose()
            for client in backends.values():
                await client.aclose()
            store.close()
            raise SystemExit('prompt_tokens_details_missing: a replica of %s did not report '
                             'cached prompt tokens on a repeated prompt (start vLLM with '
                             '--enable-prompt-tokens-details)' % args.vllm)
        probe = 'ok'
        if ledger is not None:
            ledger.probe = 'ok'
    try:
        manager = Manager(store, model, inflight_cap=args.inflight,
                          max_attempts=args.max_attempts, backoff_base=args.backoff,
                          backoff_cap=args.backoff_cap, ledger=ledger, models=backends,
                          lease_ttl=args.lease_ttl)
    except ValueError as err:
        raise SystemExit('bad backends: %s' % err)
    manager.config = {'urls': getattr(model, 'urls', None), 'workers': workers,
                      'inflight': args.inflight, 'max_attempts': args.max_attempts,
                      'backoff_base': args.backoff, 'backoff_cap': args.backoff_cap,
                      'lease_ttl': manager.lease_ttl,
                      'route': getattr(model, 'route', 'rr'), 'gen_ledger': args.gen_ledger,
                      'gen_bodies': args.gen_bodies,
                      'expect_prompt_tokens_details': args.expect_prompt_tokens_details,
                      'probe': probe, 'pid': os.getpid(), 'mgr': mgr,
                      'models': {name: {'urls': getattr(c, 'urls', None),
                                        'workers': getattr(c, 'workers', None),
                                        'route': getattr(c, 'route', 'rr')}
                                 for name, c in sorted(backends.items())}}
    await manager.start()
    claim_socket(args.socket)
    server = await serve(manager, args.socket)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    print(json.dumps({'event': 'ready', 'socket': args.socket, 'store': args.store,
                      'model_rev': model.model_rev, 'defaults': model.defaults,
                      'models': manager.model_table(), 'pid': os.getpid()}), flush=True)
    await stop.wait()
    await rpc.close_server(server)
    await manager.drain()
    stats = manager.stats()
    await manager.stop()
    if ledger is not None:
        ledger.close(stats)
    await model.aclose()
    for client in backends.values():
        await client.aclose()
    store.close()
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    print(json.dumps({'event': 'stopped', 'stats': stats}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m evalrt.manager', description=__doc__)
    parser.add_argument('--store', required=True, help='SQLite sample store (created if absent)')
    parser.add_argument('--socket', required=True, help='Unix socket path to serve on')
    parser.add_argument('--mock', help='serve the mock model with this nonce')
    parser.add_argument('--latency', type=float, default=0.0, help='mock latency per call, s')
    parser.add_argument('--vllm', help='base URL of a vLLM OpenAI-compatible server; several '
                        'identical replicas may be given comma separated (round robin)')
    parser.add_argument('--served-model', help='the model name the server serves')
    parser.add_argument('--model-dir', help='the model files, hashed into model_rev')
    parser.add_argument('--defaults', help='pinned sampling table: JSON or @file')
    parser.add_argument('--identity-extra', help='serving settings for model_rev: JSON or @file')
    parser.add_argument('--timeout', type=float, default=900.0, help='per-request timeout, s')
    parser.add_argument('--wait-ready', type=float, default=600.0, help='wait for the server, s')
    parser.add_argument('--inflight', type=int, default=64, help='cap on in-flight model calls')
    parser.add_argument('--workers', type=int, default=None,
                        help='vLLM client threads (default: --inflight)')
    parser.add_argument('--route', choices=ROUTES, default='rr',
                        help='replica choice: round robin, or by case key (prefix-cache affinity)')
    parser.add_argument('--max-attempts', type=int, default=6)
    parser.add_argument('--lease-ttl', type=float, default=LEASE_TTL,
                        help='a renewing holder\'s tool lease expires this long after its last '
                             'renewal, s (>= 5; 0: never)')
    parser.add_argument('--backoff', type=float, default=0.5, help='base retry backoff, s')
    parser.add_argument('--backoff-cap', type=float, default=30.0, help='retry backoff cap, s')
    parser.add_argument('--gen-ledger', default=None,
                        help='append one JSONL row per model attempt here (SCHEMA section 4)')
    parser.add_argument('--gen-bodies', default=None,
                        help='append each distinct prompt here once, by prompt_digest')
    parser.add_argument('--ledger-flush', type=float, default=1.0, help='ledger flush interval, s')
    parser.add_argument('--expect-prompt-tokens-details', action='store_true',
                        help='refuse to start unless every replica reports cached prompt tokens')
    parser.add_argument('--probe-chars', type=int, default=8000,
                        help='size of the startup probe prompt, characters')
    parser.add_argument('--models', default=None,
                        help='named backends besides the actor: JSON or @file (models INTERFACE J3)')
    parser.add_argument('--mock-models', default=None,
                        help='named mock backends with --mock: name[:reward],...')
    parser.add_argument('--no-group-commit', action='store_true',
                        help='commit every store write on its own (default: group commit, one '
                             'fsync per batch; a reply still leaves only after its writes are '
                             'durable)')
    parser.add_argument('--mock-meta', action='store_true',
                        help='the mock model returns synthetic metering (tests and smokes)')
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)
    if not (args.lease_ttl == 0 or 5 <= args.lease_ttl < float('inf')):
        parser.error('--lease-ttl is 0 (no expiry) or a number of seconds >= 5')
    asyncio.run(_run(args, argv))


if __name__ == '__main__':
    sys.exit(main())
