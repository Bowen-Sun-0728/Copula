"""Attributes of a model call when nobody declared them (docs/concepts/identical-calls.md).

    temperature 0 or top_k 1, upstream identity established  -> cached, deterministic (global)
    temperature 0 or top_k 1, no established identity         -> cached, stochastic (W12)
    an explicit seed                                         -> cached, stochastic, seed in the key
    a foreign previous_response_id / conversation / thread   -> uncached
    a server-side tool in the request (not type "function")  -> uncached
    anything else                                            -> cached, stochastic

A seed selects a stream: it is part of the group, so each seed has its own numbered
samples, and `n > 1` expands to the streams seed+i. The seed itself is not sent upstream; the
engine seed of every generation is derived from its sample address (copula.upstream). Natively,
identical seeded requests inside one run may return identical text on a seed-honouring server;
here the k-th of them reads the k-th sample of the seed's stream.
"""
import collections

STATEFUL = ('previous_response_id', 'conversation', 'thread_id')

Attrs = collections.namedtuple('Attrs', 'randomness cache greedy seed reason body notes')


def infer(body, identity=True):
    """The Attrs of one chat-completions request body (a dict, model and messages included)."""
    body = dict(body)
    notes = []
    extra = body.get('extra_body') if isinstance(body.get('extra_body'), dict) else {}
    temperature = body.get('temperature', extra.get('temperature'))
    top_k = body.get('top_k', extra.get('top_k'))
    greedy = 'temperature' if temperature == 0 else ('top_k' if top_k == 1 else None)
    for name in STATEFUL:
        if body.get(name) or extra.get(name):
            return Attrs('stochastic', False, greedy, None, 'stateful: %s' % name, body, notes)
    tools = body.get('tools') or []
    server_side = sorted({t.get('type') for t in tools if isinstance(t, dict)
                          and t.get('type') not in (None, 'function')})
    if server_side:
        return Attrs('stochastic', False, greedy, None,
                     'server-side tool: %s' % ', '.join(server_side), body, notes)
    seed = body.pop('seed', None)
    if seed is None and 'seed' in extra:
        extra = dict(extra)
        seed = extra.pop('seed')
        body['extra_body'] = extra
    if seed is not None:
        notes.append('seed %r selects its own sample stream (V2); repeats inside one run are '
                     'numbered, not identical' % (seed,))
        return Attrs('stochastic', True, greedy, seed, 'seeded', body, notes)
    if greedy and identity:
        return Attrs('deterministic', True, greedy, None, 'greedy', body, notes)
    if greedy:
        notes.append('greedy, but no established upstream identity: cached per group (W12)')
    return Attrs('stochastic', True, greedy, None, 'default', body, notes)


def node_name(backend, attrs):
    """The registered node an inferred call goes to: one per (backend, attributes)."""
    if not attrs.cache:
        return '%s:u' % backend
    if attrs.randomness == 'deterministic':
        return '%s:d%s' % (backend, '' if attrs.greedy == 'temperature' else 'k')
    return '%s:s' % backend


def node_spec(backend, attrs):
    """register_llm keyword arguments for node_name(backend, attrs)."""
    if not attrs.cache:
        return dict(cache=False, randomness='stochastic', backend=backend)
    if attrs.randomness == 'deterministic':
        pin = {'temperature': 0.0} if attrs.greedy == 'temperature' else {'top_k': 1}
        return dict(cache=True, randomness='deterministic', backend=backend, pin=pin)
    return dict(cache=True, randomness='stochastic', backend=backend)
