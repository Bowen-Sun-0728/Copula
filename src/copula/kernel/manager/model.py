"""The model boundary: what a client returns and how its failures are classified.

Retrying is unbiased only when the failure is independent of the output it interrupted; retrying an
output-correlated failure (a long generation that times out) selects samples by outcome
(RUNTIME_V0 section 12.5). Hence:

- `TransportError`: connection refused or reset, 5xx, a timeout set far above any legitimate
  generation. Retried; nothing is written.
- `RequestError`: a failure the request itself determines (context overflow, a 4xx on a well-formed
  request). Written under the key as an error sample, like any graph failure.
- A normal return is a sample, whether it stopped or was truncated at `max_tokens`.

Anything else a client raises fails the flight without retrying, because it may depend on the output.

Metering (campaign 2026-10-02, SCHEMA section 3): `generate_with_meta` also returns what one HTTP
attempt cost -- timing, the replica, the cache salt sent, token counts with the cached prefix -- as
a separate `meta` dict. Meta never enters a sample (`validate_sample` keeps the sample schema), so
it can never reach the store or a key. A client attaches the meta of a failed attempt to the
exception it raises (`.meta`).
"""

FINISH_REASONS = ('stop', 'length', 'tool_calls')


class TransportError(Exception):
    """Output-independent infrastructure failure."""
    meta = None                         # the failed attempt's meta, when the client has one


class RequestError(Exception):
    """A failure determined by the request; becomes an error sample."""
    meta = None

    def __init__(self, kind, detail=''):
        super().__init__('%s: %s' % (kind, detail))
        self.kind = kind
        self.detail = detail


def error_sample(kind, detail=''):
    return {'error': str(kind), 'detail': str(detail)[:500]}


def _tool_calls_ok(calls):
    return calls is None or (isinstance(calls, list) and calls and all(
        isinstance(c, dict) and isinstance(c.get('id'), str) and c.get('type') == 'function'
        and isinstance(c.get('function'), dict) and isinstance(c['function'].get('name'), str)
        and isinstance(c['function'].get('arguments'), str) for c in calls))


def validate_sample(sample):
    """The one sample schema. A malformed return is treated as a transport failure. `tool_calls`
    (the model's calls of the request's tools, sysval build 2026-10-09) is kept only when present,
    so a sample without them is stored exactly as before."""
    ok = (isinstance(sample, dict)
          and isinstance(sample.get('content'), str)
          and sample.get('finish_reason') in FINISH_REASONS
          and (sample.get('reasoning') is None or isinstance(sample.get('reasoning'), str))
          and _tool_calls_ok(sample.get('tool_calls'))
          and all(type(sample.get(k)) is int and sample[k] >= 0
                  for k in ('prompt_tokens', 'completion_tokens')))
    if not ok:
        raise TransportError('malformed sample from the model client: %r' % (sample,))
    out = {'content': sample['content'], 'reasoning': sample.get('reasoning'),
           'finish_reason': sample['finish_reason'], 'prompt_tokens': sample['prompt_tokens'],
           'completion_tokens': sample['completion_tokens']}
    if sample.get('tool_calls') is not None:
        out['tool_calls'] = [{'id': c['id'], 'type': 'function',
                              'function': {'name': c['function']['name'],
                                           'arguments': c['function']['arguments']}}
                             for c in sample['tool_calls']]
    return out


class ModelClient:
    """Interface of a model client.

    `defaults` is the pinned table of sampling parameters every request is completed with, and
    `model_rev` identifies everything the output distribution depends on beyond the request:
    weights, tokenizer and chat template, server version, dtype, and that table.
    """
    model_rev = None
    defaults = None

    async def generate(self, body, key_hash):
        """Return one sample for the normalised `body`. `key_hash` is for tests and logs only."""
        raise NotImplementedError

    async def generate_with_meta(self, body, key_hash, ctx):
        """(sample, meta or None) of one attempt. `ctx` is {salt, route, attempt, owner, node,
        key_hash}: the cache salt to send (never part of the body), the routing key (the case key),
        the attempt number and who pays. A client without metering returns no meta."""
        return (await self.generate(body, key_hash)), None

    async def aclose(self):
        pass
