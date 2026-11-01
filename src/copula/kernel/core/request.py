"""Model requests: validation and the explicit, canonical body that is hashed into a key.

The manager completes every request with its pinned default table before hashing, so a server-side
default can never differ silently between two requests that hash alike (RUNTIME_V0 section 4.2).
The table is part of `model_rev`. The served model name is not part of the body: model identity
lives in `model_rev`, and the client adds the name when it sends the request.

Tool calling and structured output (sysval build, 2026-10-09): `tools`, `tool_choice`,
`response_format` and `parallel_tool_calls` are part of the request, and so of its pool, when they
are set; messages may then carry tool calls (an assistant turn) and tool results (role `tool`). A
body without them normalises exactly as before, so no existing key changes.
"""
import math

FLOAT_PARAMS = ('temperature', 'top_p', 'min_p', 'presence_penalty', 'frequency_penalty',
                'repetition_penalty')
INT_PARAMS = ('top_k',)
OTHER_PARAMS = ('stop', 'chat_template_kwargs')
SAMPLING_PARAMS = FLOAT_PARAMS + INT_PARAMS + OTHER_PARAMS
ROLES = ('system', 'user', 'assistant', 'tool')
# Request fields that shape the output besides the sampling parameters: kept in the normalised body
# when set (never filled from a table, so a body without them hashes as before).
EXTRA_FIELDS = ('tools', 'tool_choice', 'response_format', 'parallel_tool_calls')

# Never taken from a job. `seed` would make distinct draw keys return the same sample, `n` and
# `best_of` would put several samples under one key; `model` is added by the client.
FORBIDDEN = ('seed', 'n', 'best_of', 'model', 'stream', 'logprobs', 'top_logprobs')


class RequestSpecError(ValueError):
    """A malformed request: a bug in the caller, never a model failure."""


def _require(ok, message):
    if not ok:
        raise RequestSpecError(message)


def check_defaults(defaults):
    """A default table must name every sampling parameter, so normalised bodies are explicit."""
    missing = [name for name in SAMPLING_PARAMS if name not in defaults]
    _require(not missing, 'default table lacks %s' % ', '.join(missing))
    unknown = sorted(set(defaults) - set(SAMPLING_PARAMS))
    _require(not unknown, 'default table has unknown parameters %s' % ', '.join(unknown))
    # Type-check every default by normalising a minimal request against the table.
    normalize_request({'messages': [{'role': 'user', 'content': ''}], 'max_tokens': 1}, defaults)
    return dict(defaults)


def normalize_request(body, defaults):
    """Return the explicit canonical form of `body`, completed from `defaults`."""
    _require(isinstance(body, dict), 'request body must be an object')
    bad = sorted(set(body) & set(FORBIDDEN))
    _require(not bad, 'request may not set %s' % ', '.join(bad))
    unknown = sorted(set(body) - set(SAMPLING_PARAMS) - set(EXTRA_FIELDS)
                     - {'messages', 'max_tokens'})
    _require(not unknown, 'unknown request fields %s' % ', '.join(unknown))

    messages = body.get('messages')
    _require(isinstance(messages, list) and messages, 'messages must be a non-empty list')
    clean = [normalize_message(message) for message in messages]

    max_tokens = body.get('max_tokens')
    _require(type(max_tokens) is int and max_tokens >= 1, 'max_tokens must be a positive int')

    out = {'messages': clean, 'max_tokens': max_tokens}
    for name in SAMPLING_PARAMS:
        value = body[name] if name in body else defaults[name]
        if name in FLOAT_PARAMS:
            _require(type(value) in (int, float) and math.isfinite(value), '%s must be a number' % name)
            value = float(value)
        elif name in INT_PARAMS:
            _require(type(value) is int and value >= -1, '%s must be an int >= -1' % name)
        elif name == 'stop':
            _require(isinstance(value, list) and all(isinstance(s, str) and s for s in value),
                     'stop must be a list of non-empty strings')
            value = sorted(set(value))            # a set of stop strings: order is meaningless
        else:
            _require(isinstance(value, dict), 'chat_template_kwargs must be an object')
        out[name] = value
    for name in EXTRA_FIELDS:
        if body.get(name) is not None:
            out[name] = _extra(name, body[name])
    return out


def _json_value(value, what):
    """A JSON-shaped copy of `value` (objects, arrays, text, numbers, booleans, null)."""
    if isinstance(value, dict):
        _require(all(isinstance(k, str) for k in value), '%s has non-text keys' % what)
        return {k: _json_value(v, what) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v, what) for v in value]
    if isinstance(value, float):
        _require(math.isfinite(value), '%s has a non-finite number' % what)
    _require(value is None or isinstance(value, (str, int, float, bool)), '%s is not JSON' % what)
    return value


def _extra(name, value):
    if name == 'tools':
        _require(isinstance(value, list) and value and all(
            isinstance(t, dict) and t.get('type') == 'function' and isinstance(t.get('function'), dict)
            and isinstance(t['function'].get('name'), str) for t in value),
            'tools is a non-empty list of {type: function, function: {name, ...}}')
    elif name == 'tool_choice':
        _require(value in ('auto', 'none', 'required') or (
            isinstance(value, dict) and value.get('type') == 'function'),
            "tool_choice is 'auto', 'none', 'required' or {type: function, function: {name}}")
    elif name == 'response_format':
        _require(isinstance(value, dict) and value.get('type') in ('text', 'json_object',
                                                                    'json_schema'),
                 'response_format is {type: text | json_object | json_schema, ...}')
    else:
        _require(type(value) is bool, 'parallel_tool_calls must be a boolean')
    return _json_value(value, name)


def normalize_tool_calls(calls):
    """The canonical tool calls of an assistant turn: [{id, type: function, function: {name,
    arguments}}], in order."""
    _require(isinstance(calls, list) and calls, 'tool_calls is a non-empty list')
    out = []
    for call in calls:
        fn = call.get('function') if isinstance(call, dict) else None
        _require(isinstance(fn, dict) and isinstance(call.get('id'), str)
                 and call.get('type', 'function') == 'function'
                 and isinstance(fn.get('name'), str) and isinstance(fn.get('arguments'), str),
                 'a tool call is {id, type: function, function: {name, arguments: text}}')
        out.append({'id': call['id'], 'type': 'function',
                    'function': {'name': fn['name'], 'arguments': fn['arguments']}})
    return out


def normalize_message(message):
    """One message in canonical form. A plain turn is exactly {role, content} (as before); an
    assistant turn may carry `tool_calls` (its content then may be None) and `reasoning_content`;
    a tool result is {role: tool, content, tool_call_id}. Any turn may carry OpenAI's `name`, which
    is then part of the request."""
    _require(isinstance(message, dict), 'each message is an object')
    role = message.get('role')
    _require(role in ROLES, 'message role %r' % (role,))
    name = message.get('name')
    _require(name is None or isinstance(name, str), 'a message name is text')
    keys = set(message) - {'name'}
    if role in ('system', 'user'):
        _require(keys == {'role', 'content'}, 'a %s message is exactly {role, content}' % role)
    elif role == 'tool':
        _require(keys == {'role', 'content', 'tool_call_id'},
                 'a tool message is exactly {role, content, tool_call_id}')
        _require(isinstance(message['tool_call_id'], str), 'tool_call_id must be text')
    else:
        _require('content' in keys or 'tool_calls' in keys, 'an assistant message has content')
        _require(not keys - {'role', 'content', 'tool_calls', 'reasoning_content'},
                 'an assistant message is {role, content, tool_calls?, reasoning_content?}')
    content = message.get('content')
    if role == 'assistant' and message.get('tool_calls') is not None:
        _require(content is None or isinstance(content, str), 'message content must be text')
    else:
        _require(isinstance(content, str), 'message content must be text')
    out = {'role': role, 'content': content}
    if name is not None:
        out['name'] = name
    if role == 'tool':
        out['tool_call_id'] = message['tool_call_id']
    if role == 'assistant':
        if message.get('tool_calls') is not None:
            out['tool_calls'] = normalize_tool_calls(message['tool_calls'])
        if message.get('reasoning_content') is not None:
            _require(isinstance(message['reasoning_content'], str), 'reasoning_content is text')
            out['reasoning_content'] = message['reasoning_content']
    return out
