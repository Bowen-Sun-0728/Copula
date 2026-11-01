"""Wire protocols of the Copula proxy (DESIGN.md §10.5, O8; preliminary).

The kernel serves chat completions. Each protocol here translates a client request into one chat
request and renders the chat result back, buffered or as that protocol's server-sent events:

    ChatProto       OpenAI Chat Completions, unchanged
    ResponsesProto  OpenAI Responses: input items, instructions, function tools, text formats,
                    and previous_response_id chains for responses this server issued (the chain is
                    expanded into the full input: the key covers everything the model sees)
    AnthropicProto  Anthropic Messages: system, content blocks (text, tool_use, tool_result),
                    tools, tool_choice, stop_sequences, top_k; ping events while generating

Ids are derived from the sample's address (the chat id), so a replay returns the same ids and a
later request that echoes them keys the same (§3.2 item 4). Anything a protocol cannot translate
faithfully is refused with a 400 rather than silently changed: images and documents, server-side
tools, extended thinking, reasoning effort, conversations, foreign response ids.

Translation notes (documented choices):
- several text blocks or parts of one turn are joined with "\\n" (vLLM's own rule for text parts);
- Anthropic `cache_control` markers are dropped (a provider prompt-cache hint);
- Anthropic `tool_result.is_error` is dropped (chat has no such flag);
- a chat 'stop' finish is reported as end_turn (the stop sequence that matched is not known),
  and as tool_use whenever the reply has a tool call (vLLM reports 'stop' for a named tool);
- a final assistant turn (Anthropic prefill) is refused: the chat upstream cannot continue it;
- Responses reasoning items in the input are dropped (the upstream is a chat model).
"""
import hashlib
import json

from .kernel.sdk.session import _chunks

PROTOCOLS = ('chat', 'responses', 'anthropic')


class ProtoError(ValueError):
    def __init__(self, detail, status=400, kind='invalid_request'):
        super().__init__(detail)
        self.status, self.kind, self.detail = status, kind, detail


def _hid(*parts, n=32):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:n]


def _sse(event, data):
    head = 'event: %s\n' % event if event else ''
    return '%sdata: %s\n\n' % (head, json.dumps(data))


def _choice(data):
    ch = data['choices'][0]
    return ch['message'], ch.get('finish_reason')


# -- OpenAI Chat Completions --------------------------------------------------------------------

class ChatProto:
    name = 'chat'
    keepalive = b': keep-alive\n\n'
    done = b'data: [DONE]\n\n'

    def __init__(self, req, registry=None):
        self.req = req
        self.chat = dict(req, messages=_chat_messages(req.get('messages')))
        self.stream = bool(req.get('stream'))

    def finish(self, data):
        pass

    def render(self, data):
        return data

    def events(self, data):
        opts = self.req.get('stream_options')
        include = isinstance(opts, dict) and bool(opts.get('include_usage'))
        return ''.join('data: %s\n\n' % json.dumps(c) for c in _chunks(data, include))

    @staticmethod
    def error(status, kind, detail):
        return {'error': {'message': str(detail), 'type': kind, 'code': kind, 'param': None}}

    def stream_error(self, status, kind, detail):
        return 'data: %s\n\n' % json.dumps(self.error(status, kind, detail))


def _chat_messages(messages):
    """Chat messages as the kernel takes them, read as the Responses input is: a content list of
    text parts is its texts joined by newlines (as vLLM joins them), and `developer` is `system`.
    Other parts (images, audio, files) are refused."""
    if not isinstance(messages, list):
        return messages
    out = []
    for m in messages:
        if isinstance(m, dict) and (isinstance(m.get('content'), list) or
                                    m.get('role') == 'developer'):
            m = dict(m)
            if m.get('role') == 'developer':
                m['role'] = 'system'
            if isinstance(m.get('content'), list):
                m['content'] = _parts_text(m['content'], '%s message' % m.get('role'))
        out.append(m)
    return out


# -- OpenAI Responses ---------------------------------------------------------------------------

RESP_DROP = ('store', 'metadata', 'user', 'include', 'service_tier', 'prompt_cache_key',
             'safety_identifier', 'stream_options', 'truncation', 'prompt_cache_retention')
RESP_KEEP = ('model', 'input', 'instructions', 'previous_response_id', 'max_output_tokens',
             'temperature', 'top_p', 'top_k', 'tools', 'tool_choice', 'parallel_tool_calls',
             'text', 'stream', 'reasoning', 'background', 'conversation', 'top_logprobs',
             'extra_body')


def _parts_text(content, where):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ProtoError('%s content is text or a list of parts' % where)
    out = []
    for p in content:
        t = p.get('type') if isinstance(p, dict) else None
        if t in ('input_text', 'output_text', 'text', 'refusal'):
            out.append(p.get('text', p.get('refusal', '')))
        else:
            raise ProtoError('%s part type %r is not served (text only in this version)' % (where, t))
    return '\n'.join(out)


def responses_messages(items):
    """Chat messages of a Responses `input` (a string or a list of items)."""
    if isinstance(items, str):
        return [{'role': 'user', 'content': items}]
    if not isinstance(items, list):
        raise ProtoError('input is a string or a list of items')
    msgs = []
    for it in items:
        if not isinstance(it, dict):
            raise ProtoError('each input item is an object')
        t = it.get('type') or ('message' if 'role' in it else None)
        if t == 'message':
            role = it.get('role')
            if role == 'developer':
                role = 'system'
            if role not in ('system', 'user', 'assistant'):
                raise ProtoError('message role %r' % (role,))
            msgs.append({'role': role, 'content': _parts_text(it.get('content'), 'message')})
        elif t == 'function_call':
            call = {'id': it.get('call_id'), 'type': 'function',
                    'function': {'name': it.get('name'), 'arguments': it.get('arguments', '')}}
            last = msgs[-1] if msgs else None
            if last is not None and last['role'] == 'assistant':
                last.setdefault('tool_calls', []).append(call)
            else:
                msgs.append({'role': 'assistant', 'content': None, 'tool_calls': [call]})
        elif t == 'function_call_output':
            out = it.get('output')
            msgs.append({'role': 'tool', 'tool_call_id': it.get('call_id'),
                         'content': out if isinstance(out, str) else _parts_text(out, 'output')})
        elif t == 'reasoning':
            continue
        else:
            raise ProtoError('input item type %r is not served in this version' % (t,))
    return msgs


class ResponsesProto:
    name = 'responses'
    keepalive = b': keep-alive\n\n'
    done = b''

    def __init__(self, req, registry):
        self.req = req
        self.reg = registry
        unknown = sorted(set(req) - set(RESP_KEEP) - set(RESP_DROP))
        if unknown:
            raise ProtoError('unknown request fields %s' % ', '.join(unknown))
        if req.get('conversation'):
            raise ProtoError('conversations are not served (pass the input or previous_response_id)')
        if req.get('background'):
            raise ProtoError('background responses are not served')
        if req.get('top_logprobs'):
            raise ProtoError('logprobs are not served in this version')
        if req.get('truncation') not in (None, 'disabled'):
            raise ProtoError('truncation %r is not served (the full input is always sent)'
                             % req.get('truncation'))
        if (req.get('text') or {}).get('verbosity') is not None:
            raise ProtoError('text.verbosity is not served in this version')
        reasoning = req.get('reasoning') or {}
        if reasoning.get('effort') or reasoning.get('summary'):
            raise ProtoError('reasoning settings are not served in this version')
        context = []
        prev = req.get('previous_response_id')
        if prev:
            context = registry.response_context(prev)
            if context is None:
                raise ProtoError('previous_response_id %s was not issued by this Copula server' % prev,
                                 404, 'not_found')
        new = responses_messages(req.get('input', []))
        self.context = context + new            # the chain, without instructions (OpenAI rule)
        msgs = list(self.context)
        instr = req.get('instructions')
        if instr:
            msgs.insert(0, {'role': 'system', 'content': _parts_text(instr, 'instructions')})
        if not msgs:
            raise ProtoError('input is empty')
        chat = {'model': req.get('model'), 'messages': msgs}
        if req.get('max_output_tokens') is not None:
            chat['max_tokens'] = req['max_output_tokens']
        for k in ('temperature', 'top_p', 'top_k', 'parallel_tool_calls', 'extra_body'):
            if req.get(k) is not None:            # top_k: the vLLM extension, as in chat
                chat[k] = req[k]
        tools = req.get('tools') or []
        if tools:
            out = []
            for t in tools:
                if t.get('type') != 'function':
                    raise ProtoError('server-side tool %r is not served in this version' % t.get('type'))
                fn = {'name': t.get('name'), 'parameters': t.get('parameters') or {}}
                for k in ('description', 'strict'):
                    if t.get(k) is not None:
                        fn[k] = t[k]
                out.append({'type': 'function', 'function': fn})
            chat['tools'] = out
        tc = req.get('tool_choice')
        if isinstance(tc, dict) and tc.get('type') == 'function':
            chat['tool_choice'] = {'type': 'function', 'function': {'name': tc.get('name')}}
        elif tc in ('auto', 'none', 'required'):
            chat['tool_choice'] = tc
        elif tc is not None:
            raise ProtoError('tool_choice %r is not served in this version' % (tc,))
        fmt = (req.get('text') or {}).get('format') or {}
        if fmt.get('type') == 'json_object':
            chat['response_format'] = {'type': 'json_object'}
        elif fmt.get('type') == 'json_schema':
            js = {k: fmt[k] for k in ('name', 'schema', 'strict', 'description') if k in fmt}
            chat['response_format'] = {'type': 'json_schema', 'json_schema': js}
        elif fmt.get('type') not in (None, 'text'):
            raise ProtoError('text.format type %r is not served' % fmt.get('type'))
        self.stream = bool(req.get('stream'))
        chat['stream'] = self.stream
        self.chat = chat
        self._resp = None

    def finish(self, data):
        """Build the response object and store its chain context (idempotent: the id comes
        from the sample's address)."""
        if self._resp is not None:
            return
        msg, finish = _choice(data)
        # The chat id names the sample; the context is part of the id too, because two requests
        # can make the same chat request (instructions vs a system item) yet differ as chains.
        rid = 'resp_' + _hid('resp', data['id'], self.context, n=40)
        out = []
        calls = msg.get('tool_calls') or []
        if msg.get('content') or not calls:
            out.append({'type': 'message', 'id': 'msg_' + _hid(rid, 'msg'), 'status': 'completed',
                        'role': 'assistant',
                        'content': [{'type': 'output_text', 'text': msg.get('content') or '',
                                     'annotations': [], 'logprobs': []}]})
        for i, c in enumerate(calls):
            out.append({'type': 'function_call', 'id': 'fc_' + _hid(rid, 'fc', i),
                        'call_id': c['id'], 'name': c['function']['name'],
                        'arguments': c['function']['arguments'], 'status': 'completed'})
        u = data.get('usage') or {}
        req = self.req
        incomplete = finish == 'length'
        self._resp = {
            'id': rid, 'object': 'response', 'created_at': int(data.get('created') or 0),
            'status': 'incomplete' if incomplete else 'completed', 'error': None,
            'incomplete_details': {'reason': 'max_output_tokens'} if incomplete else None,
            'instructions': req.get('instructions'), 'max_output_tokens': req.get('max_output_tokens'),
            'model': req.get('model'), 'output': out,
            'parallel_tool_calls': req.get('parallel_tool_calls', True),
            'previous_response_id': req.get('previous_response_id'),
            'reasoning': {'effort': None, 'summary': None}, 'store': req.get('store', True),
            'temperature': req.get('temperature'), 'text': req.get('text') or {'format': {'type': 'text'}},
            'tool_choice': req.get('tool_choice', 'auto'), 'tools': req.get('tools') or [],
            'top_p': req.get('top_p'), 'truncation': 'disabled', 'user': None,
            'metadata': req.get('metadata') or {},
            'usage': {'input_tokens': u.get('prompt_tokens', 0),
                      'input_tokens_details': {'cached_tokens': 0},
                      'output_tokens': u.get('completion_tokens', 0),
                      'output_tokens_details': {'reasoning_tokens': 0},
                      'total_tokens': u.get('total_tokens', 0)}}
        turn = {'role': 'assistant', 'content': msg.get('content')}
        if calls:
            turn['tool_calls'] = [{'id': c['id'], 'type': 'function', 'function': dict(c['function'])}
                                  for c in calls]
        elif turn['content'] is None:
            turn['content'] = ''
        self.reg.put_response(rid, self.context + [turn], self._resp)

    def render(self, data):
        self.finish(data)
        return self._resp

    def events(self, data):
        self.finish(data)
        resp = self._resp
        seq = [0]
        out = []

        def ev(t, **fields):
            fields = dict(type=t, sequence_number=seq[0], **fields)
            seq[0] += 1
            out.append(_sse(t, fields))
        start = dict(resp, status='in_progress', output=[], usage=None, incomplete_details=None)
        ev('response.created', response=start)
        ev('response.in_progress', response=start)
        for i, item in enumerate(resp['output']):
            if item['type'] == 'message':
                text = item['content'][0]['text']
                ev('response.output_item.added', output_index=i,
                   item=dict(item, status='in_progress', content=[]))
                ev('response.content_part.added', item_id=item['id'], output_index=i,
                   content_index=0, part={'type': 'output_text', 'text': '', 'annotations': [],
                                          'logprobs': []})
                if text:
                    ev('response.output_text.delta', item_id=item['id'], output_index=i,
                       content_index=0, delta=text, logprobs=[])
                ev('response.output_text.done', item_id=item['id'], output_index=i,
                   content_index=0, text=text, logprobs=[])
                ev('response.content_part.done', item_id=item['id'], output_index=i,
                   content_index=0, part=item['content'][0])
            else:
                ev('response.output_item.added', output_index=i,
                   item=dict(item, status='in_progress', arguments=''))
                ev('response.function_call_arguments.delta', item_id=item['id'], output_index=i,
                   delta=item['arguments'])
                ev('response.function_call_arguments.done', item_id=item['id'], output_index=i,
                   arguments=item['arguments'], name=item['name'])
            ev('response.output_item.done', output_index=i, item=item)
        ev('response.incomplete' if resp['status'] == 'incomplete' else 'response.completed',
           response=resp)
        return ''.join(out)

    @staticmethod
    def error(status, kind, detail):
        return {'error': {'message': str(detail), 'type': kind, 'code': kind, 'param': None}}

    def stream_error(self, status, kind, detail):
        return _sse('error', {'type': 'error', 'code': kind, 'message': str(detail),
                              'param': None, 'sequence_number': 0})


# -- Anthropic Messages -------------------------------------------------------------------------

ANT_DROP = ('metadata', 'service_tier')
ANT_KEEP = ('model', 'max_tokens', 'messages', 'system', 'temperature', 'top_p', 'top_k',
            'stop_sequences', 'stream', 'tools', 'tool_choice', 'thinking')
ANT_STATUS = {400: 'invalid_request_error', 401: 'authentication_error', 403: 'permission_error',
              404: 'not_found_error', 409: 'invalid_request_error', 410: 'invalid_request_error',
              429: 'rate_limit_error', 500: 'api_error', 502: 'api_error', 503: 'overloaded_error'}
STOP = {'stop': 'end_turn', 'length': 'max_tokens', 'tool_calls': 'tool_use'}


def _blocks_text(blocks, where):
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        raise ProtoError('%s is text or a list of blocks' % where)
    out = []
    for b in blocks:
        if not isinstance(b, dict) or b.get('type') != 'text':
            raise ProtoError('%s block type %r is not served (text only in this version)'
                             % (where, b.get('type') if isinstance(b, dict) else None))
        out.append(b.get('text', ''))
    return '\n'.join(out)


def anthropic_messages(system, messages):
    msgs = []
    if system is not None:
        msgs.append({'role': 'system', 'content': _blocks_text(system, 'system')})
    if not isinstance(messages, list) or not messages:
        raise ProtoError('messages is a non-empty list')
    for m in messages:
        role, content = m.get('role'), m.get('content')
        blocks = [{'type': 'text', 'text': content}] if isinstance(content, str) else content
        if not isinstance(blocks, list):
            raise ProtoError('message content is text or a list of blocks')
        if role == 'user':
            texts = []
            for b in blocks:
                t = b.get('type')
                if t == 'text':
                    texts.append(b.get('text', ''))
                elif t == 'tool_result':
                    msgs.append({'role': 'tool', 'tool_call_id': b.get('tool_use_id'),
                                 'content': _blocks_text(b.get('content', ''), 'tool_result')})
                else:
                    raise ProtoError('user block type %r is not served in this version' % (t,))
            if texts:
                msgs.append({'role': 'user', 'content': '\n'.join(texts)})
        elif role == 'assistant':
            texts, calls = [], []
            for b in blocks:
                t = b.get('type')
                if t == 'text':
                    texts.append(b.get('text', ''))
                elif t == 'tool_use':
                    calls.append({'id': b.get('id'), 'type': 'function',
                                  'function': {'name': b.get('name'),
                                               'arguments': json.dumps(b.get('input') or {})}})
                else:
                    raise ProtoError('assistant block type %r is not served in this version' % (t,))
            turn = {'role': 'assistant', 'content': '\n'.join(texts) if texts else None}
            if calls:
                turn['tool_calls'] = calls
            elif turn['content'] is None:
                turn['content'] = ''
            msgs.append(turn)
        else:
            raise ProtoError('message role %r (user or assistant)' % (role,))
    if messages[-1].get('role') == 'assistant':
        raise ProtoError('a final assistant turn (prefill) is not served in this version: the '
                         'chat upstream would start a new turn instead of continuing it')
    return msgs


class AnthropicProto:
    name = 'anthropic'
    keepalive = b'event: ping\ndata: {"type": "ping"}\n\n'
    done = b''

    def __init__(self, req, registry=None):
        self.req = req
        unknown = sorted(set(req) - set(ANT_KEEP) - set(ANT_DROP))
        if unknown:
            raise ProtoError('unknown request fields %s' % ', '.join(unknown))
        if (req.get('thinking') or {}).get('type') not in (None, 'disabled'):
            raise ProtoError('extended thinking is not served in this version')
        mt = req.get('max_tokens')
        if type(mt) is not int or mt < 1:
            raise ProtoError('max_tokens is required: a positive int')
        chat = {'model': req.get('model'), 'max_tokens': mt,
                'messages': anthropic_messages(req.get('system'), req.get('messages'))}
        for k in ('temperature', 'top_p', 'top_k'):
            if req.get(k) is not None:
                chat[k] = req[k]
        if req.get('stop_sequences'):
            chat['stop'] = list(req['stop_sequences'])
        tools = req.get('tools') or []
        if tools:
            out = []
            for t in tools:
                if t.get('type') not in (None, 'custom'):
                    raise ProtoError('server-side tool %r is not served in this version' % t.get('type'))
                fn = {'name': t.get('name'), 'parameters': t.get('input_schema') or {}}
                if t.get('description') is not None:
                    fn['description'] = t['description']
                out.append({'type': 'function', 'function': fn})
            chat['tools'] = out
        tc = req.get('tool_choice')
        if isinstance(tc, dict):
            kind = tc.get('type')
            if kind == 'auto':
                chat['tool_choice'] = 'auto'
            elif kind == 'any':
                chat['tool_choice'] = 'required'
            elif kind == 'none':
                chat['tool_choice'] = 'none'
            elif kind == 'tool':
                chat['tool_choice'] = {'type': 'function', 'function': {'name': tc.get('name')}}
            else:
                raise ProtoError('tool_choice type %r' % (kind,))
            if tc.get('disable_parallel_tool_use'):
                chat['parallel_tool_calls'] = False
        self.stream = bool(req.get('stream'))
        chat['stream'] = self.stream
        self.chat = chat

    def finish(self, data):
        pass

    def render(self, data):
        msg, finish = _choice(data)
        content = []
        if msg.get('content'):
            content.append({'type': 'text', 'text': msg['content']})
        for c in msg.get('tool_calls') or []:
            try:
                inp = json.loads(c['function']['arguments'] or '{}')
            except ValueError:
                inp = {'_arguments': c['function']['arguments']}
            content.append({'type': 'tool_use', 'id': c['id'], 'name': c['function']['name'],
                            'input': inp})
        if not content:
            content.append({'type': 'text', 'text': ''})
        u = data.get('usage') or {}
        return {'id': 'msg_' + _hid('msg', data['id']), 'type': 'message', 'role': 'assistant',
                'model': self.req.get('model'), 'content': content,
                'stop_reason': ('tool_use' if any(b['type'] == 'tool_use' for b in content)
                                else STOP.get(finish, 'end_turn')), 'stop_sequence': None,
                'usage': {'input_tokens': u.get('prompt_tokens', 0),
                          'output_tokens': u.get('completion_tokens', 0),
                          'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0}}

    def events(self, data):
        m = self.render(data)
        out = [_sse('message_start', {'type': 'message_start', 'message': dict(
            m, content=[], stop_reason=None, stop_sequence=None,
            usage=dict(m['usage'], output_tokens=1))})]
        for i, b in enumerate(m['content']):
            if b['type'] == 'text':
                out.append(_sse('content_block_start', {'type': 'content_block_start', 'index': i,
                                                         'content_block': {'type': 'text', 'text': ''}}))
                if b['text']:
                    out.append(_sse('content_block_delta', {
                        'type': 'content_block_delta', 'index': i,
                        'delta': {'type': 'text_delta', 'text': b['text']}}))
            else:
                out.append(_sse('content_block_start', {
                    'type': 'content_block_start', 'index': i,
                    'content_block': {'type': 'tool_use', 'id': b['id'], 'name': b['name'],
                                      'input': {}}}))
                out.append(_sse('content_block_delta', {
                    'type': 'content_block_delta', 'index': i,
                    'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(b['input'])}}))
            out.append(_sse('content_block_stop', {'type': 'content_block_stop', 'index': i}))
        out.append(_sse('message_delta', {'type': 'message_delta',
                                          'delta': {'stop_reason': m['stop_reason'],
                                                    'stop_sequence': None},
                                          'usage': {'output_tokens': m['usage']['output_tokens']}}))
        out.append(_sse('message_stop', {'type': 'message_stop'}))
        return ''.join(out)

    @staticmethod
    def error(status, kind, detail):
        return {'type': 'error', 'error': {'type': ANT_STATUS.get(status, 'api_error'),
                                           'message': '%s: %s' % (kind, detail)}}

    def stream_error(self, status, kind, detail):
        return _sse('error', self.error(status, kind, detail))


def make(kind, req, registry):
    if kind == 'chat/completions':
        return ChatProto(req, registry)
    if kind == 'responses':
        return ResponsesProto(req, registry)
    if kind == 'messages':
        return AnthropicProto(req, registry)
    raise ProtoError('no such endpoint', 404, 'not_found')


def error_for(path, status, code, detail):
    """An error body in the protocol of the endpoint at `path` (before a request was parsed):
    Anthropic's for /v1/messages and below (count_tokens included), OpenAI's otherwise."""
    if '/v1/messages' in (path or ''):
        return AnthropicProto.error(status, code, detail)
    return ChatProto.error(status, code, detail)
