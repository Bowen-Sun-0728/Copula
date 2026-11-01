"""Scripted replies for copula.testing.ScriptedMock (launcher --mock-script FILE:FUNC).

dspy_reply: answers in DSPy's ChatAdapter format for whatever output fields the system prompt
asks for (values drawn from the sample's token, so samples differ); a prompt without output
fields (GEPA's reflection prompt) gets a new instruction in a code fence.
tools_reply: calls the first tool offered once; after a tool result, answers in text.
"""
import json
import re

FIELDS = re.compile(r'Your output fields are:(.*?)(?:All interactions|\Z)', re.S)


def dspy_reply(body, token):
    msgs = body.get('messages') or []
    system = next((m['content'] for m in msgs if m['role'] == 'system'), '') or ''
    m = FIELDS.search(system)
    names = re.findall(r'`(\w+)`', m.group(1)) if m else []
    if not names:
        return '```\nAnswer with one word, yes or no. (variant %s)\n```' % token[-6:]
    out = []
    for n in names:
        if n == 'answer':
            value = 'yes' if int(token[-1], 16) % 2 else 'no'
        else:
            value = 'thinking %s' % token[-8:]
        out.append('[[ ## %s ## ]]\n%s\n' % (n, value))
    return '\n'.join(out) + '\n[[ ## completed ## ]]'


def tools_reply(body, token):
    msgs = body.get('messages') or []
    tools = body.get('tools') or []
    if tools and msgs and msgs[-1]['role'] != 'tool':
        name = tools[0]['function']['name']
        return {'content': '', 'finish_reason': 'tool_calls',
                'tool_calls': [{'id': 'call_%s' % token[-10:], 'type': 'function',
                                'function': {'name': name,
                                             'arguments': json.dumps({'q': token[-4:]})}}]}
    return 'final answer %s' % token[-6:]
