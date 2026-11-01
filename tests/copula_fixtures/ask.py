"""A child program for launcher tests: sends the requests listed in ask.json, in order, each
{"q": text, "coords": "k=v;...", "unit": n, "model": name, "temperature": t}, and prints
[[status, content], ...] as JSON."""
import json
import os
import urllib.request

out = []
for item in json.load(open('ask.json')):
    body = {'model': item.get('model') or os.environ.get('COPULA_MODEL') or 'mock',
            'messages': [{'role': 'user', 'content': item['q']}]}
    for k in ('temperature', 'top_k', 'seed', 'max_tokens'):
        if k in item:
            body[k] = item[k]
    headers = {'Content-Type': 'application/json'}
    if item.get('coords'):
        headers['X-Copula-Coords'] = item['coords']
    if item.get('unit'):
        headers['X-Copula-Unit'] = str(item['unit'])
    req = urllib.request.Request(os.environ['OPENAI_BASE_URL'] + '/chat/completions',
                                 data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req) as r:
        out.append([r.headers['x-copula-status'],
                    json.load(r)['choices'][0]['message']['content']])
print(json.dumps(out))
