"""Copula with real clients, unmodified, under `copula run` and the embedded mock manager: the
openai package (plain, raw, streamed, n>1, timeout retries) and DSPy's LM. Skipped where the
packages are missing; COPULA_DSPY_PYTHON and COPULA_ANTHROPIC_PYTHON name interpreters that have
DSPy or anthropic:
    PYTHONPATH=src:. python -m unittest tests.test_copula_clients
    COPULA_DSPY_PYTHON=/path/to/python   (for the DSPy test)
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
HAS_OPENAI = importlib.util.find_spec('openai') is not None
DSPY_PY = os.environ.get('COPULA_DSPY_PYTHON')
ANTHROPIC_PY = os.environ.get('COPULA_ANTHROPIC_PYTHON')
SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'copula_fixtures',
                       'mock_scripts.py')

OPENAI_AGENT = r'''
import json, os
from openai import OpenAI
c = OpenAI()                                   # OPENAI_BASE_URL and OPENAI_API_KEY from copula run
m = os.environ['COPULA_MODEL']
def msgs(t):
    return [{'role': 'user', 'content': t}]
out = []
for q in (open('prompt.txt').read().strip(), 'a fixed question'):
    r = c.chat.completions.with_raw_response.create(model=m, messages=msgs(q), temperature=0.7,
                                                    max_tokens=8)
    out.append([r.headers['x-copula-status'], r.parse().choices[0].message.content])
s = c.chat.completions.create(model=m, messages=msgs('stream q'), temperature=0.7, stream=True,
                              stream_options={'include_usage': True})
out.append(['stream', ''.join(ch.choices[0].delta.content or '' for ch in s if ch.choices)])
r = c.chat.completions.create(model=m, messages=msgs('n q'), temperature=0.7, n=2)
out.append(['n', [ch.message.content for ch in r.choices]])
print(json.dumps(out))
'''

RETRY_AGENT = r'''
import json, os
from openai import OpenAI
c = OpenAI(timeout=0.3, max_retries=3)
r = c.chat.completions.with_raw_response.create(
    model=os.environ['COPULA_MODEL'], temperature=0.7,
    messages=[{'role': 'user', 'content': 'slow question'}])
print(json.dumps([[r.headers['x-copula-status'], r.parse().choices[0].message.content]]))
'''

DSPY_AGENT = r'''
import json, os
import dspy
lm = dspy.LM('openai/' + os.environ['COPULA_MODEL'], api_base=os.environ['OPENAI_BASE_URL'],
             api_key='copula', cache=False, temperature=0.7, max_tokens=32)
out = [lm(open('prompt.txt').read().strip())[0], lm('a fixed question')[0]]
print(json.dumps(out))
'''


RESPONSES_AGENT = r'''
import json, os
from openai import OpenAI
c = OpenAI()
m = os.environ['COPULA_MODEL']
out = {}
r1 = c.responses.with_raw_response.create(model=m, input=open('prompt.txt').read().strip(),
                                          instructions='be brief', temperature=0.7)
p1 = r1.parse()
r2 = c.responses.create(model=m, input='and the second step?', previous_response_id=p1.id)
out['first'] = [r1.headers['x-copula-status'], p1.output_text, p1.id]
out['second'] = [r2.output_text, r2.id]
with c.responses.stream(model=m, input='a fixed question', temperature=0.7) as st:
    text = ''.join(e.delta for e in st if e.type == 'response.output_text.delta')
    final = st.get_final_response()
out['stream'] = [text, final.output_text, final.id]
tools = [{'type': 'function', 'name': 'look', 'parameters': {'type': 'object', 'properties': {}}}]
t1 = c.responses.create(model=m, input='use the tool', tools=tools)
call = [o for o in t1.output if o.type == 'function_call'][0]
t2 = c.responses.create(model=m, tools=tools, previous_response_id=t1.id, input=[
    {'type': 'function_call_output', 'call_id': call.call_id, 'output': '42'}])
out['tool'] = [call.name, t2.output_text]
print(json.dumps(out))
'''

ANTHROPIC_AGENT = r'''
import json, os
import anthropic
c = anthropic.Anthropic(timeout=float(os.environ.get('T', '600')), max_retries=4)
m = os.environ['COPULA_MODEL']
out = {}
r = c.messages.with_raw_response.create(model=m, max_tokens=64, system='be brief',
                                        messages=[{'role': 'user', 'content': open('prompt.txt').read().strip()}])
msg = r.parse()
out['first'] = [r.headers['x-copula-status'], msg.content[0].text, msg.stop_reason]
r = c.messages.with_raw_response.create(model=m, max_tokens=64, extra_body={'temperature': 0.7},
                                        messages=[{'role': 'user', 'content': 'a fixed question'}])
out['fixed'] = [r.headers['x-copula-status'], r.parse().content[0].text]
with c.messages.stream(model=m, max_tokens=64, messages=[{'role': 'user', 'content': 'stream q'}]) as st:
    text = ''.join(st.text_stream)
    final = st.get_final_message()
out['stream'] = [text, final.content[0].text]
tools = [{'name': 'look', 'description': 'look it up', 'input_schema': {'type': 'object'}}]
t1 = c.messages.create(model=m, max_tokens=64, tools=tools,
                       messages=[{'role': 'user', 'content': 'use the tool'}])
use = [b for b in t1.content if b.type == 'tool_use'][0]
t2 = c.messages.create(model=m, max_tokens=64, tools=tools, messages=[
    {'role': 'user', 'content': 'use the tool'}, {'role': 'assistant', 'content': t1.content},
    {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': use.id, 'content': '42'}]}])
out['tool'] = [t1.stop_reason, use.name, t2.content[0].text]
print(json.dumps(out))
'''

ANTHROPIC_RETRY = r'''
import json, os
import anthropic
c = anthropic.Anthropic(timeout=0.3, max_retries=4)
r = c.messages.with_raw_response.create(model=os.environ['COPULA_MODEL'], max_tokens=32,
                                        messages=[{'role': 'user', 'content': 'slow question'}])
print(json.dumps([r.headers['x-copula-status'], r.parse().content[0].text]))
'''

# An unmodified DSPy program (no copula import): GEPA on a toy yes/no task, then a held-out
# evaluation and one on the validation set.
GEPA_AGENT = r'''
import json, os, random
import dspy
lm = dspy.LM('openai/' + os.environ['COPULA_MODEL'], api_base=os.environ['OPENAI_BASE_URL'],
             api_key='x', temperature=0.7, max_tokens=64)
dspy.configure(lm=lm)
rng = random.Random(0)
def data(prefix, n):
    return [dspy.Example(question='%s question %d' % (prefix, i),
                         answer=rng.choice(['yes', 'no'])).with_inputs('question') for i in range(n)]
train, val, test = data('train', 6), data('val', 4), data('test', 5)
def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return float((pred.answer or '').strip().lower() == gold.answer)
prog = dspy.Predict('question -> answer')
opt = dspy.GEPA(metric=metric, reflection_lm=lm, max_metric_calls=40, seed=0, num_threads=2,
                reflection_minibatch_size=2, track_stats=False)
best = opt.compile(prog, trainset=train, valset=val)
held = dspy.Evaluate(devset=test, metric=metric, num_threads=2)(best)
again = dspy.Evaluate(devset=val, metric=metric, num_threads=2)(best)
print(json.dumps({'held_out': float(held.score), 'val': float(again.score),
                  'instructions': best.signature.instructions}))
'''


def explain_all(project, run):
    """The registry records of a process run and its units (copula explain)."""
    env = dict(os.environ, PYTHONPATH=SRC)
    p = subprocess.run([sys.executable, '-m', 'copula', 'explain', run, '--json', '--project', project],
                       env=env, capture_output=True, text=True, timeout=120)
    return [json.loads(l) for l in p.stdout.splitlines() if l.strip()]


class _Launch(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='copula-clients-')

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, name, text):
        with open(os.path.join(self.dir, name), 'w') as f:
            f.write(text)

    def copula(self, child, *args, python=sys.executable, env_extra=None):
        env = dict(os.environ, PYTHONPATH=SRC, **(env_extra or {}))
        cmd = [sys.executable, '-m', 'copula', 'run', '--mock', '--project', self.dir] + list(args) + [
            '--', python, child]
        p = subprocess.run(cmd, cwd=self.dir, env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return json.loads(p.stdout.strip().splitlines()[-1]), p.stderr


@unittest.skipUnless(HAS_OPENAI, 'openai is not installed')
class OpenAIClientTest(_Launch):
    def test_plain_raw_stream_and_n(self):
        self.write('agent.py', OPENAI_AGENT)
        self.write('prompt.txt', 'first')
        out1, err1 = self.copula('agent.py', '--exp', 'o')
        self.assertEqual([out1[0][0], out1[1][0]], ['generated', 'generated'])
        self.assertTrue(out1[2][1].startswith('mock:'))
        self.assertEqual(len(set(out1[3][1])), 2)                   # n=2: two samples
        self.write('prompt.txt', 'second')
        out2, err2 = self.copula('agent.py', '--exp', 'o')
        self.assertEqual([out2[0][0], out2[1][0]], ['generated', 'hit'])
        self.assertEqual(out2[1][1], out1[1][1])
        self.assertEqual(out2[2][1], out1[2][1])                    # the stream is shared too
        self.assertEqual(out2[3][1], out1[3][1])
        self.assertIn('4 reused, 1 generated', err2)                # 1 + stream + 2 of n

    def test_timeout_retry_rereads_its_occurrence(self):
        self.write('agent.py', RETRY_AGENT)
        out, err = self.copula('agent.py', '--exp', 'r', '--mock-latency', '0.8')
        self.assertIn('retry re-read', out[0][0])
        self.assertIn('0 reused, 1 generated', err)                 # one sample, no waste


@unittest.skipUnless(HAS_OPENAI, 'openai is not installed')
class ResponsesClientTest(_Launch):
    def test_responses_plain_chain_stream_tools(self):
        self.write('agent.py', RESPONSES_AGENT)
        self.write('prompt.txt', 'first')
        mock = ['--mock-script', SCRIPTS + ':tools_reply']
        out1, _ = self.copula('agent.py', '--exp', 'resp', *mock)
        self.assertEqual(out1['first'][0], 'generated')
        self.assertEqual(out1['stream'][0], out1['stream'][1])     # deltas add up to the text
        self.assertEqual(out1['tool'][0], 'look')
        self.assertTrue(out1['tool'][1].startswith('final answer'))
        self.write('prompt.txt', 'second')                         # an edit of the first input
        out2, err2 = self.copula('agent.py', '--exp', 'resp', *mock)
        self.assertEqual(out2['first'][0], 'generated')
        self.assertNotEqual(out2['first'][2], out1['first'][2])
        self.assertEqual(out2['stream'], out1['stream'])            # unchanged: shared, same id
        self.assertEqual(out2['tool'], out1['tool'])


@unittest.skipUnless(ANTHROPIC_PY and os.path.exists(ANTHROPIC_PY), 'set COPULA_ANTHROPIC_PYTHON')
class AnthropicClientTest(_Launch):
    def test_messages_plain_stream_tools(self):
        self.write('agent.py', ANTHROPIC_AGENT)
        self.write('prompt.txt', 'first')
        mock = ['--mock-script', SCRIPTS + ':tools_reply']
        out1, _ = self.copula('agent.py', '--exp', 'ant', *mock, python=ANTHROPIC_PY)
        self.assertEqual([out1['first'][0], out1['fixed'][0]], ['generated', 'generated'])
        self.assertEqual(out1['stream'][0], out1['stream'][1])
        self.assertEqual(out1['tool'][:2], ['tool_use', 'look'])
        self.write('prompt.txt', 'second')
        out2, _ = self.copula('agent.py', '--exp', 'ant', *mock, python=ANTHROPIC_PY)
        self.assertEqual([out2['first'][0], out2['fixed'][0]], ['generated', 'hit'])
        self.assertEqual(out2['fixed'][1], out1['fixed'][1])
        self.assertEqual((out2['stream'], out2['tool']), (out1['stream'], out1['tool']))

    def test_timeout_retry_rereads_its_occurrence(self):
        self.write('agent.py', ANTHROPIC_RETRY)
        out, err = self.copula('agent.py', '--exp', 'r', '--mock-latency', '0.8',
                               python=ANTHROPIC_PY)
        self.assertIn('retry re-read', out[0])
        self.assertIn('0 reused, 1 generated', err)


@unittest.skipUnless(DSPY_PY and os.path.exists(DSPY_PY), 'set COPULA_DSPY_PYTHON')
class GEPAUnmodifiedTest(_Launch):
    def run_gepa(self):
        out, err = self.copula('agent.py', '--exp', 'gepa', '--mock-script',
                               SCRIPTS + ':dspy_reply', python=DSPY_PY)
        run = err.split('run ', 1)[1].split()[0]
        return out, err, explain_all(self.dir, run)

    def test_gepa_under_copula_run(self):
        self.write('agent.py', GEPA_AGENT)
        out1, err1, rec1 = self.run_gepa()
        for msg in ('DSPy response cache turned off', 'DSPy adapter on', 'GEPA adapter on',
                    'W5b'):
            self.assertIn(msg, err1)
        units = [r for r in rec1 if r['granularity'] == 'unit']
        self.assertGreater(len(units), 20)
        cands = {u['classes']['compare'].get('candidate') for u in units}
        self.assertGreater(len(cands), 1)                           # GEPA proposed new programs
        self.assertEqual({u['classes']['replicate'].get('optimizer_seed') for u in units
                          if u['phase'] == 'search'} - {None}, {'0'})
        hold = [u for u in units if u['phase'] == 'holdout']
        self.assertEqual(len({u['classes']['case']['case'] for u in hold}), 5)   # the test set
        paired = [u for u in units if 'decision' in u['rule']]
        groups = {}
        for u in paired:
            groups.setdefault(u['group'], set()).add(u['classes']['compare']['candidate'])
        self.assertTrue(any(len(c) > 1 for c in groups.values()))   # parent and child paired
        # A cosmetic edit: a new version; GEPA replays the same trajectory from stored samples.
        self.write('agent.py', GEPA_AGENT + '\n# a comment\n')
        out2, err2, _ = self.run_gepa()
        self.assertEqual(out2, out1)
        self.assertRegex(err2, r'done: \d+ calls: \d+ reused, 0 generated')
        # An unchanged rerun: new samples.
        out3, err3, _ = self.run_gepa()
        self.assertIn('unchanged since run', err3)
        self.assertNotRegex(err3, r'done: \d+ calls: \d+ reused, 0 generated')


@unittest.skipUnless(DSPY_PY and os.path.exists(DSPY_PY), 'set COPULA_DSPY_PYTHON')
class DSPyTest(_Launch):
    def test_dspy_lm_shares_unchanged_calls(self):
        self.write('agent.py', DSPY_AGENT)
        self.write('prompt.txt', 'first')
        out1, _ = self.copula('agent.py', '--exp', 'd', python=DSPY_PY)
        self.write('prompt.txt', 'second')
        out2, err2 = self.copula('agent.py', '--exp', 'd', python=DSPY_PY)
        self.assertEqual(out2[1], out1[1])
        self.assertNotEqual(out2[0], out1[0])
        self.assertIn('1 reused, 1 generated', err2)


if __name__ == '__main__':
    unittest.main()
