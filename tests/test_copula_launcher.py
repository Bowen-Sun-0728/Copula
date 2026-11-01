"""Copula preliminary implementation: `copula run` end to end with the embedded mock manager
(reports/opensource_20261022/DESIGN.md §4.3, §10.2). A child Python program reads a prompt file
and asks two questions through OPENAI_BASE_URL; no GPU:
    PYTHONPATH=src:. python3 -m unittest tests.test_copula_launcher
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')

AGENT = r'''
import json, sys
import copula
prompt = open('prompt.txt').read().strip()
out = []
for q in (prompt, 'a fixed question'):
    r = copula.chat([{'role': 'user', 'content': q}], temperature=0.7, max_tokens=8)
    out.append([r['_copula']['x-copula-status'], r['choices'][0]['message']['content']])
with copula.run(case='c1'):
    r = copula.chat([{'role': 'user', 'content': 'unit question'}], temperature=0.7)
    out.append([r['_copula']['x-copula-status'], r['choices'][0]['message']['content']])
print(json.dumps(out))
'''

PLAIN = r'''
import json, os, urllib.request
body = {'model': os.environ['COPULA_MODEL'], 'temperature': 0.5,
        'messages': [{'role': 'user', 'content': open('q.txt').read()}]}
req = urllib.request.Request(os.environ['OPENAI_BASE_URL'] + '/chat/completions',
                             data=json.dumps(body).encode(),
                             headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req) as r:
    print(json.dumps([[r.headers['x-copula-status'], json.load(r)['choices'][0]['message']['content']]]))
'''


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='copula-launch-')
        self.write('agent.py', AGENT)
        self.write('prompt.txt', 'first prompt')

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, name, text):
        with open(os.path.join(self.dir, name), 'w') as f:
            f.write(text)

    def copula(self, *args, child=('agent.py',)):
        env = dict(os.environ, PYTHONPATH=SRC)
        cmd = [sys.executable, '-m', 'copula', 'run', '--mock', '--project', self.dir] + list(args) + [
            '--', sys.executable] + list(child)
        p = subprocess.run(cmd, cwd=self.dir, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 0, p.stderr)
        return json.loads(p.stdout.strip().splitlines()[-1]), p.stderr

    def test_rerun_edit_replicate(self):
        out1, err1 = self.copula('--exp', 'e')
        self.assertEqual([s for s, _ in out1], ['generated'] * 3)
        self.assertIn('new version', err1)
        out2, err2 = self.copula('--exp', 'e')                       # unchanged: new samples
        self.assertIn('unchanged since run', err2)
        self.assertEqual([s for s, _ in out2], ['generated'] * 3)
        self.assertNotEqual(out1[0][1], out2[0][1])
        self.write('prompt.txt', 'second prompt')                   # an edit of a file it reads
        out3, err3 = self.copula('--exp', 'e')
        self.assertIn('new version', err3)
        self.assertEqual([s for s, _ in out3], ['generated', 'hit', 'hit'])
        self.assertEqual(out3[1][1], out1[1][1])                    # the unchanged call is shared
        self.assertEqual(out3[2][1], out1[2][1])                    # and so is the unit's
        self.assertIn('2 reused, 1 generated', err3)
        out4, _ = self.copula('--exp', 'e', '--set', 'seed=2')      # a replicate: separate
        self.assertEqual([s for s, _ in out4], ['generated'] * 3)

    def test_no_experiment_shares_nothing(self):
        out1, _ = self.copula()
        out2, _ = self.copula()
        self.assertEqual([s for s, _ in out1 + out2], ['generated'] * 6)

    def test_annotation_argument_is_a_rerun_and_explain(self):
        self.write('q.txt', 'hello')
        self.write('plain.py', PLAIN)
        self.copula('--exp', 'p', child=('plain.py', '-o', 'out/a'))
        out2, err2 = self.copula('--exp', 'p', child=('plain.py', '-o', 'out/b'))
        self.assertIn('unchanged since run', err2)                  # -o is an annotation
        self.assertEqual(out2[0][0], 'generated')
        run = err2.split('run ', 1)[1].split()[0]
        env = dict(os.environ, PYTHONPATH=SRC)
        p = subprocess.run([sys.executable, '-m', 'copula', 'explain', run, '--json', '--project', self.dir],
                           env=env, capture_output=True, text=True, timeout=60)
        info = json.loads(p.stdout.splitlines()[0])
        self.assertEqual((info['slot'], info['classes']['annotation']), (2, {'o': 'out/b'}))
        self.assertGreaterEqual(info['files_read'], 2)              # plain.py and q.txt


    def test_failed_command_keeps_partial_reads_as_unknown(self):
        self.write('crash.py', "import os, copula\n"
                   "copula.chat([{'role': 'user', 'content': open('prompt.txt').read()}])\n"
                   "os._exit(3)\n")
        env = dict(os.environ, PYTHONPATH=SRC)
        p = subprocess.run([sys.executable, '-m', 'copula', 'run', '--mock', '--project', self.dir, '--exp',
                            'c', '--', sys.executable, 'crash.py'], cwd=self.dir, env=env,
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 3, p.stderr)
        self.assertIn('recorded as unknown', p.stderr)
        run = p.stderr.split('run ', 1)[1].split()[0]
        q = subprocess.run([sys.executable, '-m', 'copula', 'explain', run, '--json', '--project', self.dir],
                           env=env, capture_output=True, text=True, timeout=60)
        info = json.loads(q.stdout.splitlines()[0])
        self.assertEqual(info['status'], 'unknown')
        self.assertGreaterEqual(info['files_read'], 1)            # written as it went

if __name__ == '__main__':
    unittest.main()
