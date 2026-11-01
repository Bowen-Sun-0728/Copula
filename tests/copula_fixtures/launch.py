"""Helpers for tests that drive the `copula` command in a temporary project."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, 'src')
ASK = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ask.py')


def env_with(**extra):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('OPENAI_', 'ANTHROPIC_', 'COPULA_'))}
    env['PYTHONPATH'] = SRC
    env.update({k: v for k, v in extra.items() if v is not None})
    return env


class LaunchCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='copula-t-')
        shutil.copy(ASK, os.path.join(self.dir, 'ask.py'))
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            for f in (p.stdout, p.stderr):
                if f is not None:
                    f.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, name, text):
        with open(os.path.join(self.dir, name), 'w') as f:
            f.write(text if isinstance(text, str) else json.dumps(text))

    def cmd(self, *args):
        return [sys.executable, '-m', 'copula'] + list(args)

    def run_ask(self, items, *args, env=None, rc=0, child=None):
        """`copula run ARGS -- python ask.py` with ask.json = items: (out, stderr)."""
        self.write('ask.json', items)
        child = child or [sys.executable, 'ask.py']
        p = subprocess.run(self.cmd('run', '--project', self.dir, *args, '--', *child),
                           cwd=self.dir, env=env or env_with(), capture_output=True, text=True,
                           timeout=180)
        if rc is not None:
            self.assertEqual(p.returncode, rc, p.stderr)
        lines = p.stdout.strip().splitlines()
        return (json.loads(lines[-1]) if lines and p.returncode == 0 else None), p.stderr

    def start_server(self, *args, env=None):
        p = subprocess.Popen(self.cmd('server', '--project', self.dir, *args), cwd=self.dir,
                             env=env or env_with(), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
        self.procs.append(p)
        line = p.stdout.readline()
        if not line:
            self.fail('server did not start: %s' % p.stderr.read())
        return p, json.loads(line)

    def stop_server(self, p):
        p.terminate()
        p.wait(timeout=30)

    def explain(self, run, *extra):
        p = subprocess.run(self.cmd('explain', run, '--project', self.dir, *extra), cwd=self.dir,
                           env=env_with(), capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout

    @staticmethod
    def run_id(err):
        for line in err.splitlines():
            if line.startswith('copula: run ') and ' exp=' in line:
                return line.split()[2]
        raise AssertionError('no run id in: %s' % err)

    def wait_for(self, cond, timeout=10.0):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return True
            time.sleep(0.05)
        return False
