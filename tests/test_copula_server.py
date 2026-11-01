"""`copula server`: one server per project serves concurrent `copula run`s
(docs/guides/project-server.md).

    PYTHONPATH=src:. python3 -m unittest tests.test_copula_server
"""
import json
import os
import socket
import subprocess
import sys
import threading
import unittest
import urllib.error
import urllib.request

from tests.copula_fixtures.launch import LaunchCase, env_with


class ServerTest(LaunchCase):
    def test_concurrent_runs_share_one_server(self):
        p, ready = self.start_server('--mock', '--mock-latency', '0.2')
        disc = os.path.join(self.dir, '.copula', 'server.json')
        self.assertTrue(os.path.exists(disc))
        self.assertEqual(os.stat(disc).st_mode & 0o777, 0o600)
        with open(disc) as f:
            self.assertEqual(json.load(f)['url'], ready['url'])
        self.write('ask.json', [{'q': 'shared question', 'temperature': 0.7},
                                {'q': 'greedy question', 'temperature': 0}])
        procs = [subprocess.Popen(self.cmd('run', '--project', self.dir, '--exp', 'e',
                                           '--set', 'rep=%d' % i, '--', sys.executable, 'ask.py'),
                                  cwd=self.dir, env=env_with(), stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True) for i in range(3)]
        outs = []
        for q in procs:
            o, e = q.communicate(timeout=120)
            self.assertEqual(q.returncode, 0, e)
            self.assertIn('served by server %s' % ready['url'], e)
            outs.append(json.loads(o.strip().splitlines()[-1]))
        texts = [o[0][1] for o in outs]
        self.assertEqual(len(set(texts)), 3)                     # replicates: separate samples
        self.assertEqual(len({o[1][1] for o in outs}), 1)        # greedy: one global value
        # two compared arms at once share the stochastic sample (one generates, one reads)
        arms = [subprocess.Popen(self.cmd('run', '--project', self.dir, '--exp', 'e',
                                          '--compare', 'arm', '--set', 'arm=%s' % arm, '--',
                                          sys.executable, 'ask.py'),
                                 cwd=self.dir, env=env_with(), stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True) for arm in ('a', 'b')]
        got = []
        for q in arms:
            o, e = q.communicate(timeout=120)
            self.assertEqual(q.returncode, 0, e)
            got.append(json.loads(o.strip().splitlines()[-1]))
        self.assertEqual(got[0][0][1], got[1][0][1])
        self.assertEqual(sorted(s for g in got for s, _ in g).count('generated'), 1)
        # a launcher with its own model server is refused while the server runs
        _, err = self.run_ask([{'q': 'x'}], '--mock', '--exp', 'e', rc=None)
        self.assertIn('decides the model server, so --mock cannot be set here', err)
        self.stop_server(p)
        self.assertFalse(os.path.exists(disc))
        # embedded again once it is gone; the store kept the greedy value
        out, _ = self.run_ask([{'q': 'greedy question', 'temperature': 0}], '--mock', '--exp', 'e')
        self.assertEqual(out[0], ['hit', outs[0][1][1]])

    def test_stale_discovery_file_is_ignored(self):
        os.makedirs(os.path.join(self.dir, '.copula'))
        disc = os.path.join(self.dir, '.copula', 'server.json')
        dead = subprocess.Popen([sys.executable, '-c', 'pass'])
        dead.wait()
        with open(disc, 'w') as f:
            json.dump({'url': 'http://127.0.0.1:9', 'token': 't', 'pid': dead.pid,
                       'host': socket.gethostname()}, f)
        out, err = self.run_ask([{'q': 'x'}], '--mock')
        self.assertIn('embedded server', err)
        self.assertFalse(os.path.exists(disc))

    def test_second_embedded_launcher_gets_a_hint(self):
        self.write('slow.py', 'import time\ntime.sleep(12)\n')
        first = subprocess.Popen(self.cmd('run', '--mock', '--project', self.dir, '--',
                                          sys.executable, 'slow.py'), cwd=self.dir,
                                 env=env_with(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True)
        self.procs.append(first)
        self.assertTrue(self.wait_for(lambda: os.path.exists(
            os.path.join(self.dir, '.copula', 'registry.db')), 20))
        _, err = self.run_ask([{'q': 'x'}], '--mock', rc=None)
        self.assertIn('copula server', err)
        self.assertIn('another Copula process holds', err)

    def test_control_needs_the_token_and_remote_flag_works(self):
        p, ready = self.start_server('--mock')
        req = urllib.request.Request(ready['url'] + '/copula/v1/health',
                                     headers={'Authorization': 'Bearer wrong'})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code, 401)
        with open(os.path.join(self.dir, '.copula', 'server.json')) as f:
            token = json.load(f)['token']
        other = os.path.join(self.dir, 'elsewhere')       # a launcher outside the project
        os.makedirs(other)
        self.write('ask.json', [{'q': 'remote question'}])
        q = subprocess.run(self.cmd('run', '--project', other, '--server', ready['url'], '--',
                                    sys.executable, os.path.join(self.dir, 'ask.py')),
                           cwd=self.dir, env=env_with(COPULA_SERVER_TOKEN=token),
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(q.returncode, 0, q.stderr)
        self.assertIn('done: 1 calls', q.stderr)
        q2 = subprocess.run(self.cmd('run', '--project', other, '--server', ready['url'], '--',
                                     sys.executable, 'ask.py'), cwd=self.dir, env=env_with(),
                            capture_output=True, text=True, timeout=60)
        self.assertNotEqual(q2.returncode, 0)
        self.assertIn('COPULA_SERVER_TOKEN', q2.stderr)
        self.stop_server(p)


if __name__ == '__main__':
    unittest.main()
