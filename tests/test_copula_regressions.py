"""Regression tests of the round-3 review (reports/opensource_20261022/pkg/README.md, findings).

    PYTHONPATH=src:. python3 -m unittest tests.test_copula_regressions
"""
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

from copula import upstream as up
from copula.registry import Registry
from tests.copula_fixtures.fake_upstream import FakeUpstream
from tests.copula_fixtures.launch import LaunchCase, env_with


def dead_pid():
    p = subprocess.Popen([sys.executable, '-c', 'pass'])
    p.wait()
    return p.pid


class VersionTest(unittest.TestCase):
    """Findings 1 and 7: whatever is not known counts as unchanged (a rerun, new samples)."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.reg = Registry(os.path.join(self.dir, 'r.db'))

    def tearDown(self):
        self.reg.close()

    def test_unreadable_file_is_unknown_not_an_edit(self):
        r1 = self.reg.process_run('e', coords={})
        self.reg.finish(r1['run_id'], {'/other-host-only/agent.py': 'abc'}, True)
        r2 = self.reg.process_run('e', coords={})
        self.assertEqual((r2['rerun_of'], r2['match'], r2['slot']), (r1['run_id'], 'unchanged', 2))

    def test_launcher_digests_decide(self):
        r1 = self.reg.process_run('e', coords={}, digest_now=lambda p: 'x')
        self.reg.finish(r1['run_id'], {'/p/a.py': 'd1'}, True)
        self.assertEqual(self.reg.version_paths('e', {}), ['/p/a.py'])
        same = self.reg.process_run('e', coords={}, digest_now={'/p/a.py': 'd1'}.get)
        self.assertEqual(same['rerun_of'], r1['run_id'])
        self.reg.finish(same['run_id'], {'/p/a.py': 'd1'}, True)
        edit = self.reg.process_run('e', coords={}, digest_now={'/p/a.py': 'd2'}.get)
        self.assertIsNone(edit['rerun_of'])
        self.assertEqual(edit['slot'], 1)

    def test_running_run_of_a_dead_launcher_becomes_unknown(self):
        r1 = self.reg.process_run('e', coords={},
                                  owner={'host': socket.gethostname(), 'pid': dead_pid()})
        r2 = self.reg.process_run('e', coords={})
        self.assertEqual((r2['rerun_of'], r2['match']), (r1['run_id'], 'unknown'))
        self.assertEqual(self.reg.get(r1['run_id'])['status'], 'unknown')
        live = self.reg.process_run('f', coords={},
                                    owner={'host': socket.gethostname(), 'pid': os.getpid()})
        r3 = self.reg.process_run('f', coords={})
        self.assertEqual((r3['rerun_of'], r3['match']), (live['run_id'], 'running'))


class IdentityTest(unittest.TestCase):
    """Finding 2: the identity covers every weight file and only files the server loaded."""

    def model(self, shards=2):
        d = tempfile.mkdtemp()
        for name in ('config.json', 'generation_config.json'):
            with open(os.path.join(d, name), 'w') as f:
                json.dump({'temperature': 0.6}, f)
        for i in range(shards):
            with open(os.path.join(d, 'model-%05d-of-%05d.safetensors' % (i + 1, shards)),
                      'wb') as f:
                f.write(b'w' * (10 + i))
        return d

    def test_resaved_shard_is_a_new_identity(self):
        d = self.model()
        a = up.model_identity(d)
        self.assertEqual(len(a['weights']), 2)
        shard = os.path.join(d, 'model-00002-of-00002.safetensors')
        st = os.stat(shard)
        os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
        self.assertNotEqual(up.model_identity(d), a)

    def test_only_a_local_directory_is_exact(self):
        d = self.model()
        self.assertEqual(up.locate(d, True), (d, True))
        self.assertEqual(up.locate(d, False), (d, False))       # a server on another host
        self.assertEqual(up.locate('org/absent-model', True), (None, False))

    def test_lora_takes_the_base_table_and_needs_its_adapter(self):
        base = self.model()
        adapter = tempfile.mkdtemp()
        with open(os.path.join(adapter, 'adapter_config.json'), 'w') as f:
            json.dump({'r': 8}, f)
        fake = FakeUpstream('vllm', models=('base', 'lora'), roots={'base': base, 'lora': adapter},
                            parents={'lora': 'base'})
        try:
            ns = type('A', (), dict(upstream=fake.url, served_model=None, model_dir=None,
                                    defaults=None, identity_extra=None, engine_seed='auto',
                                    api_key_env=None, model=[], timeout=30.0, workers=2,
                                    wait_ready=0))()
            b = up.build(ns, 'secret', {})
            lora = b['models'][b['names']['lora']]
            self.assertEqual(lora.defaults['temperature'], 0.6)        # the base's table
            self.assertTrue(b['identity']['actor'])
            self.assertTrue(b['identity'][b['names']['lora']])
            os.unlink(os.path.join(adapter, 'adapter_config.json'))
            b2 = up.build(ns, 'secret', {})
            self.assertFalse(b2['identity'][b2['names']['lora']])
        finally:
            fake.close()


class ApiPayloadTest(unittest.TestCase):
    """Finding 3: an explicit default and an absent parameter share a key and are sent alike."""

    def client(self):
        return up.ApiClient('http://x/v1', 'gpt', dict(up.API_DEFAULTS), {}, api_key='k')

    def test_explicit_defaults_are_sent(self):
        c = self.client()
        body = {'messages': [{'role': 'user', 'content': 'q'}], 'max_tokens': 5,
                **up.API_DEFAULTS}
        p = c.payload_for(body, 'k', None)
        self.assertEqual((p['temperature'], p['top_p']), (1.0, 1.0))
        self.assertNotIn('top_k', p)
        self.assertNotIn('stop', p)
        p2 = c.payload_for(dict(body, top_k=5, temperature=0.2), 'k', None)
        self.assertEqual((p2['top_k'], p2['temperature']), (5, 0.2))

    def test_max_completion_tokens_switch(self):
        c = self.client()
        fixed = c.retry_payload(400, "Use 'max_completion_tokens' instead.", {'max_tokens': 5})
        self.assertEqual(fixed, {'max_completion_tokens': 5})
        self.assertIn('max_completion_tokens',
                      c.payload_for({'messages': [], 'max_tokens': 5}, 'k', None))


class LaunchRegressionTest(LaunchCase):
    def test_misconfiguration_is_not_stored_and_explained(self):
        fake = FakeUpstream('api', models=('gpt-x',), key='sk-right', open_listing=True)
        try:
            env = env_with(OPENAI_BASE_URL=fake.url, OPENAI_API_KEY='sk-wrong')
            fast = ('--max-attempts', '2', '--backoff', '0.01')
            _, err = self.run_ask([{'q': 'hello'}], '--exp', 'a', *fast, env=env, rc=None)
            self.assertIn('1 failed', err)                           # finding 9
            db = sqlite3.connect(os.path.join(self.dir, '.copula', 'entries.db'))
            self.assertEqual(db.execute('SELECT COUNT(*) FROM entries').fetchone()[0], 0)
            db.close()
            fake.key = 'sk-wrong'                                    # the key is fixed
            out, err2 = self.run_ask([{'q': 'hello'}], '--exp', 'a', '--set', 'arm=b', *fast,
                                     env=env)
            self.assertEqual(out[0][0], 'generated')
            self.assertIn('earlier calls failed', err2)
        finally:
            fake.close()

    def test_max_completion_tokens_upstream(self):
        fake = FakeUpstream('api', models=('o-x',), completion_tokens=True)
        try:
            env = env_with(OPENAI_BASE_URL=fake.url)
            out, _ = self.run_ask([{'q': 'a'}, {'q': 'b'}], env=env)
            self.assertEqual([s for s, _ in out], ['generated', 'generated'])
            bodies = [r['body'] for r in fake.chats()]
            self.assertEqual(['max_tokens' in b for b in bodies], [True, False, False])
        finally:
            fake.close()

    def test_extra_identical_unit_is_more_repeats(self):
        self.run_ask([{'q': 'A', 'coords': 'task=t', 'unit': 1}], '--mock', '--exp', 'x')
        _, err = self.run_ask([{'q': 'A', 'coords': 'task=t', 'unit': 1},
                               {'q': 'A', 'coords': 'task=t', 'unit': 2}], '--mock', '--exp', 'x')
        self.assertIn('1 generated: more repeats than stored', err)
        self.assertIn('unit ordinal 2', err)

    def test_state_is_private_and_tokens_guard_every_endpoint(self):
        p, ready = self.start_server('--mock')
        self.assertEqual(os.stat(os.path.join(self.dir, '.copula')).st_mode & 0o777, 0o700)
        for path in ('/v1/models', '/v1/responses/resp_x'):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(ready['url'] + path)
            self.assertEqual(cm.exception.code, 401)
        _, err = self.run_ask([{'q': 'x'}], '--no-identity', rc=None)
        self.assertIn('--no-identity cannot be set here', err)
        self.stop_server(p)

    def test_failed_run_banner_says_unknown(self):
        self.write('fail.py', 'import sys; open("ask.json").read(); sys.exit(3)\n')
        self.run_ask([], '--mock', '--exp', 'f', child=[sys.executable, 'fail.py'], rc=3)
        self.write('fail.py', 'open("ask.json").read()\n')
        _, err = self.run_ask([], '--mock', '--exp', 'f', child=[sys.executable, 'fail.py'])
        self.assertIn('failed, so the files it read are unknown', err)
        self.assertNotIn('unchanged since run', err)


if __name__ == '__main__':
    unittest.main()
