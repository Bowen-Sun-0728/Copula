"""Upstream model servers (copula.upstream): a self-hosted vLLM-like server (explicit parameters
from the model's generation_config, an engine seed per sample address, identity from the model
files) and an OpenAI-style API (only non-default parameters, the server's own key), reached from
`copula run` with no backend flag through the caller's OPENAI_BASE_URL.

    PYTHONPATH=src:. python3 -m unittest tests.test_copula_upstream
"""
import json
import os
import tempfile
import unittest

from copula import upstream as up
from tests.copula_fixtures.fake_upstream import FakeUpstream
from tests.copula_fixtures.launch import LaunchCase, env_with


def model_dir(gen=None):
    d = tempfile.mkdtemp(prefix='copula-model-')
    with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump({'model_type': 'fake'}, f)
    if gen is not None:
        with open(os.path.join(d, 'generation_config.json'), 'w') as f:
            json.dump(gen, f)
    return d


class UnitTest(unittest.TestCase):
    def test_generation_defaults(self):
        d = model_dir({'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'repetition_penalty': 1.05,
                       'do_sample': True})
        table, source = up.generation_defaults(d)
        self.assertEqual(source, 'generation_config')
        self.assertEqual((table['temperature'], table['top_p'], table['top_k'],
                          table['repetition_penalty'], table['min_p']), (0.7, 0.8, 20, 1.05, 0.0))
        self.assertEqual(up.generation_defaults(None), (up.NEUTRAL, 'neutral'))

    def test_engine_seed_is_a_function_of_the_address(self):
        a = up.engine_seed('secret', 'k1')
        self.assertEqual(a, up.engine_seed('secret', 'k1'))
        self.assertNotEqual(a, up.engine_seed('secret', 'k2'))
        self.assertNotEqual(a, up.engine_seed('other', 'k1'))
        self.assertTrue(0 <= a < 2 ** 31)

    def test_hf_snapshot(self):
        hub = tempfile.mkdtemp()
        snap = os.path.join(hub, 'models--org--name', 'snapshots', 'abc')
        os.makedirs(snap)
        os.makedirs(os.path.join(hub, 'models--org--name', 'refs'))
        with open(os.path.join(hub, 'models--org--name', 'refs', 'main'), 'w') as f:
            f.write('abc')
        old = os.environ.get('HF_HUB_CACHE')
        os.environ['HF_HUB_CACHE'] = hub
        try:
            self.assertEqual(up.hf_snapshot('org/name'), snap)
            self.assertIsNone(up.hf_snapshot('org/other'))
        finally:
            if old is None:
                os.environ.pop('HF_HUB_CACHE')
            else:
                os.environ['HF_HUB_CACHE'] = old


class SelfHostedTest(LaunchCase):
    def setUp(self):
        super().setUp()
        self.mdir = model_dir({'temperature': 0.6, 'top_k': 20})
        self.fake = FakeUpstream('vllm', root=self.mdir)

    def tearDown(self):
        self.fake.close()
        super().tearDown()

    def test_default_upstream_parameters_seed_and_identity(self):
        env = env_with(OPENAI_BASE_URL=self.fake.url)
        items = [{'q': 'same'}, {'q': 'same'}, {'q': 'explicit', 'temperature': 1.0},
                 {'q': 'greedy', 'temperature': 0}]
        out, err = self.run_ask(items, '--exp', 'u', env=env)
        self.assertIn('sampling defaults: fake-model generation_config', err)
        self.assertIn('engine seed per address', err)
        sent = [r['body'] for r in self.fake.chats()]
        self.assertEqual(len(sent), 4)
        self.assertEqual([b['model'] for b in sent], ['fake-model'] * 4)
        self.assertEqual((sent[0]['temperature'], sent[0]['top_k']), (0.6, 20))   # native default
        self.assertEqual(sent[2]['temperature'], 1.0)                              # as given
        seeds = [b['seed'] for b in sent]
        self.assertEqual(len(set(seeds)), 4)            # one per sample address
        self.assertNotEqual(seeds[0], seeds[1])          # two samples of one request
        # an edit that keeps the first requests: they are read, nothing is sent again for them
        items2 = [{'q': 'same'}, {'q': 'same'}, {'q': 'edited'}, {'q': 'greedy', 'temperature': 0}]
        out2, err2 = self.run_ask(items2 + [{'q': 'new'}], '--exp', 'u', env=env)
        self.assertEqual([s for s, _ in out2], ['hit', 'hit', 'generated', 'hit', 'generated'])
        self.assertEqual(len(self.fake.chats()), 6)
        # the greedy value is global (identity established from the model files)
        out3, _ = self.run_ask([{'q': 'greedy', 'temperature': 0}], '--exp', 'other', env=env)
        self.assertEqual(out3[0], ['hit', out[3][1]])

    def test_without_model_files_greedy_is_per_group(self):
        fake = FakeUpstream('vllm', models=('m2',), root='/nonexistent/m2')
        try:
            env = env_with(OPENAI_BASE_URL=fake.url)
            out, err = self.run_ask([{'q': 'g', 'temperature': 0}], '--exp', 'a', env=env)
            self.assertIn('identity of m2 not established (no model files found)', err)
            self.assertIn('sampling defaults: m2 neutral', err)
            out2, _ = self.run_ask([{'q': 'g', 'temperature': 0}], '--exp', 'b', env=env)
            self.assertEqual(out2[0][0], 'generated')   # W12: not shared across experiments
        finally:
            fake.close()

    def test_engine_seed_off_and_model_alias(self):
        env = env_with(OPENAI_BASE_URL=self.fake.url)
        out, err = self.run_ask([{'q': 'x', 'model': 'gpt-4o-mini'}], '--engine-seed', 'off',
                                '--model', '*', env=env)
        self.assertEqual(out[0][0], 'generated')
        self.assertNotIn('seed', self.fake.chats()[-1]['body'])
        _, err = self.run_ask([{'q': 'x', 'model': 'gpt-4o-mini'}], env=env, rc=None)
        self.assertIn('HTTP Error 404', err)                    # model_not_found, to the child


class ApiTest(LaunchCase):
    def test_sparse_parameters_and_server_key(self):
        fake = FakeUpstream('api', models=('gpt-x',), key='sk-test')
        try:
            env = env_with(OPENAI_BASE_URL=fake.url, OPENAI_API_KEY='sk-test')
            out, err = self.run_ask([{'q': 'hello'}, {'q': 'warm', 'temperature': 0.3},
                                     {'q': 'one', 'temperature': 1.0}, {'q': 'k', 'top_k': 5}],
                                    '--exp', 'api', env=env, rc=None)
            sent = fake.chats()
            self.assertEqual(sent[0]['auth'], 'Bearer sk-test')
            self.assertEqual(set(sent[0]['body']), {'model', 'messages', 'max_tokens',
                                                    'temperature', 'top_p', 'presence_penalty',
                                                    'frequency_penalty'})
            self.assertEqual(sent[0]['body']['temperature'], 1.0)      # absent: sent as 1.0
            self.assertEqual(sent[1]['body']['temperature'], 0.3)
            self.assertEqual(sent[2]['body']['temperature'], 1.0)      # explicit: sent alike
            self.assertIn('top_k', sent[3]['body'])      # sent as given: the API refuses it
            self.assertIn('HTTP Error 400', err)         # the refusal reached the child
            self.assertNotIn('seed', sent[0]['body'])
            self.assertIn('API upstream: temperature and top_p are always sent', err)
        finally:
            fake.close()

    def test_wrong_key_is_found_at_start(self):
        fake = FakeUpstream('api', models=('gpt-x',), key='sk-right')
        try:
            _, err = self.run_ask([{'q': 'x'}], env=env_with(OPENAI_BASE_URL=fake.url,
                                                             OPENAI_API_KEY='sk-wrong'), rc=None)
            self.assertIn('HTTP Error 401', err)
            self.assertEqual(fake.chats(), [])
        finally:
            fake.close()

    def test_no_upstream_is_a_clear_error(self):
        _, err = self.run_ask([{'q': 'x'}], env=env_with(), rc=None)
        self.assertIn('no model server', err)


if __name__ == '__main__':
    unittest.main()
