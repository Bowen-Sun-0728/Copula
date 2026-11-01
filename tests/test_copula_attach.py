"""Copula: copula.attach and its adapters (src/copula/_attach.py, adapters/;
docs/guides/optimizers.md), against small fake `dspy` and `gepa` packages written here, so the
test needs neither. Real DSPy and GEPA runs are in tests.test_copula_clients. No GPU:
    PYTHONPATH=src:. python3 -m unittest tests.test_copula_attach
"""
import asyncio
import concurrent.futures
import http.server
import importlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import textwrap
import threading
import unittest
import urllib.request

import copula
from copula import _attach as A
from copula import client

FAKE = {
    'dspy/__init__.py': '''
        import contextlib
        class _Settings:
            def __init__(self):
                self.main_thread_config = {'callbacks': []}
                self.configured = 0
            def get(self, k, d=None):
                return self.main_thread_config.get(k, d)
            def configure(self, **kw):
                self.configured += 1
                self.main_thread_config.update(kw)
            @contextlib.contextmanager
            def context(self, **kw):
                old = dict(self.main_thread_config)
                self.main_thread_config.update(kw)
                try:
                    yield
                finally:
                    self.main_thread_config.clear()
                    self.main_thread_config.update(old)
        settings = _Settings()
        configure = settings.configure
        context = settings.context
        cache_calls = []
        def configure_cache(**kw):
            cache_calls.append(kw)
        ''',
    'dspy/utils/__init__.py': '',
    'dspy/utils/callback.py': '''
        class BaseCallback:
            pass
        ''',
    'gepa/__init__.py': 'from gepa.proposer.reflective_mutation import reflective_mutation\n',
    'copula_probe_mod.py': 'X = 1\n',
    'gepa/proposer/__init__.py': '',
    'gepa/proposer/reflective_mutation/__init__.py': '',
    'gepa/proposer/reflective_mutation/reflective_mutation.py': '''
        from copula import client
        class ReflectiveMutationProposer:
            def propose(self, state):
                return dict(client.current())
            def propose_new_texts(self, *a):
                return client.headers()
        ''',
}


class _Echo(http.server.BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length') or 0))
        _Echo.seen.append({k: v for k, v in self.headers.items() if k.lower().startswith('x-copula')})
        self.send_response(200)
        self.send_header('Content-Length', '2')
        self.end_headers()
        self.wfile.write(b'{}')

    def log_message(self, *a):
        pass


def serve():
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Echo)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, 'http://127.0.0.1:%d' % srv.server_address[1]


class AttachTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix='copula-attach-')
        for path, text in FAKE.items():
            full = os.path.join(cls.dir, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, 'w') as f:
                f.write(textwrap.dedent(text))
        sys.path.insert(0, cls.dir)
        cls.srv, cls.url = serve()
        cls.other, cls.other_url = serve()
        os.environ['COPULA_QUIET'] = '1'
        A.attach(url=cls.url + '/r/tok/v1')

    @classmethod
    def tearDownClass(cls):
        A._state['base'] = None                   # stop adding headers for later tests
        A._state['host'] = None
        if A._state['finder'] in sys.meta_path:
            sys.meta_path.remove(A._state['finder'])
        sys.path.remove(cls.dir)
        for m in [m for m in sys.modules if m.split('.')[0] in ('dspy', 'gepa', 'copula_probe_mod')]:
            del sys.modules[m]
        cls.srv.shutdown()
        cls.other.shutdown()
        shutil.rmtree(cls.dir, ignore_errors=True)
        os.environ.pop('COPULA_QUIET', None)

    def post(self, url):
        _Echo.seen.clear()
        req = urllib.request.Request(url + '/r/tok/v1/chat/completions', data=b'{}',
                                     method='POST')
        urllib.request.urlopen(req, timeout=10).read()
        return _Echo.seen[-1]

    def test_threads_copy_the_context(self):
        with copula.run(case='t1'):
            with concurrent.futures.ThreadPoolExecutor(2) as ex:
                got = ex.submit(lambda: client.current().get('coords')).result()

            async def via_loop():
                return await asyncio.get_running_loop().run_in_executor(
                    None, lambda: client.current().get('coords'))
            looped = asyncio.run(via_loop())
        self.assertEqual((got, looped), ({'case': 't1'}, {'case': 't1'}))

    def test_headers_only_to_the_copula_server(self):
        with copula.run(case='h1'), copula.decision('d9'):
            ours = self.post(self.url)
            theirs = self.post(self.other_url)
        self.assertEqual((ours['X-Copula-Coords'], ours['X-Copula-Decision']), ('case=h1', 'd9'))
        self.assertEqual(theirs, {})
        with copula.run(case='h2'):                              # explicit headers win, once
            got = copula.chat([{'role': 'user', 'content': 'x'}], model='m',
                              base_url=self.url + '/r/tok/v1')
        self.assertEqual(got, {'_copula': {}})
        self.assertEqual(list(_Echo.seen[-1]).count('X-Copula-Coords'), 1)

    def test_submit_keeps_its_signature(self):
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            self.assertEqual(ex.submit(lambda fn: fn + 1, fn=1).result(), 2)

    def test_post_import_hook_survives_a_probe(self):
        ran = []
        A._state['finder'].on('copula_probe_mod', lambda m: ran.append(m.X))
        self.assertIsNotNone(importlib.util.find_spec('copula_probe_mod'))  # a probe only
        self.assertEqual(ran, [])
        import copula_probe_mod                                    # noqa: F401
        self.assertEqual(ran, [1])
        self.assertNotIsInstance(copula_probe_mod.__spec__.loader, type(None))

    def test_phase_and_quoting_without_a_unit(self):
        with copula.holdout(), copula.role('jüdge'):
            h = client.headers()
        self.assertEqual(h, {'X-Copula-Phase': 'holdout', 'X-Copula-Role': 'j%C3%BCdge'})

    def test_attach_checks_the_experiment(self):
        os.environ['COPULA_EXP'] = 'a'
        try:
            with self.assertRaises(ValueError):
                A.attach(exp='b', url=self.url)
            A.attach(exp='a', url=self.url)                     # agrees: a no-op
        finally:
            del os.environ['COPULA_EXP']

    def test_gepa_decisions_and_proposer_role(self):
        gepa = importlib.import_module('gepa')
        P = gepa.reflective_mutation.ReflectiveMutationProposer
        a, b = P().propose(None), P().propose(None)
        self.assertTrue(a['decision'].startswith('d-'))
        self.assertNotEqual(a['decision'], b['decision'])        # one scope per proposal
        self.assertEqual(P().propose_new_texts()['X-Copula-Role'], 'proposer')
        self.assertNotIn('decision', client.current())

    def test_dspy_adapter(self):
        dspy = importlib.import_module('dspy')
        self.assertEqual(dspy.cache_calls, [{'enable_disk_cache': False,
                                             'enable_memory_cache': False}])
        cb = dspy.settings.get('callbacks')[-1]
        self.assertEqual(dspy.settings.configured, 0)             # no configure ownership taken
        dspy.configure(callbacks=[])                              # the user's own list keeps ours
        self.assertIs(dspy.settings.get('callbacks')[-1], cb)
        with dspy.context(callbacks=[]):
            self.assertIs(dspy.settings.get('callbacks')[-1], cb)

        class Prog:
            def __init__(self, instr):
                self.instr = instr

            def dump_state(self):
                return {'instr': self.instr}

        class Opt:
            seed = 7

        class Ex(dict):
            def inputs(self):
                return self

        def module(prog, **kw):
            cb.on_module_start('m', prog, {'args': (), 'kwargs': kw})
            inner = client.headers()
            cb.on_module_start('n', prog, {'args': (), 'kwargs': {'nested': 1}})
            nested = client.headers()
            cb.on_module_end('n', None)
            cb.on_module_end('m', None)
            return inner, nested

        cb.on_compile_start('c', Opt(), {'trainset': [Ex(question='q7')], 'valset': []})
        h1, nested = module(Prog('v1'), question='q1')
        h2, _ = module(Prog('v2'), question='q1')
        h3, _ = module(Prog('v1'), question='q1')
        cb.on_compile_end('c', None)
        self.assertEqual(h1, nested)                             # the innermost is not a unit
        c1, c2 = dict(p.split('=') for p in h1['X-Copula-Coords'].split(';')), \
            dict(p.split('=') for p in h2['X-Copula-Coords'].split(';'))
        self.assertEqual((c1['optimizer_seed'], c1['case']), ('7', c2['case']))
        self.assertNotIn('seed', c1)                             # never the process's own seed
        self.assertNotEqual(c1['candidate'], c2['candidate'])
        self.assertEqual((h1['X-Copula-Unit'], h3['X-Copula-Unit']), ('1', '2'))
        self.assertEqual(client.current(), {})

        cb.on_evaluate_start('e1', None, {'devset': [Ex(question='q1')]})   # used in compile
        self.assertNotIn('phase', client.current())
        cb.on_evaluate_end('e1', None)
        self.assertTrue(any('W5b' in n for n in A.notes()))
        cb.on_evaluate_start('e3', None, {'devset': [Ex(question='q7')]})   # a trainset example
        self.assertNotIn('phase', client.current())                         # never ran, still seen
        cb.on_evaluate_end('e3', None)
        cb.on_evaluate_start('e2', None, {'devset': [Ex(question='q9')]})   # new examples
        h4, _ = module(Prog('v2'), question='q9')
        cb.on_evaluate_end('e2', None)
        self.assertEqual((h4['X-Copula-Phase'], h4['X-Copula-Phase-Source']), ('holdout', 'inferred'))
        self.assertIn('optimizer_seed=7', h4['X-Copula-Coords'])          # carried after compile
        self.assertEqual(client.current(), {})


if __name__ == '__main__':
    unittest.main()
