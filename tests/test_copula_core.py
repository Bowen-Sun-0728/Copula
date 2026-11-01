"""Copula preliminary implementation: coordinates, the registry and call attributes
(reports/opensource_20261022/DESIGN.md §4, §6.2). No manager, no model:
    PYTHONPATH=src:. python3 -m unittest tests.test_copula_core
"""
import os
import shutil
import tempfile
import unittest

from copula.attrs import infer, node_name
from copula.coords import Classes, parse_argv
from copula.registry import HoldoutOverlap, Registry, check_route, file_digest


class CoordsTest(unittest.TestCase):
    def test_default_classes(self):
        c = Classes()
        for name in ('seed', 'SLURM_ARRAY_TASK_ID', 'epoch', 'attempt', 'lr', 'split', 'foo'):
            self.assertEqual(c.of(name), 'replicate', name)       # unknown names too (V1)
        for name in ('variant', 'candidate', 'prompt'):
            self.assertEqual(c.of(name), 'compare', name)
        for name in ('case', 'example', 'doc_id'):
            self.assertEqual(c.of(name), 'case', name)
        for name in ('-o', '--output-dir', 'log_file', 'save_path', 'iteration', 'metric_key',
                     'run-name', 'group_id'):
            self.assertEqual(c.of(name), 'annotation', name)

    def test_recipe_and_overrides(self):
        self.assertEqual(Classes('eval').of('task'), 'compare')
        self.assertEqual(Classes('dev').of('task'), 'replicate')
        self.assertEqual(Classes('dev', {'lr': 'compare'}).of('--lr'), 'compare')
        with self.assertRaises(ValueError):
            Classes('nope')
        with self.assertRaises(ValueError):
            Classes('dev', {'x': 'version'})

    def test_parse_argv(self):
        got = parse_argv(['--split', 'dev', 'lr=3e-4', '-o', 'out/v2', '--flag', '--k=v', 'pos',
                          '--lr2', '-0.1', '+db.port=5'])
        self.assertEqual(got, {'split': 'dev', 'lr': '3e-4', 'o': 'out/v2', 'k': 'v',
                               'lr2': '-0.1', 'db.port': '5'})
        self.assertEqual(parse_argv(['--debug', 'lr=3', '-m', 'pkg', '--', 'x', '--k', 'v']),
                         {'lr': '3', 'm': 'pkg'})


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='copula-reg-')
        self.reg = Registry(os.path.join(self.dir, 'registry.db'))

    def tearDown(self):
        self.reg.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def launch(self, exp='e', coords=None, phase='search', overrides=None, reads=None):
        rec = self.reg.process_run(exp, 'dev', phase, coords, overrides)
        if reads is not None:
            self.reg.finish(rec['run_id'], reads, True)
        return rec, rec['rerun_of']

    def write(self, name, text):
        p = os.path.join(self.dir, name)
        with open(p, 'w') as f:
            f.write(text)
        return p

    def test_no_experiment_gets_fresh_groups(self):
        a, _ = self.launch(exp=None)
        b, _ = self.launch(exp=None)
        self.assertNotEqual(a['grp'], b['grp'])
        self.assertIsNone(a['slot'])

    def test_rerun_new_slot_edit_shares_revert_is_rerun(self):
        f = self.write('prompt.txt', 'v1')
        r1, _ = self.launch(reads={f: file_digest(f)})
        self.assertEqual(r1['slot'], 1)
        r2, rerun_of = self.launch(reads={f: file_digest(f)})     # unchanged
        self.assertEqual((r2['slot'], rerun_of, r2['version']), (2, r1['run_id'], r1['version']))
        self.assertNotEqual(r2['grp'], r1['grp'])
        self.write('prompt.txt', 'v2')                             # an edit
        r3, rerun_of = self.launch(reads={f: file_digest(f)})
        self.assertIsNone(rerun_of)
        self.assertNotEqual(r3['version'], r1['version'])
        self.assertEqual((r3['slot'], r3['grp']), (1, r1['grp']))  # compared: shares slot 1
        self.write('prompt.txt', 'v1')                             # back to v1: a rerun of v1
        r4, rerun_of = self.launch(reads={f: file_digest(f)})
        self.assertEqual((r4['version'], r4['slot']), (r1['version'], 3))

    def test_unknown_read_set_is_a_rerun(self):
        r1, _ = self.launch()
        r2, rerun_of = self.launch()
        self.assertEqual((r2['slot'], rerun_of), (2, r1['run_id']))

    def test_replicates_and_annotations(self):
        a, _ = self.launch(coords={'seed': '1'})
        b, _ = self.launch(coords={'seed': '2'})
        self.assertEqual((a['slot'], b['slot']), (1, 1))
        self.assertNotEqual(a['grp'], b['grp'])                    # replicate in the key
        c, _ = self.launch(coords={'seed': '1', 'o': 'out/v9'})
        self.assertEqual(c['slot'], 2)                             # annotation: a rerun of a

    def test_compare_coordinates_share(self):
        a, _ = self.launch(coords={'variant': 'a'})
        b, rerun_of = self.launch(coords={'variant': 'b'})
        self.assertIsNone(rerun_of)
        self.assertEqual((b['slot'], b['grp']), (1, a['grp']))
        c, _ = self.launch(coords={'lr': '1'}, overrides={'lr': 'compare'})
        d, _ = self.launch(coords={'lr': '2'}, overrides={'lr': 'compare'})
        self.assertEqual(c['grp'], d['grp'])

    def test_units(self):
        p, _ = self.launch()
        u1 = self.reg.unit(p, {'case': '1'})
        u1b = self.reg.unit(p, {'case': '1'}, ordinal=2)
        u2 = self.reg.unit(p, {'case': '2'})
        self.assertEqual((u1['slot'], u1b['slot'], u2['slot']), (1, 2, 1))
        self.assertEqual(len({u1['grp'], u1b['grp'], u2['grp']}), 3)
        self.assertEqual(self.reg.unit(p, {'case': '1'})['run_id'], u1['run_id'])   # same unit
        a = self.reg.unit(p, {'candidate': 'a', 'case': '3'})
        b = self.reg.unit(p, {'candidate': 'b', 'case': '3'})
        self.assertEqual((a['grp'], b['slot']), (b['grp'], 1))     # compared candidates share

    def test_decision_scope_first_member_rule(self):
        p, _ = self.launch()
        self.reg.unit(p, {'candidate': 'parent', 'case': '1'})          # parent used slot 1
        par = self.reg.unit(p, {'candidate': 'parent', 'case': '1'}, ordinal=2, decision='d1')
        child = self.reg.unit(p, {'candidate': 'child', 'case': '1'}, decision='d1')
        self.assertEqual((par['slot'], child['slot']), (2, 2))
        self.assertEqual(par['grp'], child['grp'])                       # paired
        loose = self.reg.unit(p, {'candidate': 'other', 'case': '1'})   # no scope: slot 1
        self.assertEqual(loose['slot'], 1)

    def test_holdout_cases_must_be_new(self):
        p, _ = self.launch()
        self.reg.unit(p, {'case': '1'})
        h, _ = self.launch(phase='holdout')
        with self.assertRaises(HoldoutOverlap):
            self.reg.unit(h, {'case': '1'})
        u = self.reg.unit(h, {'case': '9'})
        self.assertNotEqual(u['grp'], self.reg.unit(p, {'case': '9'})['grp'])

    def test_persistence_and_tokens(self):
        r1, _ = self.launch()
        self.reg.close()
        self.reg = Registry(os.path.join(self.dir, 'registry.db'))
        r2, _ = self.launch()
        self.assertEqual(r2['slot'], 2)
        tok = self.reg.token(r1['run_id'])
        self.assertEqual(self.reg.run_of(tok)['run_id'], r1['run_id'])
        self.assertIsNone(self.reg.run_of(tok[:-1] + ('0' if tok[-1] != '0' else '1')))
        self.assertIsNone(self.reg.run_of('r-nothing.abc'))

    def test_simultaneous_launches_are_reruns(self):
        a, _ = self.launch()                                       # still running
        b, rerun_of = self.launch()
        self.assertEqual((b['slot'], rerun_of, b['version']), (2, a['run_id'], a['version']))

    def test_a_failed_run_does_not_capture_later_versions(self):
        f = self.write('p.txt', 'v1')
        r1, _ = self.launch(reads={f: file_digest(f)})
        self.write('p.txt', 'v2')
        r2, _ = self.launch()
        self.reg.finish(r2['run_id'], {}, False)                  # crashed: reads unknown
        self.assertEqual(self.reg.get(r2['run_id'])['status'], 'unknown')
        r3, rerun_of = self.launch(reads={f: file_digest(f)})    # newest is unknown: a rerun
        self.assertEqual((rerun_of, r3['slot']), (r2['run_id'], 2))
        self.write('p.txt', 'v3')
        r4, rerun_of = self.launch(reads={f: file_digest(f)})    # r3 is known and differs
        self.assertIsNone(rerun_of)
        self.assertEqual((r4['slot'], r4['grp']), (1, r1['grp']))

    def test_case_coordinates_of_process_runs(self):
        a, _ = self.launch(coords={'task_id': '1'})
        b, _ = self.launch(coords={'task_id': '2'})
        self.assertEqual((a['slot'], b['slot']), (1, 1))
        self.assertNotEqual(a['grp'], b['grp'])                    # cases are separate
        d = self.reg.process_run('e', 'dev', 'search', {'task_id': '2'}, None, 'v-new')
        e = self.reg.process_run('e', 'dev', 'search', {'task_id': '1'}, None, 'v-new')
        self.assertEqual((d['slot'], d['grp'], e['grp']), (1, b['grp'], a['grp']))

    def test_decision_ids_are_scoped_to_their_process_run(self):
        for _ in range(2):
            p, _ = self.launch(coords={'seed': str(_)})
            self.reg.unit(p, {'candidate': 'parent', 'case': '1'})
            par = self.reg.unit(p, {'candidate': 'parent', 'case': '1'}, 2, 'accept')
            child = self.reg.unit(p, {'candidate': 'child', 'case': '1'}, 1, 'accept')
            self.assertEqual((par['slot'], child['slot']), (2, 2))

    def test_check_route(self):
        for rule in ('run', 'global', 'fresh', 'per:seed'):
            self.assertEqual(check_route(rule), rule)
        for rule in ('per:', 'everywhere', None):
            with self.assertRaises(ValueError):
                check_route(rule)

    def test_routes(self):
        a, _ = self.launch(coords={'seed': '1'})
        b, _ = self.launch(coords={'seed': '2'})
        self.assertNotEqual(self.reg.route(a, 'proposer', 'per:seed'),
                            self.reg.route(b, 'proposer', 'per:seed'))
        self.assertEqual(self.reg.route(a, 'judge', 'global'), self.reg.route(b, 'judge', 'global'))
        self.assertEqual(self.reg.route(a, 'x', 'run'), a['grp'])


class AttrsTest(unittest.TestCase):
    def test_greedy_needs_identity(self):
        self.assertEqual(infer({'temperature': 0}, True).randomness, 'deterministic')
        a = infer({'temperature': 0}, False)
        self.assertEqual((a.randomness, a.cache), ('stochastic', True))
        self.assertTrue(a.notes)
        self.assertEqual(infer({'top_k': 1, 'temperature': 0.7}, True).greedy, 'top_k')
        self.assertEqual(infer({'temperature': 0.7}).randomness, 'stochastic')

    def test_seed_is_stochastic_and_keyed(self):
        a = infer({'temperature': 0.7, 'seed': 5})
        self.assertEqual((a.randomness, a.cache, a.seed), ('stochastic', True, 5))
        self.assertNotIn('seed', a.body)
        b = infer({'temperature': 0, 'extra_body': {'seed': 3}}, True)
        self.assertEqual((b.randomness, b.seed), ('stochastic', 3))
        self.assertNotIn('seed', b.body['extra_body'])

    def test_uncached_requests(self):
        self.assertFalse(infer({'previous_response_id': 'resp_x'}).cache)
        self.assertFalse(infer({'tools': [{'type': 'web_search'}]}).cache)
        self.assertTrue(infer({'tools': [{'type': 'function', 'function': {'name': 'f'}}]}).cache)

    def test_node_names(self):
        names = {node_name('actor', infer(b, True)) for b in (
            {'temperature': 0}, {'top_k': 1}, {'temperature': 1}, {'previous_response_id': 'x'})}
        self.assertEqual(names, {'actor:d', 'actor:dk', 'actor:s', 'actor:u'})


if __name__ == '__main__':
    unittest.main()
