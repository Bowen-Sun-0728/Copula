"""evalrt core: canonical JSON, requests, draws, designs and keys (RUNTIME_V0 section 13, step 1).

Standard library only, so it runs under /usr/bin/python3 3.9 as well as the project's pytest:
    PYTHONPATH=src python3 -m unittest tests.kernel.test_evalrt_core
"""
import itertools
import math
import random
import unittest

from copula.kernel.core import canon, key as keys
from copula.kernel.core.design import Design, DesignError, Draw
from copula.kernel.core.request import RequestSpecError, check_defaults, normalize_request
from copula.kernel.testing.mock_model import MOCK_DEFAULTS

BODY = {'messages': [{'role': 'user', 'content': 'q'}], 'max_tokens': 16}


class CanonTest(unittest.TestCase):
    def test_order_independent_and_unicode_kept(self):
        self.assertEqual(canon.canonical({'b': 1, 'a': '法条'}), canon.canonical({'a': '法条', 'b': 1}))
        self.assertIn('法条'.encode('utf-8'), canon.canonical({'a': '法条'}))

    def test_nan_refused(self):
        with self.assertRaises(ValueError):
            canon.canonical({'x': math.nan})

    def test_keyed_digest_needs_secret_and_differs_from_plain(self):
        with self.assertRaises(ValueError):
            canon.keyed_digest(b'short', {'a': 1})
        secret = b'0123456789abcdef'
        self.assertEqual(canon.keyed_digest(secret, {'a': 1}), canon.keyed_digest(secret, {'a': 1}))
        self.assertNotEqual(canon.keyed_digest(secret, {'a': 1}), canon.digest({'a': 1}))
        self.assertNotEqual(canon.keyed_digest(secret, {'a': 1}),
                            canon.keyed_digest(b'fedcba9876543210', {'a': 1}))


class RequestTest(unittest.TestCase):
    def test_every_parameter_is_explicit(self):
        out = normalize_request(BODY, MOCK_DEFAULTS)
        for name in MOCK_DEFAULTS:
            self.assertIn(name, out)
        self.assertEqual(out['max_tokens'], 16)

    def test_forbidden_and_unknown_fields_refused(self):
        for bad in ('seed', 'n', 'best_of', 'model', 'stream', 'logprobs', 'nonsense'):
            with self.assertRaises(RequestSpecError, msg=bad):
                normalize_request(dict(BODY, **{bad: 1}), MOCK_DEFAULTS)

    def test_equivalent_spellings_hash_alike(self):
        a = normalize_request(dict(BODY, temperature=1, stop=['b', 'a', 'b']), MOCK_DEFAULTS)
        b = normalize_request(dict(BODY, temperature=1.0, stop=['a', 'b']), MOCK_DEFAULTS)
        self.assertEqual(canon.digest(a), canon.digest(b))

    def test_different_parameters_hash_differently(self):
        a = normalize_request(BODY, MOCK_DEFAULTS)
        b = normalize_request(dict(BODY, top_p=0.95), MOCK_DEFAULTS)
        self.assertNotEqual(canon.digest(a), canon.digest(b))

    def test_malformed_requests_refused(self):
        for body in ({'messages': [], 'max_tokens': 4},
                     {'messages': [{'role': 'tool', 'content': 'x'}], 'max_tokens': 4},
                     {'messages': [{'role': 'user', 'content': 'x', 'colour': 'y'}],
                      'max_tokens': 4},
                     {'messages': [{'role': 'user', 'content': 'x'}], 'max_tokens': 0},
                     {'messages': [{'role': 'user', 'content': 'x'}], 'max_tokens': True},
                     dict(BODY, temperature=math.inf)):
            with self.assertRaises(RequestSpecError):
                normalize_request(body, MOCK_DEFAULTS)
        # a message name is part of the request (sysval B10; it used to be refused)
        named = normalize_request({'messages': [{'role': 'user', 'content': 'x', 'name': 'y'}],
                                   'max_tokens': 4}, MOCK_DEFAULTS)
        self.assertEqual(named['messages'], [{'role': 'user', 'content': 'x', 'name': 'y'}])

    def test_default_table_must_be_complete(self):
        partial = dict(MOCK_DEFAULTS)
        del partial['top_k']
        with self.assertRaises(RequestSpecError):
            check_defaults(partial)
        with self.assertRaises(RequestSpecError):
            check_defaults(dict(MOCK_DEFAULTS, seed=1))


class DrawTest(unittest.TestCase):
    def test_coordinates_equal_to_one_are_omitted_but_round_is_kept(self):
        self.assertEqual(Draw(1, {'draft': 1}).canonical(), {'r': 1, 'fan': []})
        self.assertEqual(Draw(1, {'draft': 1}).canonical(), Draw(1).canonical())
        self.assertEqual(Draw(2, {'draft': 3, 'a': 1}).canonical(), {'r': 2, 'fan': [['draft', 3]]})
        self.assertNotEqual(Draw(1).canonical(), Draw(2).canonical())

    def test_a_node_named_r_cannot_collide_with_the_round(self):
        self.assertNotEqual(Draw(1, {'r': 2}).canonical(), Draw(2).canonical())

    def test_wire_round_trip_and_validation(self):
        d = Draw(3, {'x': 2, 'y': 1})
        self.assertEqual(Draw.from_wire(d.to_wire()), d)
        for bad in (lambda: Draw(0), lambda: Draw(1, {'x': 0}), lambda: Draw(1, [('x', 1), ('x', 2)]),
                    lambda: Draw.from_wire({'fan': {}})):
            with self.assertRaises(DesignError):
                bad()


def chain():
    return {'input': [], 'constrain': ['input'], 'draft': ['input', 'constrain'],
            'critique': ['input', 'draft'], 'answer': ['draft', 'critique']}


def diamond():
    return {'input': [], 'constrain': ['input'], 'draft': ['input'],
            'critique': ['constrain', 'draft'], 'answer': ['critique']}


class DesignTest(unittest.TestCase):
    def test_interface_example_1_chain(self):
        d = Design(2, {'draft': 4}, chain())
        self.assertEqual([d.size(n) for n in ('constrain', 'draft', 'critique', 'answer')],
                         [2, 8, 8, 8])

    def test_interface_example_2_diamond_crosses_private_coordinates(self):
        d = Design(2, {'constrain': 2, 'draft': 3}, diamond())
        self.assertEqual([d.size(n) for n in ('constrain', 'draft', 'critique', 'answer')],
                         [4, 6, 12, 12])
        draw = Draw(1, {'constrain': 2, 'draft': 3})
        self.assertEqual(d.parent_draw(draw, 'constrain'), Draw(1, {'constrain': 2}))
        self.assertEqual(d.parent_draw(draw, 'draft'), Draw(1, {'draft': 3}))
        self.assertEqual(len(set(d.draws('critique'))), 12)

    def test_uniform_design(self):
        d = Design(8, {}, chain())
        self.assertTrue(all(d.size(n) == 8 for n in d.order))

    def test_invalid_designs_refused(self):
        with self.assertRaises(DesignError):
            Design(0, {}, chain())
        with self.assertRaises(DesignError):
            Design(1, {'nope': 2}, chain())
        with self.assertRaises(DesignError):
            Design(1, {'constrain': 0}, chain())
        with self.assertRaises(DesignError):
            Design(1, {'input': 2}, chain(), allowed={'draft'})
        with self.assertRaises(DesignError):
            Design(1, {}, {'a': ['b'], 'b': ['a']})


def random_dag(rng, n):
    nodes = ['n%d' % i for i in range(n)]
    return {node: sorted(rng.sample(nodes[:i], rng.randint(0, min(i, 3)))) if i else []
            for i, node in enumerate(nodes)}


class KeyTest(unittest.TestCase):
    def key(self, node, rep, sample=0, case='c1', generation='exec/0', body=BODY):
        return keys.make_key(case, 'm1', node, body, rep, sample, generation, MOCK_DEFAULTS)[1]

    def test_replicate_index_counts_earlier_draws_with_the_same_input(self):
        self.assertEqual(keys.replicate_index([], 'x'), 1)
        self.assertEqual(keys.replicate_index(['x', None, 'y', 'x'], 'x'), 3)
        self.assertEqual(keys.replicate_index(['y', None], 'x'), 1)

    def test_stochastic_keys_are_distinct_within_a_job(self):
        """T2, structural: over random DAGs, designs and input collisions, every draw of a job
        gets its own key -- also when different draws happen to see identical input."""
        rng = random.Random(1)
        for _ in range(60):
            parents = random_dag(rng, rng.randint(2, 6))
            fanout = {n: rng.randint(1, 3) for n in parents if rng.random() < 0.4}
            design = Design(rng.randint(1, 3), fanout, parents)
            seen = set()
            count = 0
            for case in ('c1', 'c2'):
                for node in design.order:
                    inputs = []
                    for draw in design.draws(node):
                        text = 'q%d' % rng.randint(0, 2)          # frequent identical inputs
                        body = dict(BODY, messages=[{'role': 'user', 'content': text}])
                        rep = keys.replicate_index(inputs, text)
                        inputs.append(text)
                        seen.add(self.key(node, rep, case=case, body=body))
                        count += 1
            self.assertEqual(len(seen), count)

    def test_samples_for_one_input_align_whatever_the_design_shape(self):
        """A node is memoryless: the k-th sample for an input is one key in every design."""
        def keys_of(rounds, fanout):
            design = Design(rounds, fanout, chain())
            out, inputs = [], []
            for _ in design.draws('constrain'):                  # a root: one input throughout
                rep = keys.replicate_index(inputs, 'same')
                inputs.append('same')
                out.append(self.key('constrain', rep))
            return out
        five = keys_of(5, {})
        ten_nested = keys_of(2, {'constrain': 5})
        ten_uniform = keys_of(10, {})
        self.assertEqual(ten_nested, ten_uniform)
        self.assertEqual(len(set(five) & set(ten_nested)), 5)

    def test_generation_case_node_body_and_rep_separate_keys(self):
        base = self.key('draft', 1)
        self.assertNotEqual(base, self.key('draft', 2))
        self.assertNotEqual(base, self.key('draft', 1, generation='compile/0'))
        self.assertNotEqual(base, self.key('draft', 1, generation='fresh/T7'))
        self.assertNotEqual(base, self.key('draft', 1, case='c2'))
        self.assertNotEqual(base, self.key('critique', 1))
        self.assertNotEqual(base, self.key('draft', 1, body=dict(BODY, max_tokens=17)))

    def test_bad_arguments_refused(self):
        for label in ('exec', 'exec/x', 'fresh/', 'other/0', 'fresh/a b', 5):
            with self.assertRaises(keys.KeySpecError):
                keys.check_generation(label)
        for rep in (0, -1, True, 1.0):
            with self.assertRaises(keys.KeySpecError):
                keys.stochastic_key('c1', 'm1', 'draft', 'b' * 64, rep, 0, 'exec/0')

    def test_deterministic_and_stochastic_keys_never_coincide(self):
        det = keys.deterministic_key('c1', 'm1', 'recall', 'b' * 64)
        sto = keys.stochastic_key('c1', 'm1', 'recall', 'b' * 64, 1, 0, 'exec/0')
        self.assertNotEqual(keys.key_hash(det), keys.key_hash(sto))
        self.assertEqual(det['v'], keys.KEY_VERSION)
        self.assertEqual(sto['v'], keys.KEY_VERSION)


if __name__ == '__main__':
    unittest.main()
