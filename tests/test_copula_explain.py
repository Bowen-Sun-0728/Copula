"""Why calls were generated (copula.calls; `copula explain` and the run summary).

    PYTHONPATH=src:. python3 -m unittest tests.test_copula_explain
"""
import json
import unittest

from copula.calls import body_diff
from tests.copula_fixtures.launch import LaunchCase


def reasons(text):
    return [json.loads(l) for l in text.splitlines() if l.startswith('{"call"')]


class ExplainTest(LaunchCase):
    def ask(self, items, *args):
        out, err = self.run_ask(items, '--mock', '--exp', 'x', *args)
        return out, err, self.run_id(err)

    def calls(self, run):
        return [r for r in (json.loads(l) for l in self.explain(run, '--json', '--calls')
                            .splitlines()) if 'call' in r]

    def test_reasons(self):
        _, err1, r1 = self.ask([{'q': 'You are terse. Question 1'}, {'q': 'repeat me'}])
        self.assertIn('2 generated: new request', err1)
        # an edit: one prompt changed, one request asked twice
        _, err2, r2 = self.ask([{'q': 'You are verbose. Question 1'}, {'q': 'repeat me'},
                                {'q': 'repeat me'}])
        why = {c['reason'] for c in self.calls(r2) if c['outcome'] == 'generated'}
        self.assertEqual(why, {'new_request', 'more_repeats'})
        new = [c for c in self.calls(r2) if c['reason'] == 'new_request'][0]
        self.assertIn("messages[0] (user): 'You are terse.", new['detail'])
        self.assertIn("-> 'You are verbose.", new['detail'])
        self.assertIn('details: copula explain', err2)
        # an unchanged rerun: a new slot, by design
        _, err3, r3 = self.ask([{'q': 'You are verbose. Question 1'}, {'q': 'repeat me'},
                                {'q': 'repeat me'}])
        self.assertEqual({c['reason'] for c in self.calls(r3)}, {'rerun'})
        # a replicate
        _, _, r4 = self.ask([{'q': 'repeat me'}], '--set', 'seed=7')
        c4 = self.calls(r4)
        self.assertEqual(c4[0]['reason'], 'replicate')
        self.assertIn('seed=7 here', c4[0]['detail'])
        text = self.explain(r2)
        self.assertIn('generated: new request', text)
        self.assertIn('generated: more repeats than stored', text)

    def test_arrival_order_of_identical_units(self):
        first = [{'q': 'item A', 'coords': 'task=t', 'unit': 1},
                 {'q': 'item B', 'coords': 'task=t', 'unit': 2}]
        self.ask(first)
        self.write('edit.txt', 'a change')               # a new version reading one more file
        swapped = [{'q': 'item B', 'coords': 'task=t', 'unit': 1},
                   {'q': 'item A', 'coords': 'task=t', 'unit': 2}]
        _, err, r2 = self.ask(swapped, '--track', 'edit.txt')
        got = self.calls(r2)
        self.assertEqual([c['reason'] for c in got], ['arrival_order', 'arrival_order'])
        self.assertIn('ordinal 2', got[0]['detail'])

    def test_greedy_is_global_across_replicates(self):
        _, _, r1 = self.ask([{'q': 'g', 'temperature': 0}])
        _, _, r2 = self.ask([{'q': 'g', 'temperature': 0}, {'q': 'g2', 'temperature': 0}],
                            '--set', 'seed=3')
        got = self.calls(r2)
        self.assertEqual([c['outcome'] for c in got], ['reused', 'generated'])


class DiffTest(unittest.TestCase):
    def test_body_diff(self):
        a = {'model': 'm', 'messages': [{'role': 'system', 'content': 'S'},
                                        {'role': 'user', 'content': 'hello world'}],
             'params': {'temperature': 0.7}}
        b = {'model': 'm', 'messages': [{'role': 'system', 'content': 'S'},
                                        {'role': 'user', 'content': 'hello there'}],
             'params': {'temperature': 0.9}}
        text, score = body_diff(a, b)
        self.assertIn('temperature 0.7 -> 0.9', text)
        self.assertIn("messages[1] (user): 'hello world' -> 'hello there'", text)
        self.assertEqual(score[0], 1)
        c = dict(b, messages=b['messages'] + [{'role': 'user', 'content': 'more'}])
        self.assertIn('1 more message(s) from messages[2]', body_diff(b, c)[0])


if __name__ == '__main__':
    unittest.main()
