"""The kernel is the paper's frozen evalrt tree, byte for byte (tools/import_kernel.py).

    PYTHONPATH=src:. python3 -m unittest tests.test_provenance
"""
import hashlib
import json
import os
import unittest

import copula.kernel

KERNEL = os.path.dirname(os.path.abspath(copula.kernel.__file__))


def _py_files():
    out = set()
    for base, dirs, names in os.walk(KERNEL):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        out |= {os.path.relpath(os.path.join(base, n), KERNEL) for n in names if n.endswith('.py')}
    return out


class ProvenanceTest(unittest.TestCase):
    def test_kernel_files_match_the_frozen_tree(self):
        with open(os.path.join(KERNEL, 'PROVENANCE.json')) as f:
            record = json.load(f)['sha256']
        self.assertEqual(_py_files(), set(record), 'kernel files added or removed')
        for rel, digest in sorted(record.items()):
            with open(os.path.join(KERNEL, rel), 'rb') as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), digest, rel)

    def test_research_code_only_behind_static_graphs(self):
        """Only the static-graph entry points of the session import the research code, lazily;
        the package does not offer them."""
        for rel in _py_files():
            with open(os.path.join(KERNEL, rel), encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if s.startswith(('import ', 'from ')) and (
                            'harness_rsi_smoke' in s or '.job' in s or '.control' in s):
                        self.assertEqual(rel, os.path.join('sdk', 'session.py'), s)
                        self.assertTrue(line.startswith('        '), 'not lazy: %s' % s)


if __name__ == '__main__':
    unittest.main()
