#!/usr/bin/env python3
"""Copy the kernel from a frozen evalrt source tree into src/copula/kernel, byte for byte, and
port its tests into tests/kernel.

    python tools/import_kernel.py SRC_DIR [--tests TESTS_DIR]

SRC_DIR is the directory that holds the `evalrt` package (the paper's frozen tree). The kernel is
the SDK, manager, store, keys, RPC and test doubles. Static graphs (`job/`, `control/`,
`compile.py`, `testing/kinds.py`) and the cluster tools (`tools/`) stay behind: they import the
research code. The kernel uses relative imports only, so the copied files are unchanged;
src/copula/kernel/PROVENANCE.json records every file's sha256 and tests/test_provenance.py checks
it.

Tests are not part of the provenance: imports are rewritten from `evalrt` to `copula.kernel`, and
the two static-graph tests of the SDK suite are skipped.
"""
import argparse
import hashlib
import json
import os
import re
import shutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL = ['__init__.py', 'client.py', 'errors.py', 'rpc.py', 'core', 'sdk', 'manager', 'store',
          'testing/__init__.py', 'testing/mock_model.py', 'testing/rig.py', 'testing/fake_vllm.py']
TESTS = ['test_evalrt_core.py', 'test_evalrt_lease_expiry.py', 'test_evalrt_leases.py',
         'test_evalrt_manager.py', 'test_evalrt_queue_commit.py', 'test_evalrt_sdk.py',
         'test_evalrt_sdk_clients.py', 'test_evalrt_vllm_client.py']
STATIC = ['test_static_graphs_run_without_interception', 'test_graph_inputs_keep_their_ids']


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def files(src):
    out = []
    for item in KERNEL:
        p = os.path.join(src, 'evalrt', item)
        if os.path.isdir(p):
            for base, dirs, names in os.walk(p):
                dirs[:] = sorted(d for d in dirs if d != '__pycache__')
                out += [os.path.relpath(os.path.join(base, n), os.path.join(src, 'evalrt'))
                        for n in sorted(names) if n.endswith('.py')]
        else:
            out.append(item)
    return sorted(out)


def port_test(text):
    text = text.replace('from tests.test_evalrt_models import graph, node\n',
                        'graph = node = None          # static graphs are not in the package\n')
    text = re.sub(r'\bevalrt\.', 'copula.kernel.', text)
    text = re.sub(r'\bfrom evalrt import\b', 'from copula.kernel import', text)
    text = text.replace("'-m', 'evalrt.manager'", "'-m', 'copula.kernel.manager'")
    text = re.sub(r'\btests\.test_evalrt_', 'tests.kernel.test_evalrt_', text)
    text = text.replace('os.path.dirname(os.path.dirname(os.path.abspath(__file__)))',
                        'os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))')
    for name in STATIC:
        text = text.replace('    async def %s(' % name,
                            "    @unittest.skip('static graphs are not in the package')\n"
                            '    async def %s(' % name)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--tests')
    a = ap.parse_args()
    dst = os.path.join(HERE, 'src', 'copula', 'kernel')
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    record = {}
    for rel in files(a.src):
        s = os.path.join(a.src, 'evalrt', rel)
        d = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(s, d)
        record[rel] = sha(d)
    with open(os.path.join(dst, 'PROVENANCE.json'), 'w') as f:
        json.dump({'source': 'evalrt (the paper\'s frozen tree, src digest 54efa3fe)',
                   'note': 'byte-identical copies; tests/test_provenance.py checks them',
                   'sha256': record}, f, indent=1, sort_keys=True)
        f.write('\n')
    print('kernel: %d files' % len(record))
    if a.tests:
        tdst = os.path.join(HERE, 'tests', 'kernel')
        os.makedirs(tdst, exist_ok=True)
        open(os.path.join(tdst, '__init__.py'), 'a').close()
        for name in TESTS:
            with open(os.path.join(a.tests, name), encoding='utf-8') as f:
                text = port_test(f.read())
            with open(os.path.join(tdst, name), 'w', encoding='utf-8') as f:
                f.write(text)
        print('tests: %d files' % len(TESTS))


if __name__ == '__main__':
    main()
