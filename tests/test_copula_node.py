"""Read sets of Node children (the NODE_OPTIONS preload, _shim/node/copula-reads.js): an edit of a
file a Node program reads makes a new version; an unchanged rerun is a rerun. Skipped without node.

    PYTHONPATH=src:. python3 -m unittest tests.test_copula_node
"""
import os
import shutil
import subprocess
import unittest

from tests.copula_fixtures.launch import LaunchCase



def _node():
    """A node >= 18 (global fetch) that runs here, or None."""
    path = shutil.which('node')
    try:
        v = subprocess.run([path, '--version'], capture_output=True, text=True, timeout=20)
        return path if v.returncode == 0 and int(v.stdout.strip().lstrip('v').split('.')[0]) >= 18 \
            else None
    except (TypeError, OSError, ValueError, subprocess.TimeoutExpired):
        return None


NODE = _node()

AGENT = r'''
const fs = require('fs');
const helper = require('./helper.js');
const prompt = fs.readFileSync('prompt.txt', 'utf8').trim();
const style = fs.readFileSync('style.txt', 'latin1').trim();      // an encoding, not flags
fs.writeFileSync('out.txt', String(Date.now()));          // written: never part of the version
fs.readFileSync('out.txt');
fs.open('style.txt', (err, fd) => { if (!err) fs.closeSync(fd); });   // a callback, not flags
(async () => {
  const esm = await import('./extra.mjs');
  const out = [];
  for (const q of [prompt, helper.question, style + ' ' + esm.word]) {
    const r = await fetch(process.env.OPENAI_BASE_URL + '/chat/completions', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: process.env.COPULA_MODEL,
                            messages: [{role: 'user', content: q}]})});
    out.push([r.headers.get('x-copula-status'), (await r.json()).choices[0].message.content]);
  }
  console.log(JSON.stringify(out));
})();
'''


@unittest.skipUnless(NODE, 'no working node >= 18')
class NodeTest(LaunchCase):
    def test_node_read_set_defines_the_version(self):
        self.write('agent.js', AGENT)
        self.write('helper.js', "module.exports = {question: 'a fixed question'};\n")
        self.write('prompt.txt', 'first')
        self.write('style.txt', 'plain')
        self.write('extra.mjs', "export const word = 'one';\n")
        child = [NODE, 'agent.js']
        out1, err1 = self.run_ask([], '--mock', '--exp', 'n', child=child)
        self.assertIn('new version', err1)
        out2, err2 = self.run_ask([], '--mock', '--exp', 'n', child=child)
        self.assertIn('unchanged since run', err2)               # out.txt changed: written
        self.write('extra.mjs', "export const word = 'two';\n")
        _, err_esm = self.run_ask([], '--mock', '--exp', 'n', child=child)
        self.assertIn('new version', err_esm)                    # an imported ES module changed
        self.write('style.txt', 'fancy')
        _, err_enc = self.run_ask([], '--mock', '--exp', 'n', child=child)
        self.assertIn('new version', err_enc)                    # read with an encoding
        self.write('helper.js', "module.exports = {question: 'another question'};\n")
        out3, err3 = self.run_ask([], '--mock', '--exp', 'n', child=child)
        self.assertIn('new version', err3)                     # a required module changed
        self.assertEqual(out3[0], ['hit', out1[0][1]])          # the version's slot-1 sample
        self.write('prompt.txt', 'second')
        _, err4 = self.run_ask([], '--mock', '--exp', 'n', child=child)
        self.assertIn('new version', err4)
        runs = os.listdir(os.path.join(self.dir, '.copula', 'runs'))
        self.assertEqual(len(runs), 6)


if __name__ == '__main__':
    unittest.main()
