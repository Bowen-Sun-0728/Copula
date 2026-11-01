"""The documentation is checked against the code (docs/**/*.md, README.md, mkdocs.yml).

Examples run as written. A page is a script, run in a fresh project directory:
    ```python title="agent.py"      a fenced block with a title writes that file
    ```console                      each `$ ` line is run with bash; its output (stdout and stderr)
                                    must equal the lines below it, up to the next `$ ` line
    <!-- docs-test: run CMD -->     a hidden step whose output is not compared (an edit the text
                                    describes, a cleanup)
    <!-- docs-test: bg CMD -->      start CMD in the background; stopped when the page ends
    <!-- docs-test: wait PATH -->   wait (up to 30 s) until PATH exists
    <!-- docs-test: skip -->        the next console block is not run (a real model server, a
                                    cluster, an interactive step)
    <!-- docs-test: fails -->       the commands of the next console block exit non-zero (every
                                    other command must exit 0)
Ids that differ from run to run (runs, units, versions, groups, tokens, ports, and mock replies,
which depend on the group) are compared by
pattern: across a page, each id shown must stand for one id of the output, and two ids shown for
two. A line `...` matches any number of lines.

    PYTHONPATH=src:. python -m unittest tests.test_docs
    COPULA_DOCS_UPDATE=1 PYTHONPATH=src:. python -m unittest tests.test_docs    # rewrite outputs

The examples need the `openai` package in the interpreter that runs them: $COPULA_DOCS_PYTHON, else
this one; pages whose files import a missing package are skipped.
"""
import glob
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
import unittest

from tests.copula_fixtures.launch import env_with

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, 'docs')
SRC = os.path.join(ROOT, 'src')
PAGES = sorted(glob.glob(os.path.join(DOCS, '**', '*.md'), recursive=True))
UPDATE = bool(os.environ.get('COPULA_DOCS_UPDATE'))

FENCE = re.compile(r'^(`{3,})(\w*)(.*)$')
HIDDEN = re.compile(r'^<!-- docs-test: (\w+)(?: (.*?))? -->$')
IDS = [('run', re.compile(r'\br-[0-9a-f]{16}\b')), ('unit', re.compile(r'\bu-[0-9a-f]{20}\b')),
       ('version', re.compile(r'\bv-[0-9a-f]{16}\b')), ('group', re.compile(r'\bg-[0-9a-f]{20,24}\b')),
       ('token', re.compile(r'\b[0-9a-f]{40,64}\b')), ('mock', re.compile(r'\bmock:[0-9a-f]{24}\b')),
       ('port', re.compile(r'(?<=127\.0\.0\.1:)\d+|(?<=localhost:)\d+'))]


def blocks(text):
    """The steps of a page: ('file', name, body), ('console', body, skip), ('run'|'bg'|'wait', arg)
    and their line spans, in order."""
    lines = text.split('\n')
    out, i, skip, fails = [], 0, False, False
    while i < len(lines):
        m = HIDDEN.match(lines[i].strip())
        if m:
            if m.group(1) == 'skip':
                skip = True
            elif m.group(1) == 'fails':
                fails = True
            else:
                out.append((m.group(1), m.group(2) or '', None))
            i += 1
            continue
        f = FENCE.match(lines[i])
        if not f:
            i += 1
            continue
        ticks, lang, rest = f.groups()
        j = i + 1
        while j < len(lines) and lines[j].rstrip() != ticks:
            j += 1
        body = lines[i + 1:j]
        title = re.search(r'title="([^"]+)"', rest)
        if title:
            out.append(('file', title.group(1), body))
        elif lang == 'console':
            out.append(('console', body, (i + 1, j), skip, fails))
            skip = fails = False
        i = j + 1
    return out


def commands(body):
    """[(command, expected lines)] of a console block."""
    out = []
    for line in body:
        if line.startswith('$ '):
            out.append((line[2:], []))
        elif out:
            out[-1][1].append(line)
    return out


ID = re.compile('|'.join('(?P<%s>%s)' % (k, rx.pattern) for k, rx in IDS))


def split_ids(line):
    """(text with ids replaced by their kind, [(kind, id)])."""
    found = []

    def sub(m):
        found.append((m.lastgroup, m.group(0)))
        return '<%s>' % m.lastgroup
    return ID.sub(sub, line.rstrip()), found


def bind(binding, pairs):
    """`binding` extended by the id pairs (page id -> output id, one to one), or None."""
    fwd, back = dict(binding[0]), dict(binding[1])
    for (kind, a), (_, b) in pairs:
        if fwd.setdefault((kind, a), b) != b or back.setdefault((kind, b), a) != a:
            return None
    return fwd, back


def match(expected, actual, binding):
    """The binding under which `actual` fits `expected` (a line '...' stands for any lines), or
    None."""
    if not expected:
        return binding if not actual else None
    if expected[0] == '...':
        for k in range(len(actual) + 1):
            got = match(expected[1:], actual[k:], binding)
            if got is not None:
                return got
        return None
    if not actual:
        return None
    (te, ie), (ta, ia) = split_ids(expected[0]), split_ids(actual[0])
    if te != ta or [k for k, _ in ie] != [k for k, _ in ia]:
        return None
    b = bind(binding, list(zip(ie, ia)))
    return None if b is None else match(expected[1:], actual[1:], b)


def interpreter():
    return os.environ.get('COPULA_DOCS_PYTHON') or sys.executable


def has_module(py, name):
    return subprocess.run([py, '-c', 'import %s' % name], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


class Page:
    def __init__(self, path):
        self.path = path
        with open(path, encoding='utf-8') as f:
            self.text = f.read()
        self.steps = blocks(self.text)

    def runnable(self):
        return any(s[0] == 'console' and not s[3] for s in self.steps)

    def imports(self):
        mods = set()
        for s in self.steps:
            if s[0] == 'file' and s[1].endswith('.py'):
                for line in s[2]:
                    m = re.match(r'\s*(?:from|import) ([A-Za-z_]\w*)', line)
                    if m:
                        mods.add(m.group(1))
        return mods


class DocExamplesTest(unittest.TestCase):
    maxDiff = None

    def run_page(self, page):
        py = interpreter()
        local = {os.path.splitext(s[1])[0] for s in page.steps if s[0] == 'file'}
        std = set(getattr(sys, 'stdlib_module_names', ())) | {'copula', 'dspy', 'gepa'} | local
        missing = sorted(m for m in page.imports() - std if not has_module(py, m))
        if missing:
            self.skipTest('%s needs %s in %s' % (os.path.relpath(page.path, ROOT),
                                                 ', '.join(missing), py))
        work = tempfile.mkdtemp(prefix='copula-docs-')
        bindir = os.path.join(work, '.bin')
        os.makedirs(bindir)
        with open(os.path.join(bindir, 'copula'), 'w') as f:
            f.write('#!/bin/sh\nexec "%s" -m copula "$@"\n' % py)
        os.chmod(os.path.join(bindir, 'copula'), 0o755)
        for name in ('python', 'python3'):           # a wrapper: a venv's python keeps its venv
            with open(os.path.join(bindir, name), 'w') as f:
                f.write('#!/bin/sh\nexec "%s" "$@"\n' % py)
            os.chmod(os.path.join(bindir, name), 0o755)
        proj = os.path.join(work, 'project')
        os.makedirs(proj)
        env = env_with(PATH=bindir + os.pathsep + os.environ.get('PATH', ''), PYTHONPATH=SRC,
                       PYTHONUNBUFFERED='1', HOME=work, COLUMNS='100')
        background = []
        updates = []
        binding = ({}, {})
        try:
            for step in page.steps:
                kind = step[0]
                if kind == 'file':
                    target = os.path.join(proj, step[1])
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with open(target, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(step[2]) + '\n')
                elif kind == 'run':
                    p = subprocess.run(['bash', '-c', step[1]], cwd=proj, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
                    self.assertEqual(p.returncode, 0, '%s: %s\n%s' % (page.path, step[1],
                                                                     p.stdout.decode()))
                elif kind == 'bg':
                    background.append(subprocess.Popen(
                        ['bash', '-c', 'exec ' + step[1]], cwd=proj, env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        start_new_session=True))
                elif kind == 'wait':
                    deadline = time.time() + 30
                    while not os.path.exists(os.path.join(proj, step[1])):
                        self.assertLess(time.time(), deadline, 'waited for %s' % step[1])
                        time.sleep(0.1)
                elif kind == 'console' and not step[3]:
                    new_body = []
                    for cmd, expected in commands(step[1]):
                        p = subprocess.run(['bash', '-c', cmd], cwd=proj, env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           timeout=300)
                        actual = p.stdout.decode('utf-8', 'replace').rstrip('\n').split('\n')
                        if actual == ['']:
                            actual = []
                        actual = [a.replace(proj, '/home/me/project') for a in actual]
                        self.assertEqual(p.returncode != 0, step[4], '%s: `%s` exited %d:\n%s' % (
                            os.path.relpath(page.path, ROOT), cmd, p.returncode,
                            '\n'.join(actual)))
                        new_body += ['$ ' + cmd] + actual
                        if UPDATE:
                            continue
                        got = match(expected, actual, binding)
                        self.assertIsNotNone(got, '%s: `%s` printed\n%s\n\nthe page shows\n%s' % (
                            os.path.relpath(page.path, ROOT), cmd, '\n'.join(actual),
                            '\n'.join(expected)))
                        binding = got
                    updates.append((step[2], new_body))
        finally:
            for b in background:
                try:
                    os.killpg(b.pid, signal.SIGTERM)
                    b.wait(timeout=20)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    os.killpg(b.pid, signal.SIGKILL)
            shutil.rmtree(work, ignore_errors=True)
        if UPDATE and updates:
            lines = page.text.split('\n')
            for (start, end), body in reversed(updates):
                old = lines[start:end]
                keep = [l for l in old if l == '...']
                lines[start:end] = body if not keep else old   # pages with '...' are kept
            with open(page.path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))


def _make(page):
    def test(self):
        self.run_page(page)
    return test


for _path in PAGES:
    _page = Page(_path)
    if _page.runnable():
        _name = 'test_' + re.sub(r'\W', '_', os.path.relpath(_path, DOCS)[:-3])
        setattr(DocExamplesTest, _name, _make(_page))


def slug(text):
    """The anchor MkDocs (Python-Markdown toc) gives a heading."""
    text = re.sub(r'`([^`]*)`', r'\1', text)
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    return re.sub(r'[-\s]+', '-', text)


def anchors(path):
    out = set()
    fenced = False
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('```'):
                fenced = not fenced
            if fenced:
                continue
            m = re.match(r'^#{1,6} (.*?)\s*(\{#([\w-]+)\})?\s*$', line)
            if m:
                out.add(m.group(3) or slug(m.group(1)))
            out.update(re.findall(r'<a id="([\w-]+)"', line))
    return out


def prose(path):
    """The page without fenced blocks."""
    with open(path, encoding='utf-8') as f:
        text = f.read()
    return re.sub(r'^(`{3,}).*?^\1\s*$', '', text, flags=re.M | re.S)


class DocReferenceTest(unittest.TestCase):
    def test_links_resolve(self):
        bad = []
        for page in PAGES + [os.path.join(ROOT, 'README.md')]:
            for target in re.findall(r'\]\(([^)\s]+)\)', prose(page)):
                if re.match(r'^[a-z]+:', target):
                    continue
                path, _, anchor = target.partition('#')
                full = os.path.normpath(os.path.join(os.path.dirname(page), path)) if path \
                    else page
                if not os.path.exists(full):
                    bad.append('%s: %s (no file)' % (os.path.relpath(page, ROOT), target))
                elif anchor and full.endswith('.md') and anchor not in anchors(full):
                    bad.append('%s: %s (no anchor)' % (os.path.relpath(page, ROOT), target))
        self.assertEqual(bad, [])

    def test_nav_lists_every_page(self):
        with open(os.path.join(ROOT, 'mkdocs.yml'), encoding='utf-8') as f:
            nav = set(re.findall(r'^\s*-\s*(?:[^:\n]+:\s*)?([\w/.-]+\.md)\s*$', f.read(), re.M))
        pages = {os.path.relpath(p, DOCS) for p in PAGES}
        self.assertEqual(sorted(pages - nav), [], 'pages missing from the nav')
        self.assertEqual(sorted(nav - pages), [], 'nav entries without a page')

    def test_cli_reference_matches_the_parser(self):
        from copula.launcher import _parser
        p = _parser()
        sub = next(a for a in p._actions if a.__class__.__name__ == '_SubParsersAction')
        real = {'--help', '--version'}
        for name, sp in sub.choices.items():
            for action in sp._actions:
                real.update(o for o in action.option_strings if o.startswith('--'))
        text = prose(os.path.join(DOCS, 'reference', 'cli.md'))
        shown = {o for span in re.findall(r'`([^`\n]+)`', text)
                 for o in re.findall(r'(?<![\w-])(--[a-z][a-z-]*)', span)}
        self.assertEqual(sorted(real - shown - {'--help'}), [], 'options not documented')
        self.assertEqual(sorted(shown - real), [], 'documented options that do not exist')
        for name in sub.choices:
            self.assertIn('## `copula %s`' % name, text)

    def test_coordinate_names_match_the_code(self):
        from copula import coords
        text = prose(os.path.join(DOCS, 'reference', 'coordinates.md'))
        rows = {}
        for line in text.split('\n'):
            m = re.match(r'^\| \*\*(\w+)\*\* \|(.*)\|\s*$', line)
            if m:
                rows[m.group(1)] = set(re.findall(r'`([^`]+)`', m.group(2)))
        for cls, names in (('compare', coords.COMPARE), ('replicate', coords.REPLICATE),
                           ('case', coords.CASE), ('annotation', coords.ANNOTATION)):
            self.assertEqual(sorted(set(names) - rows.get(cls, set())), [], cls)
        for recipe, extra in coords.RECIPES.items():
            m = re.search(r'^\| `%s` \|(.*)\|\s*$' % recipe, text, re.M)
            self.assertIsNotNone(m, 'recipe %s' % recipe)
            for names in extra.values():
                for n in names:
                    self.assertIn('`%s`' % n, m.group(1), 'recipe %s: %s' % (recipe, n))

    def test_every_explain_reason_has_an_entry(self):
        from copula.calls import LABELS
        page = os.path.join(DOCS, 'guides', 'why-generated.md')
        have = anchors(page)
        self.assertEqual(sorted(r for r in LABELS if r.replace('_', '-') not in have
                                and r not in have), [])


if __name__ == '__main__':
    unittest.main()
