"""Injected by `copula run` into Python children (DESIGN.md §4.3, §10.2 item 7; preliminary).

Records the project files the process reads and writes, through a PEP 578 audit hook, as it goes:
one line per new path ('R path' or 'W path') appended to $COPULA_READS_FILE.<pid>, and at exit the
source files of imported modules ('M path'). The launcher merges the files of every process and
keeps what was read and never written. Ignored: anything outside $COPULA_PROJECT, .copula/,
caches, virtual environments and VCS directories. With $COPULA_ATTACH=1 (copula run --exp) it
also calls copula.attach(). Then chains to the next sitecustomize on sys.path, whatever happened
here.
"""
import os
import sys

SKIP_DIRS = ('.copula', '.git', '.hg', '.svn', '.venv', 'venv', 'env', 'node_modules',
             '__pycache__', '.cache', '.mypy_cache', '.pytest_cache', '.tox', 'site-packages')


def _setup():
    root = os.environ.get('COPULA_PROJECT')
    out = os.environ.get('COPULA_READS_FILE')
    if not root or not out or not hasattr(sys, 'addaudithook'):
        return
    root = os.path.realpath(root)
    fd = os.open('%s.%d' % (out, os.getpid()), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    seen = set()
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

    def inside(p):
        if not p.startswith(root + os.sep) or p.endswith(('.pyc', '.pyo')):
            return False
        return not any(part in SKIP_DIRS for part in p[len(root) + 1:].split(os.sep)[:-1])

    def note(kind, p):
        if (kind, p) not in seen:
            seen.add((kind, p))
            try:
                os.write(fd, ('%s %s\n' % (kind, p)).encode('utf-8', 'surrogateescape'))
            except OSError:
                pass

    def hook(event, args):
        if event != 'open':
            return
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        if not isinstance(path, str):
            return
        try:
            p = os.path.realpath(path)
        except Exception:                       # noqa: BLE001
            return
        if not inside(p):
            return
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            mode is None and isinstance(flags, int) and flags & write_flags)
        note('W' if writing else 'R', p)

    sys.addaudithook(hook)

    def modules():
        for m in list(sys.modules.values()):
            f = getattr(m, '__file__', None)
            if isinstance(f, str):
                p = os.path.realpath(f)
                if inside(p):
                    note('M', p)
    import atexit
    atexit.register(modules)


def _chain():
    here = os.path.dirname(os.path.abspath(__file__))
    import importlib.machinery
    import importlib.util
    path = [p for p in sys.path if os.path.abspath(p or '.') != here]
    spec = importlib.machinery.PathFinder.find_spec('sitecustomize', path)
    if spec is not None and spec.loader is not None:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


def _attach():
    if os.environ.get('COPULA_ATTACH') == '1':
        import copula._attach
        copula._attach.attach()


try:
    _setup()
except Exception:                               # noqa: BLE001 -- never break the child
    pass
try:
    _attach()
except Exception as err:                        # noqa: BLE001
    sys.stderr.write('copula: attach failed: %r (the program runs without it)\n' % (err,))
finally:
    _chain()
