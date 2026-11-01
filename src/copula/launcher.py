"""The `copula` command (docs/reference/cli.md).

    copula run [--exp NAME] [--recipe R] [--phase search|holdout] [--set k=v ...]
               [--compare k ...] [--annotate k ...] [--replicate k ...] [--case k ...]
               [--route role=rule ...] [--track GLOB ...] [--no-attach] [--server URL]
               [model-server options] -- <command ...>
    copula server [--listen HOST[:PORT]] [model-server options]
    copula explain RUN [--calls] [--json]
    copula ls [--exp NAME]

`run` registers the launched process as one run and serves it through the project's Copula
server: the live one (`copula server`, found in .copula/server.json, or --server URL with
$COPULA_SERVER_TOKEN), else one embedded for this command. The child's OPENAI_BASE_URL,
OPENAI_API_BASE, LLM_BASE_URL and ANTHROPIC_BASE_URL point to the run's URL; the embedded server's
upstream is this process's own OPENAI_BASE_URL unless --upstream or --mock says otherwise. With
--exp the files the program reads define its version: Python children are traced by an
audit hook and attached (copula.attach), Node children by a preload, others by --track GLOB. At
the end one summary says what was reused, what was generated and why (`copula explain` has the
details).
"""
import argparse
import asyncio
import glob
import json
import os
import re
import signal
import socket
import sys

from .coords import CLASSES, PHASES, RECIPES, Classes, norm, parse_argv
from .registry import Registry, check_route, file_digest
from .server import (Remote, Server, add_backend_options, backend_overrides, discover,
                     private_state, serve_forever)

HERE = os.path.dirname(os.path.abspath(__file__))
SHIM = os.path.join(HERE, '_shim')
NODE_SHIM = os.path.join(SHIM, 'node', 'copula-reads.js')


def _parser():
    from . import __version__
    p = argparse.ArgumentParser(prog='copula')
    p.add_argument('--version', action='version', version='copula %s' % __version__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run', help='run a command as one Copula run')
    r.add_argument('--project', default=os.getcwd(),
                   help='the project directory: .copula/ and the files that version runs '
                   '(default: the current directory)')
    r.add_argument('--exp', help='the experiment: runs of one experiment share samples; '
                   'without it the run shares nothing')
    r.add_argument('--recipe', default='dev', choices=sorted(RECIPES),
                   help='default classes of coordinate names for a kind of workload')
    r.add_argument('--phase', default='search', choices=PHASES,
                   help='holdout: a group space of its own, for final numbers')
    r.add_argument('--set', action='append', default=[], metavar='K=V',
                   help='a coordinate of the run (repeatable)')
    for cls, what in (('compare', 'what you compare: runs that differ in it share'),
                      ('replicate', 'a replicate index: runs that differ in it never share'),
                      ('case', 'the input: runs that differ in it never share'),
                      ('annotation', 'recorded, ignored for sharing')):
        r.add_argument('--' + cls, action='append', default=[], metavar='NAME',
                       help='class coordinate NAME as %s %s' % (cls, '(' + what + ')'))
    r.add_argument('--route', action='append', default=[], metavar='ROLE=RULE',
                   help='group of the calls made with a role: run, global, fresh or per:COORD')
    r.add_argument('--track', action='append', default=[], metavar='GLOB',
                   help='files whose content belongs to the version (children not traced)')
    r.add_argument('--no-attach', action='store_true',
                   help='do not attach Python children (no adapters, no unit headers)')
    r.add_argument('--server', help='URL of a Copula server on another host '
                   '(token in $COPULA_SERVER_TOKEN)')
    add_backend_options(r)
    r.add_argument('cmd', nargs=argparse.REMAINDER, help='-- and the command to run')
    s = sub.add_parser('server', help='serve every run of the project from one process')
    s.add_argument('--project', default=os.getcwd(), help='the project directory (default: .)')
    s.add_argument('--listen', help='HOST[:PORT] (default 127.0.0.1, any free port)')
    add_backend_options(s)
    e = sub.add_parser('explain', help='how a run was grouped, and why its calls were generated')
    e.add_argument('run', nargs='?', default='last',
                   help='a run id, a unique prefix of one, a unit id, or last~N: the run N '
                   'before the latest (default: last, the latest run)')
    e.add_argument('--project', default=os.getcwd(), help='the project directory (default: .)')
    e.add_argument('--calls', action='store_true', help='one line per call')
    e.add_argument('--json', action='store_true',
                   help='JSON lines: the run and its units (with --calls: and calls)')
    ls = sub.add_parser('ls', help='list runs')
    ls.add_argument('--project', default=os.getcwd(), help='the project directory (default: .)')
    ls.add_argument('--exp', help='only the runs of this experiment')
    return p


def _say(msg):
    sys.stderr.write('copula: %s\n' % msg)
    sys.stderr.flush()


async def _connect(a, project, state):
    """The server this run is served by: --server, the project's live server, or embedded."""
    if a.server:
        token = os.environ.get('COPULA_SERVER_TOKEN')
        if not token:
            raise SystemExit('copula run: --server needs the token in $COPULA_SERVER_TOKEN '
                             '(see .copula/server.json on the server host)')
        remote = Remote(a.server, token)
        try:
            remote.info = remote.health()
        except OSError as err:
            raise SystemExit('copula run: the server %s does not answer: %s' % (a.server, err))
        live = remote
    else:
        live = discover(state)
    if live is not None:
        over = backend_overrides(a)
        if over:
            raise SystemExit('copula run: the Copula server %s decides the model server, so '
                             '%s cannot be set here: drop %s, or stop the server'
                             % (live.url, ', '.join(over), 'it' if len(over) == 1 else 'them'))
        return live
    return await Server(project, a).start()


def _pylib(state):
    """A directory holding only the `copula` package (a link), for the children's PYTHONPATH:
    never the directory Copula is installed in, which may hold other packages."""
    d = os.path.join(state, 'pylib')
    os.makedirs(d, exist_ok=True)
    link = os.path.join(d, 'copula')
    if os.path.realpath(link) != os.path.realpath(HERE):
        tmp = '%s.%d' % (link, os.getpid())
        try:
            os.symlink(HERE, tmp)
            os.replace(tmp, link)
        except OSError:
            if os.path.lexists(tmp):
                os.unlink(tmp)
    return d


def _child_env(a, url, rec, project, state, reads_file, model):
    env = dict(os.environ)
    for var in ('OPENAI_BASE_URL', 'OPENAI_API_BASE', 'LLM_BASE_URL', 'COPULA_URL'):
        env[var] = url
    for var in ('ANTHROPIC_BASE_URL', 'ANTHROPIC_API_BASE'):   # the SDK appends /v1/...
        env[var] = url[:-len('/v1')]
    env.setdefault('OPENAI_API_KEY', 'copula')
    env.setdefault('ANTHROPIC_API_KEY', 'copula')
    env.update(COPULA_RUN=rec['run_id'], COPULA_PROJECT=project, COPULA_READS_FILE=reads_file,
               COPULA_EXP=a.exp or '', COPULA_MODEL=model or '')
    env.pop('COPULA_SERVER_TOKEN', None)
    if a.exp and not a.no_attach:
        env['COPULA_ATTACH'] = '1'               # auto-attach with --exp
    else:
        env.pop('COPULA_ATTACH', None)
    paths = [SHIM, _pylib(state)]                # the shim records reads; attach only with --exp
    env['PYTHONPATH'] = os.pathsep.join(paths + ([env['PYTHONPATH']]
                                                 if env.get('PYTHONPATH') else []))
    if a.exp and os.path.exists(NODE_SHIM):
        opt = '--require %s' % (json.dumps(NODE_SHIM) if ' ' in NODE_SHIM else NODE_SHIM)
        env['NODE_OPTIONS'] = (env.get('NODE_OPTIONS', '') + ' ' + opt).strip()
    return env


async def _run(a):
    cmd = a.cmd[1:] if a.cmd[:1] == ['--'] else a.cmd
    if not cmd:
        raise SystemExit('copula run: no command (use: copula run [options] -- <command ...>)')
    project = os.path.realpath(a.project)
    state = os.path.join(project, '.copula')
    private_state(state)
    os.makedirs(os.path.join(state, 'runs'), exist_ok=True)
    overrides = {}
    for cls in CLASSES:
        for name in getattr(a, cls):
            overrides[norm(name)] = cls
    routes = {}
    for kv in a.route:
        role, sep, rule = kv.partition('=')
        if not sep:
            raise SystemExit('copula run: --route takes ROLE=RULE, got %r' % kv)
        try:
            routes[role] = check_route(rule)
        except ValueError as err:
            raise SystemExit('copula run: %s' % err)
    coords = parse_argv(cmd[1:])
    for kv in a.set:
        k, _, v = kv.partition('=')
        coords[norm(k)] = v
    if os.environ.get('SLURM_ARRAY_TASK_ID') is not None:
        coords.setdefault('slurm_array_task_id', os.environ['SLURM_ARRAY_TASK_ID'])
    split = Classes(a.recipe, overrides).split(coords)

    conn = await _connect(a, project, state)
    rec = result = None
    rc = 1
    ok = False
    try:
        spec = {'experiment': a.exp, 'recipe': a.recipe, 'phase': a.phase, 'coords': coords,
                'overrides': overrides, 'routes': routes,
                'owner': {'host': socket.gethostname(), 'pid': os.getpid()}}
        if a.exp:                                # the version is decided by digests taken here
            spec['digests'] = {p: file_digest(p) for p in await conn.paths(spec)}
        rec, url = await conn.register(spec)
        run_dir = os.path.join(state, 'runs', rec['run_id'])
        os.makedirs(run_dir, exist_ok=True)
        env = _child_env(a, url, rec, project, state, os.path.join(run_dir, 'reads'),
                         (conn.info or {}).get('default_model'))
        _say('run %s  exp=%s  %s  group %s' % (rec['run_id'], a.exp or '-', rec['rule'],
                                               rec['grp']))
        _say('served by %s' % conn.describe())
        for note in getattr(conn, 'notes', None) or (conn.info or {}).get('notes') or []:
            _say('  note: %s' % note)
        shown = {c: sorted(v) for c, v in split.items() if v}
        if shown:
            _say('coordinates: %s' % json.dumps(shown, sort_keys=True))
        why = {'running': 'run %s of this experiment is still running, so its files are not '
                          'known yet: counted as unchanged',
               'unknown': 'run %s failed, so the files it read are unknown: counted as '
                          'unchanged'}.get(rec.get('match'), 'unchanged since run %s')
        if a.exp and rec['rerun_of']:
            # slot 1 of a version that ran before: its replicate or case values are new
            _say(why % rec['rerun_of'] + (': a rerun (new slot, new samples)' if rec['slot'] != 1
                                          else '; first run of this version with these '
                                          'replicate or case values (new group, new samples)'))
        elif a.exp:
            _say('new version (compared with earlier versions of %s)' % a.exp)
        if a.phase == 'holdout':
            _say('holdout phase: new group space; case disjointness is checked for units only')
        proc = await asyncio.create_subprocess_exec(*cmd, env=env)
        loop = asyncio.get_running_loop()
        # a terminal's Ctrl-C already reaches the child (one process group): not forwarded
        # again; SIGTERM is (`kill <launcher>` must stop the command)
        loop.add_signal_handler(signal.SIGINT, lambda: None)
        loop.add_signal_handler(signal.SIGTERM, proc.send_signal, signal.SIGTERM)
        try:
            rc = await proc.wait()
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(sig)
        if rc < 0:
            rc = 128 - rc                        # killed by a signal: the shell's convention
        ok = rc == 0
    finally:
        try:
            if rec is not None:
                reads = _merge_reads(os.path.join(state, 'runs', rec['run_id'], 'reads'),
                                     project, a.track) if rec['experiment'] else {}
                result = await conn.end_run(rec['run_id'], reads, ok)
        finally:
            await conn.close()
        if rec is not None and result is not None:
            _summary(rec, result, a.exp, ok)
    return rc


def _merge_reads(prefix, project, track):
    """{path: digest} of the project files the run's processes read and never wrote."""
    read, written = set(), set()
    for f in glob.glob(glob.escape(prefix) + '.*'):
        try:
            with open(f, encoding='utf-8', errors='surrogateescape') as fh:
                for line in fh:
                    kind, _, p = line.rstrip('\n').partition(' ')
                    (written if kind == 'W' else read).add(p)
        except OSError:
            pass
    for pattern in track:
        for p in glob.glob(os.path.join(project, pattern), recursive=True):
            if os.path.isfile(p):
                read.add(os.path.realpath(p))
    out = {}
    for p in sorted(read - written):
        d = file_digest(p)
        if d is not None:
            out[p] = d
    return out


def _summary(rec, result, exp, ok):
    st = result.get('stats') or {}
    total = sum(st.values())
    reused = st.get('hit', 0) + st.get('own', 0)
    gen = st.get('generated', 0)
    joined = st.get('waited', 0)
    units = result.get('units') or 0
    failed = (result.get('counts') or {}).get('error', 0)
    _say('run %s done: %d calls: %d reused, %d generated, %d joined%s%s' % (
        rec['run_id'], total + failed, reused, gen, joined,
        ', %d failed' % failed if failed else '', ', %d units' % units if units else ''))
    from .calls import LABELS
    for reason, n, example in result.get('why') or []:
        _say('  %d generated: %s -- %s' % (n, LABELS.get(reason, reason), example))
    for note, n in sorted((result.get('notes') or {}).items()):
        _say('  note (%dx): %s' % (n, note))
    if exp and total and gen == 0 and reused:
        _say('  W1: every sample was read from earlier runs; if you wanted new samples, this '
             'run is a replicate (--set rep=2)')
    if exp and not ok:
        _say('  the command failed: the files it read are recorded as unknown for versioning')
    if gen:
        _say('  details: copula explain %s' % rec['run_id'])


def _registry(a, command):
    """The project's registry, read only: a directory without runs is not made a project."""
    path = os.path.join(os.path.realpath(a.project), '.copula', 'registry.db')
    if not os.path.exists(path):
        raise SystemExit('copula %s: no runs in %s (no .copula/registry.db)'
                         % (command, os.path.realpath(a.project)))
    return Registry(path)


def _explain(a):
    from .calls import LABELS, Explainer, summarize
    state = os.path.join(os.path.realpath(a.project), '.copula')
    reg = _registry(a, 'explain')
    try:
        top = _find_run(reg, a.run)
        if top is None:
            raise SystemExit('copula explain: no run %s (or the prefix is ambiguous)' % a.run)
        if top['parent']:
            top = reg.get(top['parent'])
        ex = Explainer(os.path.join(state, 'calls.db'), reg)
        try:
            items = ex.explain(top['run_id'])
        finally:
            ex.close()
        units = reg.count_units(top['run_id'])
        if a.json:
            rows = [top] + [r for r in reg.runs(top['experiment']) if r['parent'] == top['run_id']]
            for r in rows:
                print(json.dumps({'run': r['run_id'], 'parent': r['parent'],
                                  'experiment': r['experiment'], 'phase': r['phase'],
                                  'granularity': r['granularity'],
                                  'classes': reg.classes(r).split(r['coords']),
                                  'version': r['version'], 'slot': r['slot'], 'group': r['grp'],
                                  'rule': r['rule'], 'status': r['status'],
                                  'rerun_of': r['rerun_of'], 'files_read': len(r['reads'] or {})},
                                 sort_keys=True))
            if a.calls:
                for row, outcome, reason, detail in items:
                    print(json.dumps({'call': row['seq'], 'run': row['run'], 'node': row['node'],
                                      'statuses': row['statuses'], 'ranks': row['ranks'],
                                      'outcome': outcome, 'reason': reason, 'detail': detail},
                                     sort_keys=True))
            return 0
        print('run %s  experiment %s  phase %s  status %s' % (
            top['run_id'], top['experiment'] or '-', top['phase'], top['status']))
        print('  %s; group %s%s' % (top['rule'], top['grp'],
                                   '; rerun of %s' % top['rerun_of'] if top['rerun_of'] else ''))
        split = reg.classes(top).split(top['coords'])
        shown = {c: v for c, v in split.items() if v}
        if shown:
            print('  coordinates: %s' % json.dumps(shown, sort_keys=True))
        print('  files read: %d; units: %d' % (len(top['reads'] or {}), units))
        counts, top_reasons, reasons = summarize(items, limit=99)
        print('  calls: %d: %d reused, %d generated, %d joined, %d errors' % (
            len(items), counts['reused'], counts['generated'], counts['joined'],
            counts['error']))
        for reason, n, example in top_reasons:
            print('  %5d generated: %s\n          e.g. %s' % (n, LABELS.get(reason, reason),
                                                            example))
        if a.calls:
            for row, outcome, reason, detail in items:
                print('  #%-6d %-24s %-10s %-9s %s%s' % (
                    row['seq'], row['run'], row['node'], outcome, row['statuses'],
                    '  %s: %s' % (reason, detail) if reason else ''))
        if not items:
            print('  (no calls in the call log)')
        return 0
    finally:
        reg.close()


def _find_run(reg, name):
    """The run `name` names: an id or a unique prefix, or last / last~N (process runs, newest
    first)."""
    m = re.match(r'^last(?:~(\d+))?$', name)
    if m is None:
        return reg.find(name)
    runs = [r for r in reg.runs() if r['granularity'] == 'process']
    back = int(m.group(1) or 0)
    return runs[-1 - back] if back < len(runs) else None


def _ls(a):
    reg = _registry(a, 'ls')
    row = '%-22s %-12s %-8s %-8s %-8s %-19s %-5s %s'
    print(row % ('RUN', 'EXPERIMENT', 'PHASE', 'KIND', 'STATUS', 'VERSION', 'SLOT', 'GROUP'))
    for r in reg.runs(a.exp):
        print(row % (r['run_id'], r['experiment'] or '-', r['phase'], r['granularity'],
                     r['status'] if r['granularity'] == 'process' else '-', r['version'] or '-',
                     r['slot'] if r['slot'] is not None else '-', r['grp']))
    reg.close()
    return 0


def main(argv=None):
    a = _parser().parse_args(argv)
    if a.command == 'run':
        return asyncio.run(_run(a))
    if a.command == 'server':
        return asyncio.run(serve_forever(a))
    if a.command == 'explain':
        return _explain(a)
    return _ls(a)
