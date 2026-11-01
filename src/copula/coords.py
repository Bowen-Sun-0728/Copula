"""Coordinates, their classes and the recipes that class them (DESIGN.md §4.2, §5; change V1).

A coordinate is a name and a value attached to a run. Its class decides what it does to sharing:

    compare     left out of the group key, so runs that differ only in it share (part of version)
    replicate   in the group key, so runs that differ in it are separate
    case        in the group key (requests of different inputs almost never coincide anyway)
    annotation  recorded, never keyed

Every name the recipe does not know is replicate: the safe direction (principle 2).
"""
import re

CLASSES = ('compare', 'replicate', 'case', 'annotation')

COMPARE = ('variant', 'version', 'candidate', 'config', 'arm', 'method', 'budget', 'prompt')
REPLICATE = ('seed', 'trial', 'rep', 'repeat', 'replicate', 'attempt', 'epoch', 'rollout', 'shard',
             'chunk', 'fold', 'slurm_array_task_id')
CASE = ('case', 'example', 'sample_id', 'doc_id', 'task_id', 'instance', 'problem')
ANNOTATION = ('iteration', 'metric_key', 'workflow_name', 'group_id', 'trace_id', 'session_id',
              'thread_id', 'run_name', 'name', 'o', 'out', 'outdir', 'output', 'output_dir',
              'log_dir', 'logdir', 'save_dir', 'savedir')
# output, log and save paths under other spellings (--output-path, --log_file, ...)
ANNOTATION_RE = re.compile(r'^(o|out|output|log|logs|save|ckpt)(_?(dir|path|file|prefix))?$')

# Names a recipe adds to the defaults (DESIGN §5). A recipe never moves an unknown name out of
# replicate except through these lists or the user's overrides.
RECIPES = {
    'dev': {},
    'eval': {'compare': ('task', 'task_identifier', 'model_config', 'solver'),
             'case': ('sample', 'test', 'test_idx')},
    'optimize': {'annotation': ('parent',)},
    'ab': {},
    'sweep': {},
    'tts': {'compare': ('n', 'beam_width', 'num_beams')},
    'sdg': {},
    'rl': {},
}
PHASES = ('search', 'holdout')


def norm(name):
    """A coordinate name as classed: lower case, no leading dashes, '-' as '_'."""
    return str(name).strip().lstrip('-').replace('-', '_').lower()


class Classes:
    """The classing rules of one experiment: a recipe plus the user's overrides."""

    def __init__(self, recipe='dev', overrides=None):
        if recipe not in RECIPES:
            raise ValueError('unknown recipe %r (known: %s)' % (recipe, ', '.join(RECIPES)))
        self.recipe = recipe
        self.overrides = {}
        for name, cls in (overrides or {}).items():
            if cls not in CLASSES:
                raise ValueError('unknown class %r for %r' % (cls, name))
            self.overrides[norm(name)] = cls

    def of(self, name):
        n = norm(name)
        if n in self.overrides:
            return self.overrides[n]
        extra = RECIPES[self.recipe]
        for cls in ('compare', 'case', 'annotation', 'replicate'):
            if n in extra.get(cls, ()):
                return cls
        if n in COMPARE:
            return 'compare'
        if n in CASE:
            return 'case'
        if n in ANNOTATION or ANNOTATION_RE.match(n):
            return 'annotation'
        return 'replicate'                       # REPLICATE and every unknown name

    def split(self, coords):
        """{class: {name: value}} of `coords` (names normalised)."""
        out = {c: {} for c in CLASSES}
        for name, value in (coords or {}).items():
            out[self.of(name)][norm(name)] = value
        return out


HYDRA = re.compile(r'^[A-Za-z_+~][\w.+~/-]*=')


def parse_argv(argv):
    """Coordinates found in a command line: Hydra `k=v`, `--k v`, `--k=v` and `-o v`. Positional
    words and bare flags give none; a lone `--` ends the options. A Hydra override after a flag is
    an override, not the flag's value."""
    out = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--':
            break
        if a.startswith('-') and len(a) > 1 and not _number(a):
            body = a.lstrip('-')
            if '=' in body:
                k, v = body.split('=', 1)
                if k:
                    out[norm(k)] = v
            elif body and i + 1 < len(argv) and not HYDRA.match(argv[i + 1]) and not (
                    argv[i + 1].startswith('-') and not _number(argv[i + 1])):
                out[norm(body)] = argv[i + 1]
                i += 1
        elif HYDRA.match(a):
            k, v = a.split('=', 1)
            out[norm(k.lstrip('+~'))] = v
        i += 1
    return out


def _number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
