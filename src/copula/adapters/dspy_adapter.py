"""DSPy adapter (docs/guides/optimizers.md). Installed by copula.attach when dspy is
imported; the program is not changed.

- Units: every top-level module call (a `dspy.Module` __call__/acall not inside another one) is a
  unit with coordinates
      candidate      = hash of the module's state (instructions, demos, signatures: dump_state)
      case           = hash of the call's inputs
      optimizer_seed = the seed of the enclosing (or last) optimizer compile, a replicate
                       coordinate of its own, so it never replaces the process's own `seed`.
  A program re-run on the same inputs is the next ordinal of that unit, so it reads new samples;
  a changed candidate on the same case shares the samples of the calls its edit did not change.
- Holdout: an `Evaluate` outside any compile, after a compile in this process, on examples none
  of which the optimizer was given (trainset, valset) or ran, is an inferred holdout evaluation
  (a new group space). One that overlaps them is scored on selection data: reported once (W5b)
  and left in the search phase. If the registry knows the cases from the search phase of another
  process, the proxy falls back to the search phase with the same note.
- Cache: DSPy's response cache (memory and disk) returns one stored sample for every repeat of a
  request, also across runs and seeds; it is turned off (W11). Copula reuses samples instead.
- Callbacks: a DSPy callback, added to the global configuration without taking DSPy's
  configuration ownership; `dspy.configure(callbacks=...)` and `dspy.context(callbacks=...)` keep it.
"""
import functools
import hashlib
import json

from .. import client


def _hid(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def case_of(inputs):
    """The case of a module call's inputs ({'args': ..., 'kwargs': ...} or a plain dict)."""
    if isinstance(inputs, dict) and set(inputs) <= {'args', 'kwargs'}:
        args, kwargs = inputs.get('args') or (), inputs.get('kwargs') or {}
        return _hid(dict(kwargs) if not args else {'args': list(args), 'kwargs': kwargs})
    return _hid(inputs)


def _example_case(ex):
    return _hid(dict(ex.inputs().items()))


def install(dspy):
    from .._attach import note
    from dspy.utils.callback import BaseCallback

    try:
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
        note('DSPy response cache turned off (W11): it returned one stored sample for every '
             'repeat; repeated evaluations are now new samples, reused across runs by Copula')
    except Exception as err:                    # noqa: BLE001
        note('could not turn off the DSPy cache (W8): %r' % (err,))

    class CopulaCallback(BaseCallback):
        def __init__(self):
            self.tokens = {}
            self.seen = set()                   # cases used inside compiles of this process
            self.compiled = False
            self.seed = None

        def _set(self, call_id, token):
            self.tokens[call_id] = token

        def _reset(self, call_id):
            token = self.tokens.pop(call_id, None)
            if token is not None:
                try:
                    client._pop(token)
                except ValueError:              # set in another context: nothing to undo here
                    pass

        def on_compile_start(self, call_id, instance, inputs):
            coords = dict(client.current().get('coords', {}))
            seed = getattr(instance, 'seed', None)
            if seed is not None:
                coords['optimizer_seed'] = seed
            self.seed = seed
            self.compiled = True
            for key in ('trainset', 'valset'):         # given to the optimizer: not held out
                for ex in inputs.get(key) or []:
                    try:
                        self.seen.add(_example_case(ex))
                    except Exception:           # noqa: BLE001
                        pass
            self._set(call_id, client._push(coords=coords, dspy_compile=True))

        def on_compile_end(self, call_id, outputs, exception=None):
            self._reset(call_id)

        def on_evaluate_start(self, call_id, instance, inputs):
            cur = client.current()
            if cur.get('dspy_compile') or cur.get('phase') or not self.compiled:
                return
            coords = dict(cur.get('coords', {}))
            if self.seed is not None:              # the evaluated program's optimizer seed
                coords.setdefault('optimizer_seed', self.seed)
            devset = inputs.get('devset') or getattr(instance, 'devset', None) or []
            try:
                cases = {_example_case(ex) for ex in devset}
            except Exception:                   # noqa: BLE001 -- no input keys: cannot tell
                self._set(call_id, client._push(coords=coords))
                return
            overlap = cases & self.seen
            if overlap:
                note('W5b: an Evaluate after compile scores %d of %d examples the optimizer '
                     'used; those scores are optimistic' % (len(overlap), len(cases)))
                self._set(call_id, client._push(coords=coords))
                return
            self._set(call_id, client._push(coords=coords, phase='holdout', phase_inferred=True))

        def on_evaluate_end(self, call_id, outputs, exception=None):
            self._reset(call_id)

        def on_module_start(self, call_id, instance, inputs):
            cur = client.current()
            if cur.get('dspy_unit'):
                return
            try:
                cand = _hid(instance.dump_state())
            except Exception:                   # noqa: BLE001
                cand = type(instance).__qualname__
            case = case_of(inputs)
            if cur.get('dspy_compile'):
                self.seen.add(case)
            self._set(call_id, client._open_unit({'candidate': cand, 'case': case},
                                                 dspy_unit=True))

        def on_module_end(self, call_id, outputs, exception=None):
            self._reset(call_id)

    cb = CopulaCallback()
    settings = dspy.settings
    cfg = getattr(settings, 'main_thread_config', None)
    if isinstance(cfg, dict):                   # without claiming configure's ownership
        cfg['callbacks'] = list(cfg.get('callbacks') or []) + [cb]
    else:
        settings.configure(callbacks=list(settings.get('callbacks', None) or []) + [cb])

    def keep(fn):
        if getattr(fn, '_copula', False):
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if 'callbacks' in kwargs and cb not in (kwargs['callbacks'] or []):
                kwargs['callbacks'] = list(kwargs['callbacks'] or []) + [cb]
            return fn(*args, **kwargs)
        wrapper._copula = True
        return wrapper
    for name in ('configure', 'context'):
        if callable(getattr(dspy, name, None)):
            setattr(dspy, name, keep(getattr(dspy, name)))
    note('DSPy adapter on: units from module calls, candidates and seeds from optimizers, '
         'holdout evaluations')
    return cb
