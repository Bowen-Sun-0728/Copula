"""Copula: sound sharing of stochastic samples across reruns, edits and comparisons of agent
programs.

Two ways in:
- `copula run [--exp NAME] -- <command>` serves an unmodified program's model calls (OpenAI Chat
  Completions and Responses, Anthropic Messages) through a Copula server, and attaches Python
  children (`copula.attach`) with framework adapters (DSPy, GEPA). `copula.run(**coords)`,
  `copula.holdout()`, `copula.decision()` and `copula.role()` refine what a call belongs to.
- The kernel API of the paper: `copula.Session` with `s.group(name)`, `s.run(group)`,
  `s.wrap(OpenAI())` and registered nodes (copula.kernel.sdk).

The kernel (copula.kernel) is the paper's frozen evalrt tree, byte for byte. Importing copula needs
only the standard library.
"""
from .client import chat, decision, headers, holdout, role, run  # noqa: F401

__version__ = '0.1.0.dev0'
_KERNEL = ('Session', 'Group', 'Run', 'Namespace', 'SDKError')


def attach(**kw):
    """See copula._attach.attach."""
    from ._attach import attach as _attach
    return _attach(**kw)


def __getattr__(name):
    if name in _KERNEL:
        from .kernel import sdk
        return getattr(sdk, name)
    raise AttributeError('module %r has no attribute %r' % (__name__, name))
