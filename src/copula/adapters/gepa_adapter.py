"""GEPA adapter (docs/guides/optimizers.md). Installed by copula.attach when gepa is imported
(dspy.GEPA imports it too).

- Each `ReflectiveMutationProposer.propose` (sample a parent and a minibatch, evaluate the parent,
  reflect, evaluate the child) is one decision scope: on each case the child reads the parent's
  slot when it can, so their comparison is paired and the child pays only for its edit.
- Reflection (`propose_new_texts`) runs with role 'proposer'.
Units themselves come from the evaluating framework (the DSPy adapter) or from the user's
adapter code (copula.run); a plain gepa adapter without either is one process run.
"""
import functools

from .. import client


def install(gepa):
    from .._attach import note
    from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer

    cls = ReflectiveMutationProposer
    if getattr(cls.propose, '_copula', False):
        return
    propose, propose_new_texts = cls.propose, cls.propose_new_texts

    @functools.wraps(propose)
    def wrapped_propose(self, state, *a, **kw):
        with client.decision():
            return propose(self, state, *a, **kw)

    @functools.wraps(propose_new_texts)
    def wrapped_texts(self, *a, **kw):
        with client.role('proposer'):
            return propose_new_texts(self, *a, **kw)
    wrapped_propose._copula = True
    wrapped_texts._copula = True
    cls.propose = wrapped_propose
    cls.propose_new_texts = wrapped_texts
    note('GEPA adapter on: one decision scope per proposal, proposer role for reflection')
