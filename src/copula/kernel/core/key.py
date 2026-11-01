"""Cache keys: the single construction point (RUNTIME_V0 sections 4.2, 12.7 and 13).

    stochastic:    (case, model_rev, node_id, body_hash, rep, sample, generation)
    deterministic: (case, model_rev, node_id, body_hash)

A node is memoryless: its output distribution depends on its input and nothing else. So a sample is
identified by what went in -- the full normalised request, upstream outputs included -- and by `rep`,
which says which sample for that exact input it is. Where a draw sits in a design plays no part.

`rep` is assigned per node and case in canonical design order (`Design.draws`): 1 + the number of
earlier draws of the node, in the same job and case, whose input is identical. Consequences:

- within a job every stochastic key is distinct, even when two different upstream samples happen to
  produce the same text (their inputs then differ only in `rep`);
- any job needing k samples for an input reuses the samples 1..k already stored, whatever design
  shape produced them, and generates only the rest;
- an edited node that re-emits its parent's text leaves its children's inputs unchanged, so they
  hit the cache again (a certified zero).

Every key carries `KEY_VERSION` because the store is never evicted: a format change must never alias
an old entry. Version 2 replaced version 1's coordinate path (`draw`) with `rep`.
"""
import hashlib
import hmac
import re

from .canon import digest
from .request import normalize_request

KEY_VERSION = 2

# exec/<epoch> and compile/<epoch> are shared namespaces; fresh/<ticket> is one ticket's private
# namespace, so a `fresh` job is durable and resumable yet never reads or lends a draw;
# confirm/<group> is shared by the arms of one confirmation and by nothing else; dom/<name> is a
# named independence domain (models INTERFACE J6), shared by every job that names it and by
# nothing else, deterministic requests included; grp/<id> is a group of the packaged system
# (reports/evalrt_sdk_20261009/INTERFACE.md): an independent pool of stochastic samples, while its
# deterministic requests are global (U4).
GENERATION = re.compile(
    r'^(?:(?:exec|compile)/[0-9]+|(?:fresh|confirm|dom|grp)/[A-Za-z0-9_.-]{1,64})$')
DOMAIN_PREFIX = 'dom/'
GROUP_PREFIX = 'grp/'


class KeySpecError(ValueError):
    """An argument that cannot be part of a key."""


def check_generation(generation):
    if not isinstance(generation, str) or not GENERATION.match(generation):
        raise KeySpecError('bad generation label %r' % (generation,))
    return generation


def _check_common(case_key, model_rev, node_id, body_hash):
    for name, value in (('case key', case_key), ('model_rev', model_rev), ('node id', node_id),
                        ('body hash', body_hash)):
        if not isinstance(value, str) or not value:
            raise KeySpecError('%s must be a non-empty string' % name)


def stochastic_key(case_key, model_rev, node_id, body_hash, rep, sample, generation):
    _check_common(case_key, model_rev, node_id, body_hash)
    if type(rep) is not int or rep < 1:
        raise KeySpecError('replicate index must be an int >= 1')
    if type(sample) is not int or sample < 0:
        raise KeySpecError('sample index must be an int >= 0')
    return {'v': KEY_VERSION, 'kind': 'stochastic', 'case': case_key, 'model': model_rev,
            'node': node_id, 'body': body_hash, 'rep': rep, 'sample': sample,
            'gen': check_generation(generation)}


def deterministic_key(case_key, model_rev, node_id, body_hash):
    """A memoised node: no rep, sample or generation, so it runs once per distinct input."""
    _check_common(case_key, model_rev, node_id, body_hash)
    return {'v': KEY_VERSION, 'kind': 'deterministic', 'case': case_key, 'model': model_rev,
            'node': node_id, 'body': body_hash}


def key_hash(key):
    return digest(key)


def replicate_index(earlier_inputs, this_input):
    """`rep` of a draw: 1 + how many earlier draws (canonical order, same node and case) had the
    same input. Entries of `earlier_inputs` that are None (no request was made) never count."""
    return 1 + sum(1 for other in earlier_inputs if other is not None and other == this_input)


def make_key(case_key, model_rev, node_id, body, rep, sample, generation, defaults):
    """Normalise `body` against `defaults` and build the stochastic key.

    Returns (key, key_hash, normalised body). The manager calls this for execution and for the
    forecast walk's lookups alike, so the two can never disagree about a key.
    """
    normal = normalize_request(body, defaults)
    key = stochastic_key(case_key, model_rev, node_id, digest(normal), rep, sample, generation)
    return key, key_hash(key), normal


# -- key v3: pooled samples (v4 graphs) --------------------------------------------------------------
#
# A pool key names what the model is asked, and nothing else: the model revision (weights, tokenizer
# and the sampling table normalize_request folds into the body) and the normalised request. Where
# the call sits in a graph, which node made it, and how many times it was made are not part of it.
# A pool holds any number of samples, numbered 1, 2, ...; which numbers a job reads is decided by the
# manager's per-(partition, program, case) ledger, never by the key. `partition` separates what must
# never share (runs, replay replicates, fresh jobs); it is a generation label.
#
# The sample address names (partition, pool, number) and nothing else (N14, 2026-10-07, which
# superseded D1): the case is not part of it, so two cases share a sample exactly when they send the
# identical request. The ledger stays per case: a consumer is (program, case), and it claims numbers
# for its own repeats. Version 4 carried the case; version 5 drops it again, under its own version
# so that an old address never aliases a new one.
#
# A greedy request (temperature 0 or top_k 1) is deterministic: in a shared partition it has one
# address per pool, with no partition and no number (N16, deterministic + cache crosses domains).

POOL_KEY_VERSION = 5        # sample addresses
POOL_HASH_VERSION = 4       # pool names: (model, request) never held the case, so they did not
                            # change with N14, and tools reading older stores by pool still work


def pool_key(model_rev, body, defaults):
    """(pool hash, normalised body) for a request."""
    if not isinstance(model_rev, str) or not model_rev:
        raise KeySpecError('model_rev must be a non-empty string')
    normal = normalize_request(body, defaults)
    return digest({'v': POOL_HASH_VERSION, 'kind': 'pool', 'model': model_rev,
                   'body': digest(normal)}), normal


def _check_pool(pool_hash):
    if not isinstance(pool_hash, str) or not pool_hash:
        raise KeySpecError('pool hash must be a non-empty string')


def pool_address(partition, pool_hash, index):
    """The store address of sample `index` (>= 1) of a pool inside a partition."""
    check_generation(partition)
    _check_pool(pool_hash)
    if type(index) is not int or index < 1:
        raise KeySpecError('pool index must be an int >= 1')
    return {'v': POOL_KEY_VERSION, 'kind': 'pool_sample', 'part': partition, 'pool': pool_hash,
            'idx': index}


def greedy_address(pool_hash, partition=None):
    """The one store address of a greedy (deterministic) request: in shared partitions it has no
    partition (deterministic + cache crosses domains, N16); in a named domain `dom/<name>` it is
    keyed inside the domain (models INTERFACE J6), a form no earlier address has."""
    _check_pool(pool_hash)
    if partition is None:
        return {'v': POOL_KEY_VERSION, 'kind': 'pool_greedy', 'pool': pool_hash}
    check_generation(partition)
    if not partition.startswith(DOMAIN_PREFIX):
        raise KeySpecError('only a named domain keys greedy requests by partition')
    return {'v': POOL_KEY_VERSION, 'kind': 'pool_greedy', 'pool': pool_hash, 'part': partition}


def is_greedy(normal):
    """Whether a normalised request is deterministic: temperature 0 or top_k 1."""
    return normal.get('temperature') == 0.0 or normal.get('top_k') == 1


def _check_identity(identity):
    if not isinstance(identity, str) or not identity:
        raise KeySpecError('identity must be a non-empty string')


def computed_address(identity, partition=None):
    """The store address of a deterministic registered kind's result (nodes INTERFACE section 2.3;
    N14): `identity` digests the kind, its revision, its params, its port contents and the values of
    the case fields it reads. No partition (a deterministic result may cross partitions) and no case
    (the inputs say everything the result depends on) -- except in a named domain `dom/<name>`,
    where it is kept inside the domain like a greedy model call (models INTERFACE J6), under a form
    no earlier address has."""
    _check_identity(identity)
    if partition is None:
        return digest({'v': 2, 'kind': 'computed', 'id': identity})
    check_generation(partition)
    if not partition.startswith(DOMAIN_PREFIX):
        raise KeySpecError('only a named domain keys computed results by partition')
    return digest({'v': 2, 'kind': 'computed', 'id': identity, 'part': partition})


def computed_pool(identity):
    """The ledger's pool name for draws of a stochastic registered kind with this identity."""
    _check_identity(identity)
    return 'c:' + identity


def computed_draw_address(partition, identity, index):
    """The store address of draw `index` of a stochastic registered kind in a partition."""
    check_generation(partition)
    _check_identity(identity)
    if type(index) is not int or index < 1:
        raise KeySpecError('draw index must be an int >= 1')
    return digest({'v': 1, 'kind': 'computed_draw', 'part': partition, 'id': identity,
                   'idx': index})


def draw_seed(secret, address):
    """The seed of a stochastic registered kind's draw: an HMAC of its address under the store's
    secret, so every job that computes the address computes the same value, and nobody can predict
    it from the address alone."""
    return int(hmac.new(secret.encode('utf-8'), address.encode('utf-8'),
                        hashlib.sha256).hexdigest()[:16], 16)
