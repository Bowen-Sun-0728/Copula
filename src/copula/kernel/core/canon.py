"""Canonical JSON and content hashes. Every hash the runtime stores goes through here."""
import hashlib
import hmac
import json


def canonical(obj):
    """UTF-8 bytes of `obj` as canonical JSON: sorted keys, no whitespace, NaN refused.

    The same convention as `retrieval_graph.revision`, so a graph hashes identically here.
    """
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(obj):
    """sha256 hex of the canonical JSON of `obj`."""
    return hashlib.sha256(canonical(obj)).hexdigest()


def keyed_digest(secret, obj):
    """HMAC-sha256 hex of the canonical JSON of `obj` under `secret` (bytes).

    Case keys use this, never a plain hash: a plain hash of (input, reference) handed to a job
    process lets anyone holding the input enumerate a small reference space (a label, a letter)
    and recover the answer key (RUNTIME_V0 section 12.7).
    """
    if not isinstance(secret, (bytes, bytearray)) or len(secret) < 16:
        raise ValueError('case-key secret must be at least 16 bytes')
    return hmac.new(bytes(secret), canonical(obj), hashlib.sha256).hexdigest()
