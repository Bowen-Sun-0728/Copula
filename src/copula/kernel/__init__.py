"""Evaluation runtime v0: a write-once, single-flight sample cache for harness graphs.

Design and decisions: reports/harness_infra_20260924/RUNTIME_V0.md (section 13 is the v0 build
scope; it wins over earlier text). v0 delivers raw results only -- no scoring, estimates or
comparisons. Standard library only; runs under Python 3.9.
"""
