"""Pure functions shared by every process: canonical JSON, requests, draws, designs and keys.

`key.make_key` is the single construction point for cache keys (RUNTIME_V0 section 4.2): the
forecast walk and the manager must never build a key any other way.
"""
