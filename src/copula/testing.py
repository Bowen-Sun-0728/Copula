"""Test helpers: a mock model whose replies are scripted by content (preliminary).

`ScriptedMock(nonce, script)` is the kernel's MockModel, except that a reply is
`script(body, token)`, where `token` is the mock's per-sample content (a function of the sample's
key), so scripted replies still differ from sample to sample. `script` returns text, a dict of
sample fields (tool_calls, finish_reason, ...), or None for the plain mock reply.
"""
import importlib
import importlib.util
import os

from .kernel.testing.mock_model import MockModel


class ScriptedMock(MockModel):
    def __init__(self, nonce, script=None, **kw):
        super().__init__(nonce, **kw)
        self.script = script

    async def generate(self, body, key_hash):
        sample = await super().generate(body, key_hash)
        if self.script is None or sample.get('content') is None:
            return sample
        out = self.script(body, self.sample_for(key_hash))
        if isinstance(out, dict):
            sample.update(out)
            if sample.get('content') is None:
                sample['content'] = ''
        elif out is not None:
            sample['content'] = out
        return sample


def load_script(spec):
    """`FILE.py:FUNC` or `MODULE:FUNC` -> the function (None for None)."""
    if not spec:
        return None
    where, _, name = spec.rpartition(':')
    if where.endswith('.py') and os.path.exists(where):
        mod_spec = importlib.util.spec_from_file_location('copula_mock_script', where)
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(where)
    return getattr(mod, name)
