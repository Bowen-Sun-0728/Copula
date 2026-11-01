"""An in-process cache manager on a fresh store, served on a Unix socket, for tests."""
import os
import shutil
import tempfile

from ..manager.flights import Manager
from ..manager.server import AsyncClient, serve
from ..store.entries import EntryStore
from .mock_model import MockModel


class Rig:
    def __init__(self, nonce='test', backends=None, batched=True, **model_kw):
        """`backends` ({name: client}) adds named backends besides the actor (models INTERFACE J3).
        `batched`: the store commits in groups, as the manager CLI runs it."""
        self.dir = tempfile.mkdtemp(prefix='evalrt-')
        self.socket = os.path.join(self.dir, 'm.sock')
        self.model = MockModel(nonce, **model_kw)
        self.backends = dict(backends or {})
        self.batched = batched
        self.store = None
        self.manager = None
        self.server = None
        self.clients = []

    async def start(self, **manager_kw):
        manager_kw.setdefault('backoff_base', 0.0)
        self.store = EntryStore(os.path.join(self.dir, 'entries.db'), batched=self.batched)
        if self.backends:
            manager_kw.setdefault('models', self.backends)
        self.manager = Manager(self.store, self.model, **manager_kw)
        await self.manager.start()
        self.server = await serve(self.manager, self.socket)
        return self

    async def client(self, *tickets, case_keys=None, generation='exec/0'):
        """A connection, with `tickets` registered on it for `case_keys`."""
        client = await AsyncClient.connect(self.socket)
        self.clients.append(client)
        for ticket in tickets:
            await client.call('register_ticket', ticket=ticket, generation=generation,
                              model_rev=self.model.model_rev, case_keys=case_keys)
        return client

    async def close(self):
        for client in self.clients:
            await client.close()
        self.server.close()
        await self.server.wait_closed()
        await self.manager.stop()
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)
