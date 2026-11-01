"""The manager's socket front end, and the client that job processes and the control plane use.

The wire is `evalrt.rpc`. When a connection drops its pending requests are abandoned but their
flights are not: flights are shielded and run to completion, because the manager owns generation.
"""
from .. import rpc
from .flights import ManagerError

LINE_LIMIT = rpc.LINE_LIMIT


def _scope(req, connection):
    """One attempt of one ticket on one connection. A client may name its attempt (`attempt`, the
    packaged system: a run resumed on the same connection is a new attempt); absent, the scope is
    (connection, ticket) as before."""
    if req.get('attempt') is None:
        return (connection, req['ticket'])
    return (connection, req['ticket'], str(req['attempt']))


NOT_DURABLE = ('lease_renew', 'lease_release')     # replies that report nothing stored


async def _op(manager, req, connection):
    """One request; the reply leaves only once every store write made so far is durable (group
    commit: the manager's `durable`). A lease granted in a reply that never leaves (the commit
    failed) is released: its requester never learns the token; one whose reply leaves starts its
    clocks then. Renewals and releases report nothing stored, so they do not wait for a commit
    (nor fail with someone else's)."""
    out = await _answer(manager, req, connection)
    if req['op'] in NOT_DURABLE:
        return out
    try:
        await manager.durable()
    except BaseException:
        if isinstance(out, dict) and out.get('lease'):
            manager.lease_release(out['lease'], connection)
        raise
    if isinstance(out, dict) and out.get('lease'):
        manager.lease_started(out['lease'])
    return out


async def _answer(manager, req, connection):
    op = req['op']
    if op == 'ping':
        out = {'model_rev': manager.model.model_rev, 'models': manager.model_table(),
               'leases': {'ttl': manager.lease_ttl, 'max_hold': True, 'renew': True}}
        if manager.config is not None:
            out['config'] = manager.config
        return out
    if op == 'register_ticket':
        manager.register_ticket(req['ticket'], req['generation'], req['model_rev'], req['case_keys'],
                                cache_salt=req.get('cache_salt'))
        return {}
    if op == 'register_pool_ticket':
        manager.register_pool_ticket(req['ticket'], req['partitions'], req['consumer'],
                                     req['model_rev'], req['case_keys'],
                                     cache_salt=req.get('cache_salt'), models=req.get('models'))
        return {}
    if op == 'draw':
        return await manager.draw(req['ticket'], req['node'], req['body'], case=req.get('case'),
                                  partition=req.get('partition'), rank=req.get('rank'),
                                  role=req.get('role'), scope=_scope(req, connection),
                                  model=req.get('model'), randomness=req.get('randomness'))
    if op == 'memo_get':
        if req.get('lease'):
            return await manager.memo_lease(req['ticket'], req['case'], req['identity'],
                                            connection, partition=req.get('partition'),
                                            max_hold=req.get('max_hold'),
                                            renew=req.get('renew') is True)
        return manager.memo_get(req['ticket'], req['case'], req['identity'],
                                partition=req.get('partition'))
    if op == 'memo_put':
        return manager.memo_put(req['ticket'], req['case'], req['identity'], req['value'],
                                req['kind'], req['kind_rev'], partition=req.get('partition'))
    if op == 'kind_draw':
        if req.get('lease'):
            return await manager.kind_draw_lease(req['ticket'], req['case'], req['partition'],
                                                 req['identity'], connection,
                                                 role=req.get('role'),
                                                 scope=_scope(req, connection),
                                                 max_hold=req.get('max_hold'),
                                                 renew=req.get('renew') is True)
        return manager.kind_draw(req['ticket'], req['case'], req['partition'], req['identity'],
                                 role=req.get('role'), scope=_scope(req, connection))
    if op == 'end_attempt':
        return manager.end_scope(_scope(req, connection))
    if op == 'kind_put':
        return manager.kind_put(req['ticket'], req['case'], req['partition'], req['identity'],
                                req['index'], req['value'], req['kind'], req['kind_rev'])
    if op == 'lease_release':
        return manager.lease_release(req['lease'], connection)
    if op == 'lease_renew':
        return manager.lease_renew(req['lease'], connection)
    if op == 'reopen':
        return manager.reopen(req['ticket'])
    if op == 'get_or_generate':
        return await manager.get_or_generate(req['ticket'], req['case'], req['node'], req['rep'],
                                             req['sample'], req['body'],
                                             scope=(connection, req['ticket']))
    if op == 'cancel':
        return manager.cancel(req['ticket'])
    if op == 'ticket_status':
        return manager.ticket_status(req['ticket'])
    if op == 'lookup':
        return {'row': manager.lookup(req['generation'], req['model_rev'], req['case_key'],
                                      req['node'], req['rep'], req['sample'], req['body'])}
    if op == 'peek':
        return manager.peek(req['body'], req['partition'], index=req.get('index'),
                            model=req.get('model'), randomness=req.get('randomness'))
    if op == 'read':
        return {'samples': manager.read(req['key_hashes'])}
    if op == 'stats':
        return manager.stats()
    raise ManagerError('bad_request', 'unknown op %r' % (op,))


async def serve(manager, path):
    """Start serving `manager` on the Unix socket `path` (which must not exist)."""
    return await rpc.serve_json(lambda req, connection: _op(manager, req, connection), path,
                                on_close=manager.forget)


class AsyncClient(rpc.AsyncClient):
    """A connection to the manager."""
    error_class = ManagerError

    async def get_or_generate(self, ticket, case_id, node_id, rep, sample, body):
        return await self.call('get_or_generate', ticket=ticket, case=case_id, node=node_id,
                               rep=rep, sample=sample, body=body)

    async def draw(self, ticket, node, body, case, partition=None, rank=None, role=None,
                   model=None, randomness=None, attempt=None):
        extra = {} if model is None else {'model': model}     # absent: today's request bytes
        if randomness is not None:
            extra['randomness'] = randomness
        if attempt is not None:
            extra['attempt'] = attempt
        return await self.call('draw', ticket=ticket, node=node, body=body, case=case,
                               partition=partition, rank=rank, role=role, **extra)
