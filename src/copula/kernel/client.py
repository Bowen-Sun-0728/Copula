"""The client of the control plane: what a search loop or a script uses.

    client = await ControlClient.connect('/path/control.sock')
    rev = (await client.register_dataset(cases, schema))['dataset_rev']
    account = (await client.open_budget(10000))['account']
    ticket = (await client.submit(graph, rev, 'all', {'rounds': 4}, account))['ticket']
    result = await client.wait(ticket)       # {'state', 'record', 'points', 'forecast'}

Every refusal raises `ControlError` with a typed `kind` (budget_exhausted, unknown_cases, ...).
"""
import asyncio
import time
import uuid

from . import rpc
from .errors import EvalrtError


def request_id():
    """A fresh id for one logical request (INTERFACE.md section 2): pass the same id when retrying
    after a lost connection and the control plane returns the first response."""
    return 'r-' + uuid.uuid4().hex


class ControlError(EvalrtError):
    """A refusal or failure reported by the control plane."""


class ControlClient(rpc.AsyncClient):
    error_class = ControlError

    async def ping(self):
        """{model_rev, running, dataset_revs, counts: {jobs, compile_epochs, artifacts,
        plan_jobs}} (SCHEMA section 9.4): the campaign asserts zero compile/plan state with it."""
        return await self.call('ping')

    async def register_dataset(self, cases, schema):
        return await self.call('register_dataset', cases=cases, schema=schema)

    async def get_dataset(self, dataset_rev):
        return await self.call('get_dataset', dataset_rev=dataset_rev)

    async def open_budget(self, limit, prices=None, account=None, request_id=None):
        """Open an account. With `request_id` a retry after a lost connection returns the first
        response instead of opening a second account."""
        extra = {} if request_id is None else {'request_id': request_id}
        return await self.call('open_budget', limit=limit, prices=prices, account=account, **extra)

    async def budget_status(self, account):
        return await self.call('budget_status', account=account)

    async def forecast(self, graph, dataset_rev, case_ids, design=None, account=None, config=None,
                       fresh=False, generation=None, confirmation=None, plan=None, cache=None,
                       side=None):
        return await self.call('forecast', graph=graph, dataset_rev=dataset_rev,
                               case_ids=case_ids, design=design, account=account, config=config,
                               fresh=fresh, generation=generation, confirmation=confirmation,
                               plan=plan, cache=cache, side=side)

    async def submit(self, graph, dataset_rev, case_ids, design=None, account=None, max_cost=None,
                     fresh=False, labels=None, config=None, generation=None, confirmation=None,
                     plan=None, cache=None, side=None, request_id=None, cache_salt=None,
                     run=None):
        """`design` ({rounds, fanout}) or `plan` (a compile artifact id); `cache` is 'shared',
        'fresh', 'domain:<name>' or {default, nodes} of those; `side` from `open_side`. Pass
        `request_id` to make a retry safe (see `request_id()`). `cache_salt` isolates the job's
        requests in the inference server's prefix cache (never part of a key or of the job's
        identity). `run` labels one run of the program: a new label is a new consumer, which reads
        the pool from sample 1 (models INTERFACE J7)."""
        extra = {} if cache_salt is None else {'cache_salt': cache_salt}
        if run is not None:
            extra['run'] = run
        return await self.call('submit', graph=graph, dataset_rev=dataset_rev, case_ids=case_ids,
                               design=design, account=account, max_cost=max_cost, fresh=fresh,
                               labels=labels, config=config, generation=generation,
                               confirmation=confirmation, plan=plan, cache=cache, side=side,
                               request_id=request_id, **extra)

    async def resume(self, ticket, request_id=None):
        """Resume a failed or cancelled job under the same ticket."""
        return await self.call('resume', ticket=ticket, request_id=request_id)

    async def open_side(self, account, request_id=None):
        """A side: tickets submitted with it form one graph's evaluation."""
        return (await self.call('open_side', account=account, request_id=request_id))['side']

    async def open_compile_epoch(self, dataset_rev, case_ids, account, scope, pilot=None,
                                 request_id=None):
        return await self.call('open_compile_epoch', dataset_rev=dataset_rev, case_ids=case_ids,
                               account=account, scope=scope, pilot=pilot, request_id=request_id)

    async def compile(self, graph, epoch, scorer, budget_per_case, account, config=None,
                      max_worse=0.10, min_rounds=3, request_id=None, min_gain=None):
        """Start or find a graph's pilot in an epoch; returns {artifact, ticket, state}.
        `min_gain` (None: the runtime's default, compile.MIN_GAIN): a fanout plan needs at least
        this efficiency over uniform, with bootstrap confidence 1 - max_worse."""
        extra = {} if min_gain is None else {'min_gain': min_gain}
        return await self.call('compile', graph=graph, epoch=epoch, scorer=scorer,
                               budget_per_case=budget_per_case, account=account, config=config,
                               max_worse=max_worse, min_rounds=min_rounds, request_id=request_id,
                               **extra)

    async def artifact(self, artifact):
        return await self.call('artifact', artifact=artifact)

    async def wait_artifact(self, artifact, poll=0.2, timeout=None):
        """Poll `artifact` until it is ready or failed."""
        deadline = None if timeout is None else time.time() + timeout
        while True:
            art = await self.artifact(artifact)
            if art['state'] in ('ready', 'failed'):
                return art
            if deadline is not None and time.time() > deadline:
                raise ControlError('timeout', 'artifact %s still %s' % (artifact, art['state']))
            await asyncio.sleep(poll)

    async def replan(self, artifact, budget_per_case, max_worse=0.10, min_rounds=3,
                     min_gain=None):
        extra = {} if min_gain is None else {'min_gain': min_gain}
        return await self.call('replan', artifact=artifact, budget_per_case=budget_per_case,
                               max_worse=max_worse, min_rounds=min_rounds, **extra)

    async def open_confirmation(self, account, request_id=None):
        """A fresh confirmation group: its arms share samples with each other only."""
        return (await self.call('open_confirmation', account=account,
                                request_id=request_id))['confirmation']

    async def result(self, ticket):
        return await self.call('result', ticket=ticket)

    async def wait(self, ticket, poll=0.2, timeout=None, poll_max=None):
        """Poll `result` until the job is settled; returns the settled result. With `poll_max`
        the interval grows x1.5 per poll from `poll` up to `poll_max` (many waiting clients)."""
        deadline = None if timeout is None else time.time() + timeout
        while True:
            res = await self.result(ticket)
            if res['state'] not in ('queued', 'running'):
                return res
            if deadline is not None and time.time() > deadline:
                raise ControlError('timeout', 'job %s still %s' % (ticket, res['state']))
            await asyncio.sleep(poll)
            if poll_max is not None:
                poll = min(poll_max, poll * 1.5)

    async def trace(self, ticket, cases=None):
        """Every generate node's output of a completed job: {case: {node: [{draw, content}]}}."""
        return (await self.call('trace', ticket=ticket, cases=cases))['trace']

    async def cancel(self, ticket):
        return await self.call('cancel', ticket=ticket)

    async def jobs(self, **filters):
        return await self.call('jobs', **filters)

    async def compare(self, reference, candidates, scorer, level=0.95, family=None):
        """Append candidate tickets or sides to a family (estimates only until `close_family`)."""
        return await self.call('compare', reference=reference, candidates=candidates,
                               scorer=scorer, level=level, family=family)

    async def family(self, family):
        return await self.call('family', family=family)

    async def close_family(self, family):
        """Close a family: verdicts are computed once, over all its rows."""
        return await self.call('close_family', family=family)
