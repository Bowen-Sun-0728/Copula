"""The cache manager (RUNTIME_V0 section 4.4): single flight over the write-once store.

Core invariant: every key maps to at most one sample over all history. The rules that hold it:

- the store lookup, the flight lookup and the flight creation run with no `await` between them, so
  a key can never have two flights;
- a sample is committed to the store before any waiter is woken, so no job ever holds a sample that
  is not durable; with group commit (a batched store, sysval build 2026-10-09) one commit covers
  every write made since the last, and no reply or wake-up leaves before the commit covering what
  it reports (`durable`);
- the manager owns generation (no leases): a flight runs to completion whether or not its requesters
  are still connected, so a crashed job loses nothing and its restart hits what it had started;
- a flight whose retries are exhausted writes nothing; the next request opens a new flight.

Results that jobs compute themselves (registered tools and kinds) are the exception: the job runs
the code, so the manager hands out leases instead (2026-10-13). A requester that asks for one gets
the stored value or the lease to compute it; while another requester holds the lease it waits until
the holder writes the value, releases the lease or its connection closes, then looks again. This
makes single flight cross-process for tools too; requesters that do not ask behave as before.
A lease also ends when it expires (2026-10-13, evening). A requester that promises to renew
(`renew`, the SDK) has its lease expire when it goes `lease_ttl` seconds without renewing it: a
frozen or wedged holder whose connection stays open. Any `max_hold` (seconds) declared by the
holder or by a waiter caps the wait on that execution, whatever the renewals. Clocks start when
the grant's reply leaves, after its commit. Either way one waiter takes over. Expiry cannot give two
requesters different results for one address: the first write wins, and a later writer is told
the stored value (`put` replies with it), which is what its caller then gets. That value is the
right one because a result is a function of its address (a deterministic tool, or a draw seeded
from it).

Dispatch order: flights more tickets are waiting on first, then round-robin across owning tickets,
then arrival. `cancel` drops a ticket's flights that are neither sent nor awaited by another ticket;
an unsent flight another ticket awaits passes to that ticket, which then pays for it; sent flights
finish, are written, and are billed to their owner.

Metering (campaign 2026-10-02, SCHEMA sections 2-4). A ticket may carry a `cache_salt`, frozen with
its spec: every attempt of a flight is sent with its owner's salt (the owner is fixed once sent).
The salt only isolates the server's prefix cache; it never enters a key, a pool, a sample or
`model_rev`. With a ledger, every attempt -- failed ones included -- is one row. A generated result
returned to the ticket that paid for it carries the successful attempt's `meta` (tokens, cached
tokens, replica, timing); waiters, hits and re-reads never do.
"""
import asyncio
import collections
import heapq
import itertools
import random
import re
import time
from collections import Counter

from ..core import key as keys
from ..core.canon import digest
from ..core.request import check_defaults
from ..errors import EvalrtError
from .ledger import meta_fields
from .model import RequestError, TransportError, error_sample, validate_sample

SALT = re.compile(r'^[A-Za-z0-9_.:-]{1,128}$')
MODEL_NAME = re.compile(r'^[a-z][a-z0-9_]{0,31}$')
ACTOR = 'actor'                         # the stack's primary model: what a node without `model` calls
MODEL_TYPES = ('generate', 'reward')
LEASE_TTL = 30.0                        # s a renewing holder's lease lives between renewals


class ManagerError(EvalrtError):
    """A typed failure returned to a requester."""


def _positive(x):
    return type(x) in (int, float) and x == x and 0 < x < float('inf')


def _check_hold(max_hold):
    if max_hold is not None and not _positive(max_hold):
        raise ManagerError('bad_request', 'max_hold must be a number > 0, or absent')


def _consume(future):
    """Mark a flight's exception as retrieved: every requester may already be gone."""
    if not future.cancelled():
        future.exception()


def check_salt(salt):
    if salt is not None and (not isinstance(salt, str) or not SALT.match(salt)):
        raise ManagerError('bad_request', 'cache_salt is None or [A-Za-z0-9_.:-]{1,128}')
    return salt


class _Ticket:
    __slots__ = ('ticket', 'generation', 'model_rev', 'case_keys', 'cancelled', 'salt')

    def __init__(self, ticket, generation, model_rev, case_keys, salt=None):
        self.ticket = ticket
        self.generation = generation
        self.model_rev = model_rev
        self.case_keys = case_keys
        self.cancelled = False
        self.salt = salt

    def spec(self):
        return self.generation, self.model_rev, self.case_keys, self.salt


class _PoolTicket:
    """A key-v3 ticket: draws from pools inside its `partitions`, as program `consumer`, for the
    cases in `case_keys` (case id -> case key; the ledger is per case). `models` maps the model
    names its graph calls to their revisions (models INTERFACE J3); `actor` is always one."""
    __slots__ = ('ticket', 'partitions', 'consumer', 'model_rev', 'case_keys', 'cancelled', 'salt',
                 'models')

    def __init__(self, ticket, partitions, consumer, model_rev, case_keys, salt=None, models=None):
        self.ticket = ticket
        self.partitions = tuple(partitions)
        self.consumer = consumer
        self.model_rev = model_rev
        self.case_keys = case_keys
        self.cancelled = False
        self.salt = salt
        self.models = dict(models or {}, **{ACTOR: model_rev})

    def spec(self):
        return ('pool', self.partitions, self.consumer, self.model_rev,
                sorted(self.case_keys.items()), self.salt, sorted(self.models.items()))


class _Flight:
    __slots__ = ('key_hash', 'key', 'body', 'owner', 'waiters', 'future', 'state', 'seq', 'attempts',
                 'node', 't_created', 't_dispatch', 't_first_send', 'meta', 'prompt_digest', 'route',
                 'model')

    def __init__(self, key_hash, key, body, owner, seq, future, node=None, t_created=None,
                 route=None, model=ACTOR):
        self.model = model              # the backend that generates it (models INTERFACE J3)
        self.key_hash = key_hash
        self.key = key
        self.body = body
        self.owner = owner              # the ticket that pays; changes only while queued
        self.waiters = []               # tickets awaiting this key, the owner included
        self.future = future
        self.state = 'queued'
        self.seq = seq
        self.attempts = 0
        self.node = node if node is not None else key.get('node', '')   # billing label
        self.t_created = t_created
        self.t_dispatch = None
        self.t_first_send = None        # the first attempt's t_send (queueing ends there)
        self.meta = None                # the meta of the attempt whose sample was written
        self.prompt_digest = None
        # The case key that routes the request to a replica (route='case') and labels its ledger
        # rows: the creating draw's case. Key v5 addresses carry no case, so it lives here.
        self.route = route if route is not None else key.get('case')


class _Lease:
    """One requester's right to compute one registered result (an address of `computed`)."""
    __slots__ = ('token', 'address', 'connection', 'future', 't', 'max_hold', 'renews', 'base',
                 'hold_until', 'wait_bounds', 'deadline', 'started', 'nudge')

    def __init__(self, token, address, connection, future, nudge, t, max_hold, renews, now):
        self.token = token
        self.address = address
        self.connection = connection    # closing it releases the lease (the holder died)
        self.future = future            # done when the lease ends: written, released, dropped or
        #                                 expired
        self.t = t                      # when it was granted (`stats` reports the oldest)
        self.max_hold = max_hold        # the holder's own bound, s (None: none)
        self.renews = renews            # the holder renews it, so `lease_ttl` applies
        self.started = False            # its grant's reply has left (the holder's clocks run)
        self.base = now                 # loop time of the start or the last renewal
        self.hold_until = None          # start + the holder's max_hold
        self.wait_bounds = {}           # waiting request -> its arrival here + its max_hold
        self.deadline = None            # loop time it expires (None: never); see Manager._refresh
        self.nudge = nudge              # set when the deadline comes forward: waiters re-time


class _Queue:
    """The queued flights, in dispatch order: more distinct waiting tickets first, then the owner
    served least recently, then arrival -- the order `_pick` has always used, kept in heaps so that
    each event costs O(log n) instead of a scan of the queue (sysval build 2026-10-09).

    Per (backend, owner) a heap of that owner's flights by (-waiters, seq); per backend a heap of
    owners by (-waiters, last served, seq) of their best flight. Entries are never updated in place:
    every change to a flight's waiters or owner pushes its current key (`touched`), so an entry
    whose key is no longer the flight's is simply dropped when it reaches the top. An owner has one
    live entry (`live`); others are dropped, and the live one is replaced when the owner's key has
    changed since (its last-served number changes at every dispatch, without notice)."""

    def __init__(self, served):
        self.served = served            # owner -> number of its last dispatch, or -1
        self.items = {}                 # key_hash -> flight, in arrival order
        self.mine = {}                  # (model, owner) -> heap of (-w, seq, n, flight)
        self.owners = {}                # model -> heap of (-w, served, seq, n, owner)
        self.live = {}                  # (model, owner) -> n of its live owner entry
        self.n = itertools.count()

    def __len__(self):
        return len(self.items)

    def __bool__(self):
        return bool(self.items)

    def __iter__(self):
        return iter(list(self.items.values()))

    @staticmethod
    def _w(flight):
        return len(set(flight.waiters))

    def _queued(self, flight):
        return self.items.get(flight.key_hash) is flight

    def add(self, flight):
        self.items[flight.key_hash] = flight
        self.touched(flight)

    def remove(self, flight):
        if self._queued(flight):
            del self.items[flight.key_hash]

    def clear(self):
        self.items.clear()
        self.mine.clear()
        self.owners.clear()
        self.live.clear()

    def touched(self, flight):
        """`flight`'s waiters or owner changed (or it is new): enter its current key."""
        if not self._queued(flight):
            return
        heapq.heappush(self.mine.setdefault((flight.model, flight.owner), []),
                       (-self._w(flight), flight.seq, next(self.n), flight))
        self._push_owner(flight.model, flight.owner)

    def _best(self, model, owner):
        """(-w, seq, flight) of the owner's first queued flight on `model`, or None. Every change
        pushed a current entry, so an entry that no longer matches its flight is dropped."""
        heap = self.mine.get((model, owner))
        while heap:
            nw, seq, _, flight = heap[0]
            if (not self._queued(flight) or flight.owner != owner or flight.model != model
                    or -self._w(flight) != nw):
                heapq.heappop(heap)
                continue
            return nw, seq, flight
        if heap is not None:
            del self.mine[(model, owner)]
        return None

    def _push_owner(self, model, owner):
        """Enter the owner's current key as its one live entry (unless it already is)."""
        best = self._best(model, owner)
        if best is None:
            self.live.pop((model, owner), None)
            return
        key = (best[0], self.served(owner), best[1])
        live = self.live.get((model, owner))
        if live is not None and live[0] == key:
            return
        n = next(self.n)
        self.live[(model, owner)] = (key, n)
        heapq.heappush(self.owners.setdefault(model, []), key + (n, owner))

    def _top(self, model):
        """(key, owner, flight) of the first flight on `model`, or None."""
        heap = self.owners.get(model)
        while heap:
            nw, served, seq, n, owner = heap[0]
            live = self.live.get((model, owner))
            if live is None or live[1] != n:            # superseded by a later entry
                heapq.heappop(heap)
                continue
            best = self._best(model, owner)
            if best is None:
                heapq.heappop(heap)
                del self.live[(model, owner)]
                continue
            key = (best[0], self.served(owner), best[1])
            if key != (nw, served, seq):                # changed since: enter its current key
                heapq.heappop(heap)
                del self.live[(model, owner)]
                self._push_owner(model, owner)
                continue
            return key, owner, best[2]
        return None

    def pick(self, room):
        """Remove and return the first flight whose backend has `room(model)`, or None."""
        tops = [t for t in (self._top(m) for m in list(self.owners) if room(m)) if t is not None]
        if not tops:
            return None
        key, owner, flight = min(tops, key=lambda t: t[0])
        del self.items[flight.key_hash]
        self._push_owner(flight.model, owner)           # the owner's next flight, if any
        return flight


def check_backends(model, models):
    """{name: client} of a stack: `model` is the actor, `models` the other named backends. A client's
    `kind` is 'generate' (default) or 'reward'; a reward backend's pinned table has temperature 0,
    so every scoring request is deterministic (models INTERFACE J4)."""
    out = {ACTOR: model}
    for name, client in sorted((models or {}).items()):
        if not isinstance(name, str) or not MODEL_NAME.match(name) or name == ACTOR:
            raise ValueError('model name %r must match %s and not be %r'
                             % (name, MODEL_NAME.pattern, ACTOR))
        out[name] = client
    for name, client in out.items():
        if not getattr(client, 'model_rev', None):
            raise ValueError('the model client %r has no model_rev' % name)
        kind = getattr(client, 'kind', 'generate')
        if kind not in MODEL_TYPES:
            raise ValueError('model %r: kind must be one of %s' % (name, MODEL_TYPES))
        defaults = check_defaults(client.defaults)
        if kind == 'reward' and defaults['temperature'] != 0.0:
            raise ValueError('reward model %r: its pinned temperature must be 0' % name)
    if getattr(model, 'kind', 'generate') != 'generate':
        raise ValueError('the actor must be a generate model')
    return out


class Manager:
    def __init__(self, store, model, inflight_cap=64, max_attempts=6, backoff_base=0.5,
                 backoff_cap=30.0, clock=time.time, rng=None, ledger=None, models=None,
                 lease_ttl=LEASE_TTL):
        """`model` is the actor; `models` adds named backends (models INTERFACE J3). The in-flight
        cap applies to each backend."""
        if type(inflight_cap) is not int or inflight_cap < 1:
            raise ValueError('inflight_cap must be an int >= 1')
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError('max_attempts must be an int >= 1')
        if lease_ttl is not None and lease_ttl != 0 and not _positive(lease_ttl):
            raise ValueError('lease_ttl must be a number > 0, or 0/None for no expiry')
        self.lease_ttl = lease_ttl or None      # a renewing holder's lease lives this long between
        #                                         renewals (None: no expiry by TTL)
        self.store = store
        self.models = check_backends(model, models)
        self.model = model
        self.defaults = check_defaults(model.defaults)
        self._defaults = {name: check_defaults(c.defaults) for name, c in self.models.items()}
        self._busy = Counter()          # model name -> flights in flight on that backend
        self.cap = inflight_cap
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.clock = clock
        self.rng = rng or random.Random()
        self.tickets = {}
        self.flights = {}               # key_hash -> _Flight, queued or sent
        self._served = {}               # owning ticket -> dispatch number of its last sent flight
        self.queued = _Queue(lambda owner: self._served.get(owner, -1))     # flights not yet sent
        self.inflight = 0
        self.counters = Counter()
        self._seq = itertools.count()
        self._dispatches = itertools.count()
        self._wake = None
        self._dispatcher = None
        self._tasks = set()
        self._asked = {}                # scope -> key hashes that scope has requested
        self._claimed = {}              # (partition, consumer, case, pool) -> claimed numbers
        self._pool_served = {}          # scope -> {(partition, case, pool): numbers served}
        self._leases = {}               # computed address -> _Lease: a requester is computing it
        self._lease_tokens = {}         # lease token -> _Lease
        self._lease_seq = itertools.count(1)
        self.ledger = ledger           # GenLedger or None: one row per attempt
        self.cached_missing = 0         # replies with usage but no prompt_tokens_details (D36)
        self.config = None              # what `ping` reports (set by the CLI)
        self._commit_waiters = []       # group commit: who waits for the next commit
        self._commit_scheduled = False
        self._ended = collections.OrderedDict()     # recently ended scopes (refused afterwards)

    # -- durability (group commit) -----------------------------------------------------------------

    async def durable(self):
        """Return once every store write made so far is committed. With a batched store one commit
        (one fsync) covers every write made since the last; otherwise every write already is."""
        if not getattr(self.store, 'dirty', False):
            return
        future = asyncio.get_running_loop().create_future()
        self._commit_waiters.append(future)
        if not self._commit_scheduled:
            self._commit_scheduled = True
            asyncio.get_running_loop().call_soon(self._commit)
        await asyncio.shield(future)

    def _commit(self):
        self._commit_scheduled = False
        waiters, self._commit_waiters = self._commit_waiters, []
        try:
            self.store.commit()
        except Exception as err:            # noqa: BLE001 -- every waiter learns it failed
            self._claimed.clear()           # claims of a lost batch: reload from the store
            for f in waiters:
                if not f.done():
                    f.set_exception(ManagerError('store_failed', str(err)))
                    f.add_done_callback(_consume)
            return
        for f in waiters:
            if not f.done():
                f.set_result(None)

    # -- lifecycle ------------------------------------------------------------------------------

    async def start(self):
        self._wake = asyncio.Event()
        self._dispatcher = asyncio.ensure_future(self._dispatch_loop())
        if self.ledger is not None:
            self.ledger.start()

    async def drain(self):
        """Wait until no flight is left: a clean shutdown, since sent calls are already paid for."""
        while self.flights:
            pending = [asyncio.shield(f.future) for f in list(self.flights.values())]
            await asyncio.gather(*pending, return_exceptions=True)
        await self.durable()
        if self.ledger is not None:
            self.ledger.flush()

    async def stop(self):
        """Stop at once, as a crash would: unfinished flights are abandoned and write nothing."""
        tasks = [t for t in [self._dispatcher] + list(self._tasks) if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for flight in list(self.flights.values()):
            if not flight.future.done():
                flight.future.set_exception(ManagerError('manager_stopped'))
        self.flights.clear()
        self.queued.clear()
        if self.ledger is not None:
            await self.ledger.stop()

    # -- tickets --------------------------------------------------------------------------------

    def register_ticket(self, ticket, generation, model_rev, case_keys, cache_salt=None):
        """Freeze what the manager derives keys from. Idempotent for an identical spec (the cache
        salt included: a ticket has one salt for life)."""
        if not isinstance(ticket, str) or not ticket:
            raise ManagerError('bad_request', 'ticket must be a non-empty string')
        keys.check_generation(generation)
        if model_rev != self.model.model_rev:
            raise ManagerError('model_mismatch', 'ticket wants %s, the server runs %s'
                               % (model_rev, self.model.model_rev))
        if not isinstance(case_keys, dict) or not case_keys or not all(
                isinstance(c, str) and c and isinstance(k, str) and k for c, k in case_keys.items()):
            raise ManagerError('bad_request', 'case_keys must map case ids to case keys')
        new = _Ticket(ticket, generation, model_rev, dict(case_keys), check_salt(cache_salt))
        old = self.tickets.get(ticket)
        if old is None:
            self.tickets[ticket] = new
        elif old.spec() != new.spec():
            raise ManagerError('ticket_conflict', ticket)
        elif old.cancelled:
            raise ManagerError('cancelled', ticket)

    def model_table(self):
        """{name: {model_rev, type, defaults}} of every backend (what `ping` reports): the pinned
        table lets a job normalise its requests as the manager does (role order), and a reward
        backend also says how it takes its input."""
        out = {}
        for name, c in self.models.items():
            row = {'model_rev': c.model_rev, 'type': getattr(c, 'kind', 'generate'),
                   'defaults': self._defaults[name]}
            if row['type'] == 'reward':
                row['reward_input'] = getattr(c, 'reward_input', 'raw')
            out[name] = row
        return out

    def register_pool_ticket(self, ticket, partitions, consumer, model_rev, case_keys,
                             cache_salt=None, models=None):
        """A key-v3 ticket. `partitions` are the generation labels its draws may use: the shared
        partition of its space, its private `fresh/<ticket>` (INTERFACE.md D8) and the named domains
        its cache policy names (models INTERFACE J6). `models` maps the other model names its graph
        calls to their revisions (J3). Idempotent for an identical spec."""
        if not isinstance(ticket, str) or not ticket:
            raise ManagerError('bad_request', 'ticket must be a non-empty string')
        if isinstance(partitions, str):
            partitions = [partitions]
        if not isinstance(partitions, (list, tuple)) or not partitions:
            raise ManagerError('bad_request', 'partitions must be a non-empty list')
        for part in partitions:
            keys.check_generation(part)
        if not isinstance(consumer, str) or not consumer:
            raise ManagerError('bad_request', 'consumer must be a non-empty string')
        if model_rev != self.model.model_rev:
            raise ManagerError('model_mismatch', 'ticket wants %s, the server runs %s'
                               % (model_rev, self.model.model_rev))
        if models is not None and not isinstance(models, dict):
            raise ManagerError('bad_request', 'models maps model names to revisions')
        for name, rev in sorted((models or {}).items()):
            client = self.models.get(name)
            if client is None:
                raise ManagerError('unknown_model', 'ticket names model %r, the stack has %s'
                                   % (name, sorted(self.models)))
            if rev != client.model_rev:
                raise ManagerError('model_mismatch', 'ticket wants %s for %r, the server runs %s'
                                   % (rev, name, client.model_rev))
        if not isinstance(case_keys, dict) or not case_keys or not all(
                isinstance(c, str) and c and isinstance(k, str) and k for c, k in case_keys.items()):
            raise ManagerError('bad_request', 'case_keys must map case ids to case keys')
        new = _PoolTicket(ticket, partitions, consumer, model_rev, dict(case_keys),
                          check_salt(cache_salt), models)
        old = self.tickets.get(ticket)
        if old is None:
            self.tickets[ticket] = new
        elif old.spec() != new.spec():
            raise ManagerError('ticket_conflict', ticket)
        elif old.cancelled:
            raise ManagerError('cancelled', ticket)

    def reopen(self, ticket):
        """Clear a ticket's cancel so that the same ticket can be resumed (INTERFACE.md D7). A ticket
        the manager does not know (a restart) needs nothing: it is registered afresh."""
        t = self.tickets.get(ticket)
        if t is not None:
            t.cancelled = False
        return {'known': t is not None}

    def _live(self, ticket):
        t = self.tickets.get(ticket)
        if t is None:
            raise ManagerError('unknown_ticket', str(ticket))
        if t.cancelled:
            raise ManagerError('cancelled', ticket)
        return t

    def cancel(self, ticket):
        t = self.tickets.get(ticket)
        if t is None:
            raise ManagerError('unknown_ticket', str(ticket))
        t.cancelled = True
        dropped = transferred = sent = 0
        for flight in list(self.flights.values()):
            others = [w for w in flight.waiters if w != ticket]
            if flight.owner != ticket:
                if len(others) != len(flight.waiters):
                    flight.waiters = others
                    self.queued.touched(flight)
            elif flight.state == 'sent':
                flight.waiters = others        # it finishes, is written and billed to this ticket
                sent += 1
            elif others:
                flight.owner, flight.waiters = others[0], others
                self.queued.touched(flight)
                transferred += 1
            else:
                self.queued.remove(flight)
                del self.flights[flight.key_hash]
                flight.future.set_exception(ManagerError('cancelled', ticket))
                dropped += 1
        return {'dropped': dropped, 'transferred': transferred, 'sent': sent}

    def end_scope(self, scope):
        """Drop one finished attempt's memory (the packaged system ends every run attempt). A draw
        that arrives on an ended scope afterwards (a call that raced its run's end) is refused, so
        it can never re-read the attempt's first claim."""
        self._asked.pop(scope, None)
        self._pool_served.pop(scope, None)
        self._ended[scope] = None
        while len(self._ended) > 65536:
            self._ended.popitem(last=False)
        return {}

    def _check_scope(self, scope):
        if scope in self._ended:
            raise ManagerError('run_ended', 'this attempt has ended')

    def forget(self, connection):
        """Drop the duplicate-key memory of every scope opened on a closed connection, and release
        the leases it held: their waiters look again, and one of them computes the value."""
        for scope in [s for s in self._asked if s[0] == connection]:
            del self._asked[scope]
        for scope in [s for s in self._pool_served if s[0] == connection]:
            del self._pool_served[scope]
        for lease in [l for l in self._leases.values() if l.connection == connection]:
            self.counters['leases_dropped'] += 1
            self._end_lease(lease)

    def ticket_status(self, ticket):
        """What settlement needs: flights the ticket still owns, and the entries it generated."""
        t = self.tickets.get(ticket)             # unknown after a manager restart: the store knows
        owned = sum(1 for f in self.flights.values() if f.owner == ticket)
        return {'cancelled': bool(t and t.cancelled), 'owned_flights': owned,
                'generated': self.store.generated_counts(ticket)}

    # -- keys -----------------------------------------------------------------------------------

    @staticmethod
    def _result(row, source):
        return {'key_hash': row['key_hash'], 'sample': row['sample'], 'source': source,
                'generated_by': row['generated_by'], 'created_at': row['created_at'],
                'attempts': row['attempts']}

    def lookup(self, generation, model_rev, case_key, node_id, rep, sample, body):
        """Read-only probe for the forecast walk: the stored row, or None. Never opens a flight."""
        _, key_hash, _ = keys.make_key(case_key, model_rev, node_id, body, rep, sample,
                                       generation, self.defaults)
        row = self.store.get(key_hash)
        return None if row is None else self._result(row, 'existing')

    async def get_or_generate(self, ticket, case_id, node_id, rep, sample, body, scope=None):
        """The one sample under this key: stored, awaited, or generated now.

        `scope` identifies one attempt of one job (the server passes connection and ticket). A
        scope asking again for a key it was already served means two of its draws would share a
        sample, which breaks independence within the job, so it is refused rather than served.
        A request that failed delivered nothing and may be repeated.
        """
        t = self._live(ticket)
        case_key = t.case_keys.get(case_id)
        if case_key is None:
            raise ManagerError('unknown_case', '%s is not in ticket %s' % (case_id, ticket))
        key, key_hash, normal = keys.make_key(case_key, t.model_rev, node_id, body, rep, sample,
                                              t.generation, self.defaults)
        if scope is not None:
            asked = self._asked.setdefault(scope, set())
            if key_hash in asked:
                raise ManagerError('duplicate_key', '%s asked twice for %s/%s rep %d'
                                   % (ticket, case_id, node_id, rep))
            asked.add(key_hash)
        # Atomic section: no await between the store lookup and joining or creating the flight.
        row = self.store.get(key_hash)
        if row is not None:
            self.counters['hits_existing'] += 1
            return self._result(row, 'existing')
        flight = self.flights.get(key_hash)
        if flight is None:
            future = asyncio.get_running_loop().create_future()
            future.add_done_callback(_consume)
            flight = _Flight(key_hash, key, normal, ticket, next(self._seq), future,
                             t_created=self.clock())
            self.flights[key_hash] = flight
            self.queued.add(flight)
            self._wake.set()
        flight.waiters.append(ticket)
        self.queued.touched(flight)
        # End of the atomic section. Shielded: a requester that goes away never cancels the flight.
        try:
            row = await asyncio.shield(flight.future)
        except ManagerError:
            if scope is not None:                   # nothing was delivered: asking again is fine
                self._asked.get(scope, set()).discard(key_hash)
            raise
        return self._metered(self._result(row, 'generated' if row['generated_by'] == ticket
                                          else 'waited'), flight)

    async def draw(self, ticket, node, body, case=None, partition=None, rank=None, role=None,
                   scope=None, model=None, randomness=None):
        """One sample of the pool named by (model, request), for one case, for a key-v3 ticket.

        The ledger rules, which keep every job's law equal to naive sampling:
        1. a job never reads one sample twice: each draw claims a number not yet claimed by its
           program in its partition for this case (so identical calls in one job -- other nodes,
           other rounds -- get distinct samples, drawn independently);
        2. a program evaluated again reads numbers it has never read (new evidence, not a replay);
        3. different programs claim from 1 upwards, so they share whatever the pool already holds.
           The ledger is per case, but the sample address is not (N14, which superseded D1): two
           cases read one sample exactly when they send the identical request.

        A greedy request (temperature 0 or top_k 1) in a shared partition reads the pool's one
        greedy entry instead: no number, no claim, shared across partitions and cases (N16).

        `role` (1 + the job's earlier draws of this case with the identical request, in its fixed
        draw order) is stored with the claim: a restarted or resumed attempt of the ticket (a new
        scope) re-reads the number its draw of the same role claimed, so a restart hands every draw
        its own sample and generates only what is missing. Draws without a role re-read the
        ticket's claims in claim order.
        `partition` picks one of the ticket's partitions (required when it has several).
        `rank` asks for sample number `rank` when this program has not claimed it. Every v4 draw
        passes its role as its rank (n11 INTERFACE I1), so programs read the same number for the same
        canonical repeat, whatever order their draws arrive in; otherwise the rules above apply.
        `node` labels billing. `model` names the backend (models INTERFACE J3; default `actor`): the
        pool is named by that backend's revision and pinned table. In a named domain `dom/<name>` a
        greedy request is keyed inside the domain (J6). A reward backend takes greedy requests only
        (J4).
        `randomness` (the packaged system, sdk INTERFACE U4) is the node's declared field and
        overrides the reading of the request: 'deterministic' reads the one global entry of the
        pool and refuses a request that is not greedy; 'stochastic' reads numbered samples even for
        a greedy request. None keeps the behaviour above.
        """
        t = self._live(ticket)
        if not isinstance(t, _PoolTicket):
            raise ManagerError('bad_request', '%s is not a pool ticket' % ticket)
        if scope is None:                   # without a scope, draws could not be told apart
            raise ManagerError('bad_request', 'a draw needs a scope')
        self._check_scope(scope)
        case_key = t.case_keys.get(case)
        if case_key is None:
            raise ManagerError('unknown_case', '%s is not in ticket %s' % (case, ticket))
        if partition is None:
            if len(t.partitions) != 1:
                raise ManagerError('bad_request', 'ticket %s has several partitions: name one'
                                   % ticket)
            partition = t.partitions[0]
        elif partition not in t.partitions:
            raise ManagerError('bad_request', 'partition %r is not open to ticket %s'
                               % (partition, ticket))
        for name, value in (('rank', rank), ('role', role)):
            if value is not None and (type(value) is not int or value < 1):
                raise ManagerError('bad_request', '%s must be an int >= 1' % name)
        if randomness not in (None, 'deterministic', 'stochastic'):
            raise ManagerError('bad_request', 'randomness is deterministic or stochastic')
        name = ACTOR if model is None else model
        if name not in t.models:
            raise ManagerError('unknown_model', 'ticket %s did not register model %r'
                               % (ticket, name))
        pool, normal = keys.pool_key(t.models[name], body, self._defaults[name])
        greedy = keys.is_greedy(normal)
        if getattr(self.models[name], 'kind', 'generate') == 'reward':
            if not greedy:
                raise ManagerError('bad_request', 'reward model %r takes deterministic requests '
                                   'only (temperature 0)' % name)
            # A score depends on the messages only: one entry per scoring prompt, whatever the
            # calling node's max_tokens or (deterministic) sampling fields.
            pool, normal = keys.pool_key(t.models[name], {'messages': normal['messages'],
                                                          'max_tokens': 1},
                                         self._defaults[name])
        label = str(node) if name == ACTOR else '%s@%s' % (node, name)     # billing (J9)
        extra = {} if name == ACTOR else {'model': name}
        if randomness == 'deterministic':
            if not greedy:
                raise ManagerError('bad_request', 'a deterministic node sends greedy requests only '
                                   '(temperature 0 or top_k 1)')
            # Declared deterministic + cache: one global entry, whatever the partition (U4).
            address = keys.greedy_address(pool)
            return await self._join(ticket, keys.key_hash(address), address, normal, label,
                                    case_key, dict(pool=pool, index=None, partition=partition,
                                                   greedy=True, **extra), model=name)
        if greedy and randomness is None and not partition.startswith('fresh/'):
            # Deterministic: one entry per pool, shared by every partition and case (N16), or,
            # in a named domain, by the jobs of that domain only (J6); no number is claimed, and
            # repeats of the request read the same entry.
            domain = partition if partition.startswith(keys.DOMAIN_PREFIX) else None
            address = keys.greedy_address(pool, domain)
            return await self._join(ticket, keys.key_hash(address), address, normal, label,
                                    case_key, dict(pool=pool, index=None, partition=partition,
                                                   greedy=True, **extra), model=name)
        # Atomic section: no await from choosing the number to joining or creating the flight.
        slot = (partition, case_key, pool)
        served = self._pool_served.setdefault(scope, {}).setdefault(slot, [])
        index = self._claim(ticket, t.consumer, partition, case_key, pool, served, rank, role)
        address = keys.pool_address(partition, pool, index)
        try:
            return await self._join(ticket, keys.key_hash(address), address, normal, label,
                                    case_key, dict(pool=pool, index=index, partition=partition,
                                                   **extra), model=name)
        except ManagerError:                        # nothing delivered: this number is re-read
            got = self._pool_served.get(scope, {}).get(slot)
            if got is not None and index in got:
                got.remove(index)
            raise

    def peek(self, body, partition, index=None, model=None, randomness=None):
        """Read-only (claims C1): the stored sample at the address a draw of `body` in `partition`
        would read -- number `index` of a stochastic pool, or the one global entry of a greedy
        (deterministic) pool, as `draw` names them -- without a ticket, a claim, a flight or a
        write. -> {key_hash, pool, index, greedy, exists, inflight, sample (None if not stored)}.
        A stochastic peek without `index` returns the pool alone (exists None), for a caller that
        numbers its draws per pool itself. A forecast reads the store through this alone."""
        name = ACTOR if model is None else model
        client = self.models.get(name)
        if client is None:
            raise ManagerError('unknown_model', 'the stack has no model %r' % (name,))
        keys.check_generation(partition)
        if randomness not in (None, 'deterministic', 'stochastic'):
            raise ManagerError('bad_request', 'randomness is deterministic or stochastic')
        pool, normal = keys.pool_key(client.model_rev, body, self._defaults[name])
        greedy = keys.is_greedy(normal)
        if getattr(client, 'kind', 'generate') == 'reward':
            if not greedy:
                raise ManagerError('bad_request', 'reward model %r takes deterministic requests '
                                   'only (temperature 0)' % name)
            pool, normal = keys.pool_key(client.model_rev, {'messages': normal['messages'],
                                                            'max_tokens': 1},
                                         self._defaults[name])
        if randomness == 'deterministic':
            if not greedy:
                raise ManagerError('bad_request', 'a deterministic node sends greedy requests only '
                                   '(temperature 0 or top_k 1)')
            address, idx, is_greedy = keys.greedy_address(pool), None, True
        elif greedy and randomness is None and not partition.startswith('fresh/'):
            domain = partition if partition.startswith(keys.DOMAIN_PREFIX) else None
            address, idx, is_greedy = keys.greedy_address(pool, domain), None, True
        else:
            if index is None:               # the pool only: a caller numbering its own draws
                return {'key_hash': None, 'pool': pool, 'index': None, 'greedy': False,
                        'exists': None, 'inflight': None, 'sample': None}
            if type(index) is not int or index < 1:
                raise ManagerError('bad_request', 'a stochastic peek names its number (>= 1)')
            address, idx, is_greedy = keys.pool_address(partition, pool, index), index, False
        key_hash = keys.key_hash(address)
        row = self.store.get(key_hash)
        return {'key_hash': key_hash, 'pool': pool, 'index': idx, 'greedy': is_greedy,
                'exists': row is not None, 'inflight': key_hash in self.flights,
                'sample': None if row is None else row['sample']}

    def _claim(self, ticket, consumer, partition, case_key, pool, served, rank, role):
        """The number a draw reads (the ledger rules of `draw`); claims it when new. Synchronous:
        callers hold the atomic section."""
        mine = self.store.ticket_claims(ticket, partition, case_key, pool)
        ledger = (partition, consumer, case_key, pool)
        claimed = self._claimed.get(ledger)
        if claimed is None:
            claimed = self._claimed[ledger] = self.store.claimed(*ledger)
        if role is not None:                                    # re-read by role
            index = next((i for i, r in mine if r == role and i not in served), None)
        else:                                                   # re-read in claim order
            index = next((i for i, r in mine if r is None and i not in served), None)
        if index is None:
            if rank is not None and rank not in claimed:
                index = rank
            else:
                index = 1
                while index in claimed:
                    index += 1
            self.store.claim(partition, consumer, case_key, pool, index, ticket, self.clock(),
                             role=role)
            claimed.add(index)
        served.append(index)
        return index

    async def _join(self, ticket, key_hash, address, normal, node, route, extra, model=ACTOR):
        """The stored sample at `address`, or the flight that generates it (joined or created with
        no await since the caller's atomic section began) on backend `model`."""
        row = self.store.get(key_hash)
        if row is not None:
            self.counters['hits_existing'] += 1
            return dict(self._result(row, 'existing'), **extra)
        flight = self.flights.get(key_hash)
        if flight is None:
            future = asyncio.get_running_loop().create_future()
            future.add_done_callback(_consume)
            flight = _Flight(key_hash, address, normal, ticket, next(self._seq), future,
                             node=str(node), t_created=self.clock(), route=route, model=model)
            self.flights[key_hash] = flight
            self.queued.add(flight)
            self._wake.set()
        flight.waiters.append(ticket)
        self.queued.touched(flight)
        # End of the atomic section.
        row = await asyncio.shield(flight.future)
        return self._metered(dict(self._result(row, 'generated' if row['generated_by'] == ticket
                                               else 'waited'), **extra), flight)

    # -- results of registered kinds (memo='store') ---------------------------------------------

    def _memo_address(self, ticket, case, identity, partition=None):
        t = self._live(ticket)
        if t.case_keys.get(case) is None:
            raise ManagerError('unknown_case', '%s is not in ticket %s' % (case, ticket))
        if partition is not None and partition not in getattr(t, 'partitions', ()):
            raise ManagerError('bad_request', 'partition %r is not open to ticket %s'
                               % (partition, ticket))
        try:
            return keys.computed_address(identity, partition)
        except keys.KeySpecError as err:
            raise ManagerError('bad_request', str(err))

    def memo_get(self, ticket, case, identity, partition=None):
        """`partition`: a named domain the result is kept in (models INTERFACE J6), or None."""
        value, writer = self.store.get_computed(self._memo_address(ticket, case, identity,
                                                                   partition))
        return {'value': value, 'ticket': writer}

    async def memo_lease(self, ticket, case, identity, connection, partition=None,
                         max_hold=None, renew=False):
        """`memo_get` with single flight across requesters: the stored value, or a lease to
        compute it (waiting while another requester holds one)."""
        return await self._acquire(self._memo_address(ticket, case, identity, partition),
                                   connection, max_hold, renew)

    def memo_put(self, ticket, case, identity, value, kind, kind_rev, partition=None):
        if not isinstance(value, str):
            raise ManagerError('bad_request', 'a computed value is text')
        address = self._memo_address(ticket, case, identity, partition)
        return self._put(address, kind, kind_rev, value, ticket)

    # -- draws of stochastic registered kinds (cache yes; N15, I6) -------------------------------

    def _draw_ticket(self, ticket, case, partition):
        t = self._live(ticket)
        if not isinstance(t, _PoolTicket):
            raise ManagerError('bad_request', '%s is not a pool ticket' % ticket)
        case_key = t.case_keys.get(case)
        if case_key is None:
            raise ManagerError('unknown_case', '%s is not in ticket %s' % (case, ticket))
        if partition not in t.partitions:
            raise ManagerError('bad_request', 'partition %r is not open to ticket %s'
                               % (partition, ticket))
        return t, case_key

    def kind_draw(self, ticket, case, partition, identity, role=None, scope=None):
        """Draw `role` of a stochastic registered kind with this identity, for one case: the number
        is claimed by the same ledger rules as a model draw (pool `c:<identity>`); returns the
        number, the stored value (None when nobody computed it yet), its writer and the seed the
        job computes it with. The value is a function of the address, so two jobs computing one
        address concurrently write the same value."""
        index, address = self._kind_claim(ticket, case, partition, identity, role, scope)
        value, writer = self.store.get_computed(address)
        return {'index': index, 'value': value, 'ticket': writer,
                'seed': keys.draw_seed(self.store.secret, address)}

    async def kind_draw_lease(self, ticket, case, partition, identity, connection, role=None,
                              scope=None, max_hold=None, renew=False):
        """`kind_draw` with single flight across requesters: the number is claimed as there, then
        the stored value or a lease to compute it (waiting while another requester holds one)."""
        _check_hold(max_hold)
        index, address = self._kind_claim(ticket, case, partition, identity, role, scope)
        got = await self._acquire(address, connection, max_hold, renew)
        return dict(got, index=index, seed=keys.draw_seed(self.store.secret, address))

    def _kind_claim(self, ticket, case, partition, identity, role, scope):
        if scope is None:
            raise ManagerError('bad_request', 'a draw needs a scope')
        self._check_scope(scope)
        if role is not None and (type(role) is not int or role < 1):
            raise ManagerError('bad_request', 'role must be an int >= 1')
        t, case_key = self._draw_ticket(ticket, case, partition)
        try:
            pool = keys.computed_pool(identity)
        except keys.KeySpecError as err:
            raise ManagerError('bad_request', str(err))
        served = self._pool_served.setdefault(scope, {}).setdefault((partition, case_key, pool),
                                                                    [])
        index = self._claim(ticket, t.consumer, partition, case_key, pool, served, role, role)
        return index, keys.computed_draw_address(partition, identity, index)

    def kind_put(self, ticket, case, partition, identity, index, value, kind, kind_rev):
        if not isinstance(value, str):
            raise ManagerError('bad_request', 'a computed value is text')
        self._draw_ticket(ticket, case, partition)
        try:
            address = keys.computed_draw_address(partition, identity, index)
        except keys.KeySpecError as err:
            raise ManagerError('bad_request', str(err))
        return self._put(address, kind, kind_rev, value, ticket)

    def _put(self, address, kind, kind_rev, value, ticket):
        """Write a computed value; the first write wins. A writer that lost (another requester
        wrote first: an expired lease's late holder, or its taker) is told the stored value, so
        every caller ends up with the one value stored at the address."""
        first = self.store.put_computed(address, str(kind), str(kind_rev), value, ticket,
                                        self.clock())
        self._written(address)
        if first:
            return {}
        self.counters['computed_lost_races'] += 1
        stored, writer = self.store.get_computed(address)
        return {'value': stored, 'ticket': writer}

    # -- leases: single flight for results that jobs compute (2026-10-13) ------------------------

    async def _acquire(self, address, connection, max_hold=None, renew=False):
        """The stored value at `address`, or a new lease on it. The store lookup, the expiry
        check, the lease lookup and the lease creation run with no `await` between them, so an
        address never has two live leases. A waiter sleeps until the lease ends or its deadline
        passes, then looks again; a renewal moves the deadline, and a waiter's own `max_hold`
        brings it forward. A waiter's reply, like every reply, leaves only after the commit
        covering the value it reports (`durable`). A grant tells a renewing holder `ttl`, the
        interval to renew within (None: no expiry by TTL)."""
        _check_hold(max_hold)
        loop = asyncio.get_running_loop()
        me = object()                   # this request, as a key of the leases it waits on
        bounded = None                  # the lease carrying this request's own bound
        waited = False
        try:
            while True:
                value, writer = self.store.get_computed(address)
                if value is not None:
                    return {'value': value, 'ticket': writer, 'lease': None, 'waited': waited}
                lease = self._leases.get(address)
                if lease is not None and self._expired(lease, loop.time()):
                    lease = None
                if lease is None:
                    lease = _Lease('L%d' % next(self._lease_seq), address, connection,
                                   loop.create_future(), loop.create_future(), self.clock(),
                                   max_hold, bool(renew), loop.time())
                    self._leases[address] = lease
                    self._lease_tokens[lease.token] = lease
                    self.counters['leases'] += 1
                    if waited:
                        self.counters['lease_handovers'] += 1     # a waiter takes over
                    return {'value': None, 'ticket': None, 'lease': lease.token,
                            'waited': waited, 'ttl': self.lease_ttl if lease.renews else None}
                if not waited:
                    self.counters['lease_waiters'] += 1           # once per waiting request
                waited = True
                if max_hold is not None and lease is not bounded:   # each execution waited on
                    self._unbound(bounded, me)                      # is bounded from the
                    lease.wait_bounds[me] = loop.time() + max_hold  # moment the wait on it
                    bounded = lease                                 # began
                    self._refresh(lease)
                timeout = None if lease.deadline is None else max(0.0,
                                                                  lease.deadline - loop.time())
                await asyncio.wait([lease.future, lease.nudge], timeout=timeout,  # never cancels
                                   return_when=asyncio.FIRST_COMPLETED)            # either
        finally:
            self._unbound(bounded, me)                      # a waiter that leaves bounds nothing

    def _unbound(self, lease, me):
        if lease is not None and lease.wait_bounds.pop(me, None) is not None:
            self._refresh(lease)

    def _refresh(self, lease):
        """A lease expires at the earliest of: `lease_ttl` after its start or last renewal (only
        when the holder renews), the holder's `max_hold` from its start, and the `max_hold` of
        each request still waiting on it, from when its wait began. Before the start only the
        waiters' bounds run. A deadline that comes forward wakes the sleeping waiters to re-time."""
        ttl = lease.base + self.lease_ttl if lease.started and lease.renews and self.lease_ttl \
            else None
        ends = [x for x in (ttl, lease.hold_until) if x is not None]
        ends.extend(lease.wait_bounds.values())
        old, lease.deadline = lease.deadline, (min(ends) if ends else None)
        if lease.deadline is not None and (old is None or lease.deadline < old) \
                and not lease.future.done():
            nudge, lease.nudge = lease.nudge, asyncio.get_running_loop().create_future()
            nudge.set_result(None)

    def _expired(self, lease, now):
        """End `lease` if its deadline has passed (its waiters look again; one takes over)."""
        if lease.deadline is None or now < lease.deadline:
            return False
        self.counters['leases_expired'] += 1
        self._end_lease(lease)
        return True

    def lease_started(self, token):
        """The grant's reply is leaving (its commit is done): the holder's clocks start now, so
        the commit's latency does not eat into its `max_hold` or its first renewal interval."""
        lease = self._lease_tokens.get(token)
        if lease is None or lease.started:
            return
        lease.started = True
        lease.base = asyncio.get_running_loop().time()
        if lease.max_hold is not None:
            lease.hold_until = lease.base + lease.max_hold
        self._refresh(lease)                                  # nudges the sleeping waiters

    def lease_renew(self, token, connection):
        """The holder is alive and still computing: its lease lives another `lease_ttl` (never past
        a `max_hold`). Only the connection the lease was granted on may renew it; a lease that
        has ended or expired is not renewed (the holder's later write is still accepted)."""
        lease = self._lease_tokens.get(token)
        if lease is None or lease.connection != connection:
            return {'renewed': False}
        now = asyncio.get_running_loop().time()
        if self._expired(lease, now):
            return {'renewed': False}
        lease.base = now
        self._refresh(lease)
        self.counters['lease_renewals'] += 1
        return {'renewed': True}

    def _written(self, address):
        """A value was written at `address`: whoever waits on its lease reads it (whoever wrote it:
        a requester that asked for no lease may write first; values are a function of the
        address)."""
        lease = self._leases.get(address)
        if lease is not None:
            self._end_lease(lease)

    def lease_release(self, token, connection):
        """The holder computed nothing (its code raised or was cancelled): one waiter takes over.
        Only the connection the lease was granted on may release it."""
        lease = self._lease_tokens.get(token)
        if lease is None or lease.connection != connection:
            return {'released': False}
        self.counters['leases_released'] += 1
        self._end_lease(lease)
        return {'released': True}

    def _end_lease(self, lease):
        if self._leases.get(lease.address) is lease:
            del self._leases[lease.address]
        self._lease_tokens.pop(lease.token, None)
        if not lease.future.done():
            lease.future.set_result(None)

    # -- flights --------------------------------------------------------------------------------

    async def _dispatch_loop(self):
        while True:
            await self._wake.wait()
            self._wake.clear()
            while self.queued:
                flight = self._pick()
                if flight is None:              # every queued flight's backend is at its cap
                    break
                flight.state = 'sent'
                flight.t_dispatch = self.clock()
                self.inflight += 1
                self._busy[flight.model] += 1
                self._served[flight.owner] = next(self._dispatches)
                task = asyncio.ensure_future(self._fly(flight))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

    def _pick(self):
        """The next queued flight whose backend has room (the cap applies per backend), or None."""
        return self.queued.pick(lambda model: self._busy[model] < self.cap)

    def _backoff(self, attempt):
        delay = min(self.backoff_cap, self.backoff_base * 2 ** (attempt - 1))
        return delay * (0.5 + self.rng.random())

    async def _attempts(self, flight):
        last = None
        failed_replica = None           # a retry avoids the replica that just failed (failover)
        client = self.models[flight.model]
        with_meta = getattr(client, 'generate_with_meta', None)
        for attempt in range(1, self.max_attempts + 1):
            flight.attempts = attempt
            owner = self.tickets.get(flight.owner)
            ctx = {'salt': getattr(owner, 'salt', None), 'route': flight.route,
                   'attempt': attempt, 'owner': flight.owner, 'node': flight.node,
                   'key_hash': flight.key_hash, 'failed_replica': failed_replica}
            meta = None
            try:
                if with_meta is None:
                    raw = await client.generate(flight.body, flight.key_hash)
                else:
                    raw, meta = await with_meta(flight.body, flight.key_hash, ctx)
                sample = validate_sample(raw)
            except RequestError as err:
                self.counters['request_errors'] += 1
                flight.meta = meta or err.meta
                self._attempt_row(flight, ctx, flight.meta, 'request_error')
                return error_sample(err.kind, err.detail)
            except TransportError as err:
                last = err
                failed_replica = (getattr(err, 'meta', None) or meta or {}).get('replica')
                self.counters['transport_errors'] += 1
                self._attempt_row(flight, ctx, meta if meta is not None else err.meta,
                                  'transport_error')
                if attempt < self.max_attempts:
                    await asyncio.sleep(self._backoff(attempt))
            except asyncio.CancelledError:
                raise
            except Exception as err:        # may depend on the output: never retried
                self._attempt_row(flight, ctx, getattr(err, 'meta', None), 'client_error')
                raise ManagerError('model_client_error', repr(err))
            else:
                flight.meta = meta
                self._attempt_row(flight, ctx, meta, 'ok')
                return sample
        raise ManagerError('flight_failed', 'retries exhausted: %s' % (last,))

    def _attempt_row(self, flight, ctx, meta, outcome):
        """Account for one attempt: the first send time, the cached-field count and, with a
        ledger, its row. Runs on the event-loop thread."""
        meta = meta if isinstance(meta, dict) else None
        if meta is not None:
            if flight.t_first_send is None:
                flight.t_first_send = meta.get('t_send')
            if meta.get('prompt_tokens') is not None and not meta.get('cached_present'):
                self.cached_missing += 1
        if self.ledger is None:
            return
        if flight.prompt_digest is None:
            flight.prompt_digest = digest(flight.body['messages'])
        self.ledger.body(flight.prompt_digest, flight.body['messages'])
        key = flight.key
        routed = getattr(self.models[flight.model], 'route', 'rr') == 'case'
        self.ledger.row(
            'attempt', aid='%s/%d/%d' % (self.ledger.mgr, flight.seq, ctx['attempt']),
            seq=flight.seq, attempt=ctx['attempt'], key_hash=flight.key_hash,
            pool=key.get('pool'), index=key.get('idx'), partition=key.get('part', key.get('gen')),
            case_key=flight.route, owner_ticket=flight.owner, node=flight.node,
            cache_salt=meta['salt'] if meta is not None and 'salt' in meta else ctx['salt'],
            route=ctx['route'] if routed else None,
            replica=None if meta is None else meta.get('replica'),
            instance_url=None if meta is None else meta.get('instance_url'),
            prompt_digest=flight.prompt_digest, max_tokens=flight.body.get('max_tokens'),
            t_created=flight.t_created, t_dispatch=flight.t_dispatch, outcome=outcome, n=1,
            model=flight.model, **meta_fields(meta))

    @staticmethod
    def _metered(result, flight):
        """The successful attempt's meta, for the ticket that generated the sample only."""
        meta = flight.meta if flight is not None else None
        if result['source'] != 'generated' or meta is None:
            return result
        t_send = flight.t_first_send
        result['meta'] = {
            'cached_tokens': meta.get('cached_tokens'), 'cached_present': meta.get('cached_present'),
            'prompt_tokens': meta.get('prompt_tokens'),
            'completion_tokens': meta.get('completion_tokens'),
            'instance_url': meta.get('instance_url'), 't_created': flight.t_created,
            't_send': meta.get('t_send'), 't_done': meta.get('t_done'),
            'queue_s': (t_send - flight.t_created) if t_send is not None
            and flight.t_created is not None else None,
            'attempts': flight.attempts, 'response_id': meta.get('response_id')}
        return result

    async def _fly(self, flight):
        try:
            sample = await self._attempts(flight)
            try:
                row, inserted = self.store.put_once(flight.key_hash, flight.key, sample,
                                                    flight.owner, flight.attempts, self.clock(),
                                                    node=flight.node)
            except Exception as err:
                raise ManagerError('store_failed', str(err))
            if not inserted:                # impossible under single flight; counted, never hidden
                self.counters['write_conflicts'] += 1
            await self.durable()            # write before wake
        except ManagerError as err:
            self.counters['flights_failed'] += 1
            self._settle(flight, error=err)
        else:
            self.counters['generated'] += 1
            self._settle(flight, row=row)   # the row is committed: write before wake
        finally:
            self.inflight -= 1
            self._busy[flight.model] -= 1
            self._wake.set()

    def _settle(self, flight, row=None, error=None):
        if self.flights.get(flight.key_hash) is flight:
            del self.flights[flight.key_hash]
        if flight.future.done():
            return
        if error is None:
            flight.future.set_result(row)
        else:
            flight.future.set_exception(error)

    def read(self, key_hashes):
        """Stored samples by key hash (None where absent): how a client reads the intermediate
        outputs a record names. Read-only; never opens a flight."""
        out = {}
        for key_hash in key_hashes:
            row = self.store.get(key_hash)
            out[key_hash] = None if row is None else row['sample']
        return out

    def stats(self):
        out = dict(self.counters)
        out.update(flights=len(self.flights), queued=len(self.queued), inflight=self.inflight,
                   tickets=len(self.tickets), entries=self.store.count(),
                   scopes=len(self._pool_served), asked_scopes=len(self._asked),
                   commits=getattr(self.store, 'commits', None), leases_held=len(self._leases),
                   leases_overdue=sum(1 for l in self._leases.values() if l.deadline is not None
                                      and l.deadline <= time.monotonic()),
                   lease_oldest_s=round(self.clock() - min(l.t for l in self._leases.values()), 3)
                   if self._leases else None)
        if len(self.models) > 1:
            out['inflight_by_model'] = {name: self._busy[name] for name in sorted(self.models)}
        if self.ledger is not None:
            out['ledger'] = {'rows': self.ledger.rows, 'cached_missing': self.cached_missing,
                             'probe': self.ledger.probe}
        return out
