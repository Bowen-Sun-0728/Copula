"""The sampling design `{rounds, fanout}` and its coordinates (INTERFACE_V0 section 4).

Every node carries the round coordinate r in 1..R. `fanout[k] = m` gives node k and all of its
descendants an extra coordinate k in 1..m. A node's coordinate set is {r} plus the fanout
coordinates of itself and its ancestors. A draw of a node is one assignment of its coordinates; it
reads from each parent the parent draw obtained by restricting that assignment to the parent's
coordinates, so coordinates two branches share are paired and private ones are crossed. v0 has no
zip below a fanout.
"""
import itertools


class DesignError(ValueError):
    """An invalid design or draw."""


def _require(ok, message):
    if not ok:
        raise DesignError(message)


class Draw:
    """One assignment of a node's coordinates: the round `r` and one (node_id, j) per fanout.

    Coordinates equal to 1 are kept here, because restriction needs them, and are dropped only in
    the canonical form. That is what makes draw (r=1) of a uniform design and draw (r=1, draft=1)
    of a nested design the same cache key. The round is a separate field, so a node whose id is
    'r' cannot collide with it.
    """
    __slots__ = ('r', 'fan')

    def __init__(self, r, fan=()):
        _require(type(r) is int and r >= 1, 'round must be an int >= 1, got %r' % (r,))
        items = tuple(sorted(fan.items() if isinstance(fan, dict) else fan))
        ids = [node_id for node_id, _ in items]
        _require(len(set(ids)) == len(ids), 'duplicate fanout coordinate in %r' % (items,))
        for node_id, j in items:
            _require(isinstance(node_id, str) and node_id, 'fanout coordinate names a node id')
            _require(type(j) is int and j >= 1, 'fanout coordinate %s must be an int >= 1' % node_id)
        self.r = r
        self.fan = items

    def canonical(self):
        """The form hashed into a key: `r` always, fanout coordinates only where they exceed 1."""
        return {'r': self.r, 'fan': [[node_id, j] for node_id, j in self.fan if j != 1]}

    def restrict(self, coords):
        keep = set(coords)
        return Draw(self.r, [(node_id, j) for node_id, j in self.fan if node_id in keep])

    def to_wire(self):
        return {'r': self.r, 'fan': dict(self.fan)}

    @classmethod
    def from_wire(cls, obj):
        _require(isinstance(obj, dict) and set(obj) <= {'r', 'fan'} and 'r' in obj,
                 'a draw is {r, fan}')
        fan = obj.get('fan', {})
        _require(isinstance(fan, dict), 'draw fan must be an object')
        return cls(obj['r'], fan)

    def __eq__(self, other):
        return isinstance(other, Draw) and (self.r, self.fan) == (other.r, other.fan)

    def __hash__(self):
        return hash((self.r, self.fan))

    def __repr__(self):
        inner = ', '.join('%s=%d' % pair for pair in self.fan)
        return 'Draw(r=%d%s)' % (self.r, ('; ' + inner) if inner else '')


def topological_order(parents):
    """Deterministic topological order of a DAG given as {node: [parents]}; refuses cycles."""
    for node, ps in parents.items():
        for p in ps:
            _require(p in parents, 'node %r reads unknown node %r' % (node, p))
    remaining = {node: set(ps) for node, ps in parents.items()}
    order = []
    while remaining:
        ready = sorted(node for node, ps in remaining.items() if not ps)
        _require(ready, 'graph has a cycle among %s' % sorted(remaining))
        for node in ready:
            order.append(node)
            del remaining[node]
        for ps in remaining.values():
            ps.difference_update(ready)
    return order


class Design:
    """`{rounds, fanout}` bound to one DAG.

    `parents` maps every node to its parents. `allowed`, when given, is the set of nodes that may
    carry a fanout (v0: stochastic nodes only).
    """

    def __init__(self, rounds, fanout, parents, allowed=None):
        _require(type(rounds) is int and rounds >= 1, 'rounds must be an int >= 1')
        _require(isinstance(fanout, dict), 'fanout must be an object')
        for node, m in fanout.items():
            _require(node in parents, 'fanout names unknown node %r' % (node,))
            _require(allowed is None or node in allowed, 'fanout on %r is not allowed' % (node,))
            _require(type(m) is int and m >= 1, 'fanout of %r must be an int >= 1' % (node,))
        self.rounds = rounds
        self.fanout = dict(sorted(fanout.items()))
        self.parents = {node: list(ps) for node, ps in parents.items()}
        self.order = topological_order(self.parents)
        ancestors = {}
        for node in self.order:
            acc = set()
            for p in self.parents[node]:
                acc.add(p)
                acc |= ancestors[p]
            ancestors[node] = acc
        self.ancestors = ancestors
        self.coords = {node: tuple(sorted(k for k in self.fanout
                                          if k == node or k in ancestors[node]))
                       for node in self.order}

    def as_dict(self):
        return {'rounds': self.rounds, 'fanout': dict(self.fanout)}

    def size(self, node):
        """N_k: the number of draws of `node` per case."""
        n = self.rounds
        for k in self.coords[node]:
            n *= self.fanout[k]
        return n

    def draws(self, node):
        """Every draw of `node`, in a fixed order."""
        coords = self.coords[node]
        ranges = [range(1, self.fanout[k] + 1) for k in coords]
        for r in range(1, self.rounds + 1):
            for combo in itertools.product(*ranges):
                yield Draw(r, list(zip(coords, combo)))

    def parent_draw(self, draw, parent):
        """The draw of `parent` that `draw` reads: its assignment restricted to the parent."""
        return draw.restrict(self.coords[parent])


def descendants(parents):
    """{node: set of its descendants} for a DAG given as {node: [parents]}."""
    children = {node: set() for node in parents}
    for node, ps in parents.items():
        for p in ps:
            children[p].add(node)
    out = {}
    for node in reversed(topological_order(parents)):
        acc = set()
        for c in children[node]:
            acc.add(c)
            acc |= out[c]
        out[node] = acc
    return out


def pilot_nodes(parents, stochastic):
    """The generate nodes a pick-freeze pilot redraws (INTERFACE.md section 3): those whose closure
    (the node and its descendants) does not contain every generate node. Redrawing a node whose
    closure holds every generate node would only repeat a round."""
    below = descendants(parents)
    return sorted(k for k in stochastic if not set(stochastic) <= ({k} | below[k]))


class PilotDesign(Design):
    """The pick-freeze pilot `{rounds: R, pilot: "pick_freeze"}` (INTERFACE.md section 3): per round a
    base draw of the whole graph, plus, for every redrawn node a, one redraw of a and its
    descendants with everything else kept. Coordinates are those of a design with fanout 2 on every
    redrawn node, so parents are read by restriction as in any design; but the draws form a star,
    not a product: a node's draws are the base (every coordinate 1) and one draw per redrawn
    ancestor-or-self a with coordinate a = 2. Draws are listed base first, then the redraws in
    topological order of a: that order is the role order in which pilots claim samples."""

    def __init__(self, rounds, redrawn, parents, allowed=None):
        super().__init__(rounds, {a: 2 for a in redrawn}, parents, allowed)
        self.redrawn = sorted(redrawn, key=self.order.index)

    def as_dict(self):
        return {'rounds': self.rounds, 'pilot': 'pick_freeze', 'redrawn': list(self.redrawn)}

    def size(self, node):
        return self.rounds * (1 + len(self.coords[node]))

    def arms(self, node):
        """The redrawn nodes whose redraw reaches `node`, in role order."""
        return [a for a in self.redrawn if a in self.coords[node]]

    def draws(self, node):
        coords = self.coords[node]
        for r in range(1, self.rounds + 1):
            yield Draw(r, [(k, 1) for k in coords])
            for a in self.arms(node):
                yield Draw(r, [(k, 2 if k == a else 1) for k in coords])
