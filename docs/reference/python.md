# Python API

`import copula` needs only the standard library. A program run under `copula run` needs none of
this: it is for programs that want to describe their structure more precisely than Copula infers
it.

## Contexts

The contexts below travel with the code: into nested calls, asyncio tasks and, once the process is
attached, `ThreadPoolExecutor` workers. They reach Copula as `X-Copula-*` headers on the requests
made inside them ([HTTP interface](http.md)). Attaching happens automatically under
`copula run --exp`; without it, pass `copula.headers()` to your client yourself.

### `copula.run(**coordinates)`

```python
with copula.run(candidate=cand_id, case=example_id):
    agent(example)
```

A unit: the calls made inside belong to a run of their own, with these coordinates merged over
the enclosing ones. The k-th unit with the same coordinates in one process is a new unit (ordinal
k), so it reads new samples. `copula.run()` without coordinates opens no unit. See
[Runs, coordinates and versions](../concepts/runs-and-coordinates.md).

### `copula.holdout()`

```python
with copula.holdout():
    for example in test_set:
        with copula.run(case=example["id"]):
            evaluate(best, example)
```

Units opened inside are in the holdout phase, which has groups of its own: its sampled calls
share nothing with the search phase. A holdout unit whose case the search phase used is refused
(HTTP 409). Calls made inside `holdout()` but outside any `copula.run` all belong to one holdout
unit of the process.

### `copula.decision(name=None)`

```python
with copula.decision():
    parent_score = evaluate(parent, minibatch)
    child_score = evaluate(child, minibatch)
```

A decision scope: a unit inside it reads the slot of the scope's first unit on the same case
(unless it has used that slot already), so a parent and a child compared in one decision see the
same samples where their calls agree. Without a name, each scope gets a new id.

### `copula.role(name)`

```python
with copula.role("proposer"):
    new_prompt = reflect(traces)
```

Calls made inside carry the role, which `copula run --route ROLE=RULE` places
([Routes](../guides/optimizers.md#routes-for-roles)).

## Functions

### `copula.headers(role=None)`

The `X-Copula-*` headers of the current context, as a dict, for a client that `attach()` does not
cover:

```python
requests.post(url, json=body, headers={**copula.headers(), "Authorization": "Bearer x"})
```

### `copula.chat(messages, model=None, **params)`

A chat completion through the run's Copula endpoint with the standard library alone. Returns the
response as a dict, with the `x-copula-*` response headers under `"_copula"`. `model` defaults to
`$COPULA_MODEL`; other keyword arguments are request fields (`temperature=0.7`). Also takes
`base_url`, `role`, `timeout` (600 s) and `extra_headers`.

### `copula.attach(exp=None, url=None)`

Attaches the process to its run: copies the context into `ThreadPoolExecutor` workers, adds the
headers to requests sent to Copula through httpx and http.client (urllib, urllib3, requests), and
installs the framework adapters when DSPy or GEPA is imported. It is idempotent, and `copula run
--exp` calls it before the program starts. `url` defaults to `$COPULA_URL`, which `copula run`
sets; without a URL it raises `RuntimeError`. `exp`, if given, must equal the launcher's
`--exp`.

### `copula.__version__`

The version string.

## The kernel API

`copula.Session` is the interface of the paper's kernel: runs, groups and registered nodes, with
no launcher, registry or inference. The program declares everything itself: each node's
randomness and caching, and the group each run belongs to. It talks to a kernel manager, the
process that owns the sample store.

<!-- docs-test: skip -->
```console
$ python -m copula.kernel.manager --store kernel/store.db --socket kernel/m.sock --mock demo &
```
<!-- docs-test: run mkdir -p kernel -->
<!-- docs-test: bg python -m copula.kernel.manager --store kernel/store.db --socket kernel/m.sock --mock demo -->
<!-- docs-test: wait kernel/m.sock -->

```python title="kernel_demo.py"
from copula import Session

s = Session.connect_sync("kernel/m.sock")
s.register_llm("draft", cache=True, randomness="stochastic", pin={"temperature": 0.7})
s.register_llm("judge", cache=True, randomness="deterministic", pin={"temperature": 0.0})
dev = s.group("dev")


def agent(question):
    draft = s.chat_sync("draft", [{"role": "user", "content": question}])
    text = draft.choices[0].message.content
    verdict = s.chat_sync("judge", [{"role": "user", "content": "Is this right? " + text}])
    return text, verdict.choices[0].message.content


with s.run_sync(dev):           # a run in group "dev"
    print(agent("Why is the sky blue?"))
with s.run_sync(dev):           # another run of the group: its k-th call reads sample k
    print(agent("Why is the sky blue?"))
with s.run_sync(s.group()):     # a fresh group: new samples
    print(agent("Why is the sky blue?"))
s.close()
```

```console
$ python kernel_demo.py
('mock:f7b00b470587d35423fb0a4d', 'mock:6802f5537d11906853e5f288')
('mock:f7b00b470587d35423fb0a4d', 'mock:6802f5537d11906853e5f288')
('mock:5cdea639a2288d3e69098660', 'mock:4a0f5bd14755480cb74824af')
```

| call | what it does |
|---|---|
| `Session.connect_sync(socket)`, `await Session.connect(socket)` | a session on a manager's Unix socket (`trace=PATH` writes a trace) |
| `s.register_llm(name, *, cache, randomness=None, backend="actor", pin=None, max_tokens=1024)` | a model node: requests whose `model` is `name` go to `backend` with `pin` fixed. A cached node declares `randomness`, and a deterministic one must pin greedy decoding |
| `@s.tool(name=None, version=1, *, cache, randomness=None)` | a tool node; a cached stochastic tool takes an `rng` argument seeded from its address |
| `s.group(name=None)` | a named group (the same name is the same pool), or a fresh one |
| `with s.run_sync(group):`, `async with s.run(group):` | a run in the group; `run_id=` resumes a run, re-reading its samples in order |
| `s.chat_sync(model, messages, **params)`, `await s.chat(...)` | a call of a registered node inside a run |
| `s.wrap(client)` | routes an `openai` client's chat completions through the session |
| `s.close()`, `await s.aclose()` | closes the session |

`copula.Group`, `copula.Run`, `copula.Namespace` (the response objects) and `copula.SDKError`
are exported too. `copula.kernel` holds the full kernel, byte for byte the code the paper
evaluated ([Development](../project/development.md)).
