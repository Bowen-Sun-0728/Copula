# Prompt and harness optimizers

An optimizer evaluates many candidates of a program on many cases, and most candidates differ from
their parent in one place. With Copula, a candidate pays only for the calls its edit changed, and
the comparison of a child with its parent is made on the same samples wherever the two agree.

For that, Copula needs to know where each evaluation begins and ends. A whole optimizer process is
one run, and inside one run the k-th identical request reads the k-th sample, so two candidates
that make the same call inside one process would get different samples. Each evaluation of a
candidate on a case is therefore a **unit**: a run of its own inside the process, with
coordinates such as `candidate` and `case`.

## DSPy and GEPA, unchanged

Under `copula run --exp`, a DSPy program gets units without any change: the DSPy adapter makes each
top-level module call a unit whose `candidate` is a hash of the module's state (instructions,
demos, signatures) and whose `case` is a hash of the call's inputs. The seed of an optimizer is a
replicate coordinate `optimizer_seed`.

<!-- docs-test: skip -->
```console
$ copula run --exp gepa -- python gepa_math.py --out results/v1.json
```

This is the output of a cosmetic edit to such a program (DSPy 3 with GEPA, Qwen2.5-1.5B served by
vLLM): 234 of its 238 calls were read from the first run.

```text
copula: run r-d1938878157b4115  exp=gepa  dev: version v-116788f2fcb14d7f, slot 1  group g-f81577c06785524d543be1b3
copula: coordinates: {"annotation": ["out"]}
copula: new version (compared with earlier versions of gepa)
copula: attach: attached to http://127.0.0.1:32841/ (threads copy context; headers on httpx, httpx2, http.client)
copula: attach: GEPA adapter on: one decision scope per proposal, proposer role for reflection
copula: attach: DSPy response cache turned off (W11): it returned one stored sample for every repeat; repeated evaluations are now new samples, reused across runs by Copula
copula: attach: DSPy adapter on: units from module calls, candidates and seeds from optimizers, holdout evaluations
copula: attach: W5b: an Evaluate after compile scores 8 of 8 examples the optimizer used; those scores are optimistic
copula: run r-d1938878157b4115 done: 238 calls: 234 reused, 4 generated, 0 joined, 216 units
```

What the adapters do:

- **DSPy's response cache is turned off.** It returns one stored response for every repeat of a
  request, so repeated evaluations of a candidate would all see one sample. Copula reuses samples
  across runs instead, and keeps repeats independent (note W11).
- **Holdout is inferred.** A `dspy.Evaluate` after `compile`, outside it, on examples the optimizer
  was never given, is a holdout evaluation: it gets samples that no search evaluation read. An
  `Evaluate` on examples the optimizer used stays in the search phase, with warning W5b: its
  scores are optimistic.
- **GEPA proposals are decision scopes.** Each proposal (evaluate the parent on a minibatch,
  reflect, evaluate the child) is one decision: on each case, the child reads the parent's slot
  when it can, so the acceptance test is paired.
- **Reflection calls carry the role `proposer`.** `--route proposer=RULE` decides their group
  ([Routes](#routes-for-roles)).

The few calls an edit regenerates without changing their request come from evaluation order:
identical units (same candidate, same case) are numbered in the order they start, and a
minibatch that holds one example twice can take them in another order
([`arrival_order`](why-generated.md#arrival_order)).

## Your own optimizer loop

A hand-written loop marks its units with `copula.run(**coordinates)`. Here three candidate system
prompts are scored on three training questions, and the best one is evaluated on two test
questions in the holdout phase:

```python title="optimize.py"
import copula
from openai import OpenAI

client = OpenAI()
TRAIN = ["Why is the sky blue?", "What is a prime number?", "Who wrote Hamlet?"]
TEST = ["What is DNA?", "Why do leaves fall?"]
CANDIDATES = ["Answer in one sentence.", "Answer in one sentence. Be precise.",
              "Answer briefly and cite one fact."]


def solve(system, question):
    r = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct", temperature=0.7,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": question}])
    return r.choices[0].message.content


def score(answer):
    return len(answer) % 7          # stands in for a real metric


best = None
for i, system in enumerate(CANDIDATES):
    total = 0
    for q in TRAIN:
        with copula.run(candidate=i, case=q):      # one unit per (candidate, case)
            total += score(solve(system, q))
    if best is None or total > best[0]:
        best = (total, system)

with copula.holdout():                             # final numbers on samples search never read
    for q in TEST:
        with copula.run(case=q):
            solve(best[1], q)
print("best:", best[1])
```

```console
$ copula run --exp opt --mock -- python optimize.py
copula: run r-7bb1bf2e72f74072  exp=opt  dev: version v-929d10299905480b, slot 1  group g-7c415bbbf520985501803771
copula: served by embedded server (mock model)
copula: new version (compared with earlier versions of opt)
copula: attach: attached to http://127.0.0.1:45725/ (threads copy context; headers on httpx, httpx2, http.client)
best: Answer in one sentence.
copula: run r-7bb1bf2e72f74072 done: 11 calls: 0 reused, 11 generated, 0 joined, 11 units
copula:   11 generated: new request -- no earlier run of the experiment called this model
copula:   details: copula explain r-7bb1bf2e72f74072
```

Now replace the third candidate and run again:

```python
CANDIDATES = [..., "Answer briefly and name your source."]
```
<!-- docs-test: run sed -i 's/Answer briefly and cite one fact./Answer briefly and name your source./' optimize.py -->

```console
$ copula run --exp opt --mock -- python optimize.py
copula: run r-41771a4bf73c4b9e  exp=opt  dev: version v-2a09fbea2e744863, slot 1  group g-7c415bbbf520985501803771
copula: served by embedded server (mock model)
copula: new version (compared with earlier versions of opt)
copula: attach: attached to http://127.0.0.1:41315/ (threads copy context; headers on httpx, httpx2, http.client)
best: Answer in one sentence.
copula: run r-41771a4bf73c4b9e done: 11 calls: 8 reused, 3 generated, 0 joined, 11 units
copula:   3 generated: new request -- closest earlier request (run r-7bb1bf2e72f74072): messages[0] (system): '...briefly and cite one fact.' -> '...briefly and name your source.'
copula:   details: copula explain r-41771a4bf73c4b9e
```

The first two candidates read all their samples from the first run; only the new candidate's
three calls were generated. The holdout evaluation is shared too when the same candidate wins:
holdout units of different versions share with each other, as search units do, but never with the
search phase.

`copula.run` blocks nest: a unit opened inside another merges its coordinates over the outer
ones. The context follows the code into `ThreadPoolExecutor` workers and asyncio tasks, so units
work in parallel evaluation loops.

## Decisions

When a child candidate is accepted or rejected by comparing it with its parent on the same cases,
the comparison is sharper if both read the same samples where their calls agree. Wrap the pair in
`copula.decision()`:

<!-- docs-test: skip -->
```python
with copula.decision():
    parent_scores = [evaluate(parent, case) for case in minibatch]   # units inside
    child_scores = [evaluate(child, case) for case in minibatch]
```

Inside a decision, the units on one case read the samples of the decision's first unit on that
case (its slot), unless this unit has already used that slot.

## Routes for roles

Calls made inside `copula.role("proposer")` (or with the header `X-Copula-Role: proposer`) carry a
role, and `--route ROLE=RULE` chooses their group:

| rule | the calls of that role |
|---|---|
| `run` (default) | the group of the run or unit they are made in |
| `global` | one group per experiment, phase and role: every run shares them |
| `per:COORD` | one group per value of the run's coordinate `COORD`, e.g. `per:seed` |
| `fresh` | uncached: a new sample every time |

For example, `--route proposer=per:seed` lets two runs with the same seed share their reflection
calls, while different seeds propose independently. Routes apply within an experiment; a run
without `--exp` ignores them. Calls routed `global` or `per:` are shared across slots, so they
are not independent between reruns.
