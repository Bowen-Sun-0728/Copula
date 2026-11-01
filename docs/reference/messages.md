# Messages

`copula run` writes its messages to stderr, each line starting with `copula:`. This page explains
them in the order they appear.

Some messages carry a code, so that they can be searched for and named in bug reports:

| code | condition |
|---|---|
| O1 | a run without an experiment gets a fresh group |
| V2 | a seed in the request selects its own sample stream |
| W1 | a run of an experiment reused every sample |
| W5a | a holdout unit's case was used in the search phase (refused) |
| W5b | an evaluation after optimisation scores examples the optimizer used |
| W8 | a framework cache could not be turned off |
| W11 | a framework cache was turned off |
| W12 | a model's identity is not established: greedy calls are cached per group |

## Before the command starts

### Placement

```text
copula: run r-5e77ae7d7cf040a3  exp=demo  dev: version v-dcb9767f264042f9, slot 1  group g-0e5d5aee9cfa7a183956942b
```

The run id, the experiment (`-` for none), the recipe, and where the run was placed: its version,
slot and group ([How Copula works](../concepts/how-it-works.md#versions-and-slots-who-shares)).
Without `--exp`: `no experiment: fresh group (O1)`.

### `served by`

`served by embedded server (...)` when this command serves the run itself, `served by server URL
(...)` when a `copula server` does. In parentheses: the upstream's kind and URL, and `engine seed
per address` when vLLM gets per-sample engine seeds.

### Upstream notes

| note | meaning |
|---|---|
| `sampling defaults: MODEL generation_config` | the pinned sampling table comes from the model's `generation_config.json` |
| `sampling defaults: MODEL --defaults` | it comes from `--defaults` |
| `sampling defaults: MODEL neutral (...)` | no `generation_config.json` found (no model files, or none among them): neutral values. Native vLLM would apply the model's `generation_config.json` if it has one; pass `--model-dir` |
| `identity of MODEL not established (...)` (W12) | greedy calls to it are cached per group, not globally; `--model-dir` establishes it |
| `API upstream: temperature and top_p are always sent ...` | an API upstream gets OpenAI's defaults when the client sets none; pass `--defaults` for another provider |
| `an API upstream takes no engine seed` | `--engine-seed address` was given for an API upstream and is ignored |

### `coordinates:`

The coordinate names of the run by class, e.g. `{"compare": ["variant"], "replicate": ["seed"]}`.
Look here when runs you expected to share did not: an unknown name is a replicate.

### Version

| line | meaning |
|---|---|
| `new version (compared with earlier versions of EXP)` | the program, its files or its compare values changed: it shares with earlier versions |
| `unchanged since run R: a rerun (new slot, new samples)` | nothing changed: a new replicate |
| `unchanged since run R; first run of this version with these replicate or case values (new group, new samples)` | nothing changed in the program, but the replicate or case values are new |
| `run R of this experiment is still running, so its files are not known yet: counted as unchanged ...` | a rerun of a run that has not finished |
| `run R failed, so the files it read are unknown: counted as unchanged ...` | a rerun of a failed run |
| `holdout phase: new group space; ...` | `--phase holdout`: the run's groups are apart from the search phase's; whether its cases overlap the search phase is checked for units only |

### Attach

Lines from Python children, starting with `copula: attach:`; `COPULA_QUIET=1` silences them.

| line | meaning |
|---|---|
| `attached to URL (...)` | the process is attached to its run |
| `DSPy adapter on: ...`, `GEPA adapter on: ...` | an adapter was installed when the framework was imported |
| `DSPy response cache turned off (W11): ...` | DSPy's cache would have answered repeats from one sample; Copula reuses samples instead |
| `could not turn off the DSPy cache (W8): ...` | repeats may be collapsed by DSPy before they reach Copula |
| `W5b: an Evaluate after compile scores N of M examples the optimizer used; those scores are optimistic` | an evaluation on selection data, left in the search phase |
| `adapter NAME failed for ...` | the adapter could not be installed; the program runs without it |

## After the command ends

### Summary

```text
copula: run r-5e77ae7d7cf040a3 done: 6 calls: 3 reused, 3 generated, 0 joined
```

Calls by outcome: **reused** (read from the store), **generated**, **joined** (waited for a
generation in flight), and, when there are any, **failed** calls and the number of **units**.
The outcomes count samples, so a request with `n=3` counts three times; `copula explain` counts
requests.

### `N generated: REASON -- EXAMPLE`

Up to three reasons why calls were generated, with one example each
([Why was this call generated?](../guides/why-generated.md)).

### `note (Nx): ...`

Notes about the run's requests, with their count:

| note | meaning |
|---|---|
| `seed S selects its own sample stream (V2); repeats inside one run are numbered, not identical` | identical seeded requests inside one run get different samples of the seed's stream ([Seeds](../concepts/identical-calls.md#seeds)) |
| `greedy, but no established upstream identity: cached per group (W12)` | greedy calls were treated like sampled ones |
| `retry re-read an undelivered occurrence` | a client retry read the sample whose response it had lost |

### W1

```text
copula:   W1: every sample was read from earlier runs; if you wanted new samples, this run is a replicate (--set rep=2)
```

A run of an experiment generated nothing: everything was reused. That is what a cosmetic edit
should do; if you meant to draw new samples, run a replicate.

### Failed command

`the command failed: the files it read are recorded as unknown for versioning` means the command
exited non-zero: its read set is not recorded, and the next run is counted as a rerun of it.

### `details: copula explain RUN`

Shown when calls were generated: the command that explains them.
