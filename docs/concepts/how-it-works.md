# How Copula works

```text
 copula run ── registers the run ──▶ ┌──────────────── Copula server ───────────────┐
     │                               │ registry   runs, versions, slots, groups      │
     ▼                               │ store      samples, by address                │──▶ model server
 your program ── HTTP requests ────▶ │ call log   every answered call, and why       │    (vLLM, SGLang,
 (unchanged)  ◀── responses ──────── └──────────────────────────────────────────────┘     an API)
```

The program talks to Copula as if it were the model server. Copula decides, for each request,
which stored sample answers it, or generates a new one and stores it.

## The life of one call

1. **The run is registered.** `copula run` tells the server the experiment, the coordinates of the
   run and the content digests of the files that decided earlier versions. The server answers
   with the run's **version**, **slot** and **group**, and a URL with a token that names the run.
2. **A request arrives** on the run's URL. Headers may name a **unit** inside the run (an
   evaluation of one candidate on one case); a unit has a group of its own.
3. **Copula classifies the call** from the request: a sampled call (cached per group), a greedy
   call (cached once for all runs, when the model's identity is established) or an uncached call
   (a role routed `fresh`). A request Copula cannot serve faithfully, such as one that relies on
   provider-side state, is refused.
4. **The call gets an address.** For a sampled call it is the group, the model revision, the
   request after canonicalisation, and the **occurrence**: this is the k-th time the run sends
   this request in this group.
5. **The address is looked up.** A stored sample is returned (*reused*). If another run is
   generating that very sample, the call waits for it (*joined*). Otherwise the sample is
   generated upstream, stored, and returned (*generated*).
6. **The call is logged**, with its group, request digest and outcome, so that `copula explain`
   can later say why it was generated.
7. **The run ends.** The launcher sends the set of project files the program read; their content
   decides whether the next run is a new version or an unchanged rerun.

## Occurrences: repeats stay independent

Within a group, the k-th identical request of a run reads the group's k-th sample. A run that asks
the same question three times gets samples 1, 2 and 3, three independent draws, as it would
without Copula. A later version of the program that asks it five times reads samples 1 to 3 and
generates 4 and 5. A run never reads one sample twice, and it never reads a sample it did not ask
for.

## Versions and slots: who shares

A **group** is a pool of samples. Which runs share a group is decided by the registry:

- A run whose program or files changed since every earlier run of the experiment, or whose
  compare coordinates differ, is a **new version**. It takes the smallest slot its version has
  not used, which is slot 1, and slot 1 of every version of an experiment (with equal replicate
  and case values) is one group. So all versions share.
- A run whose program and files are unchanged since an earlier run is a **rerun** of that version.
  It takes the next slot (2, 3, ...), and a new slot is a new group. So reruns get new samples.
- Runs that differ in a **replicate** or **case** coordinate, in the **phase** or in the
  **experiment** are in different groups whatever their slot.

The quickstart's runs, side by side:

| run | what changed | version | slot | group | shares with |
|---|---|---|---|---|---|
| 1 | first run | v1 | 1 | A | |
| 2 | the check prompt | v2 | 1 | A | run 1, on the calls the edit did not change |
| 3 | nothing | v2 | 2 | B | nobody: a rerun is a new replicate |
| `--set seed=1` | a replicate coordinate | v2 | 1 | C | other runs with seed 1 |

## Greedy calls

A call with temperature 0 (or top_k 1) has one answer per model, so it does not need numbering.
When the model's identity is established (the files it loaded are known), Copula caches greedy
calls once for every run, whatever its experiment, phase or group, and repeats read that one
answer, as they would natively. Without an established identity, greedy calls are treated like sampled ones, per group
([When two calls are the same](identical-calls.md)).

## Concurrency and failures

- One server holds the store and serves every run of the project; identical samples that are
  needed at the same moment are generated once.
- Transport failures and an overloaded or misconfigured upstream are retried and never stored;
  an error the request itself causes (a prompt that is too long) is stored as that sample's
  result, as the model server would give it every time. A sample is written only complete.
- A client retry (the openai SDK's `x-stainless-retry-count`, or an `Idempotency-Key`) of a
  request whose response was lost reads the same occurrence again instead of taking the next
  one, so client retries do not consume samples.
- A run that ended cannot send more requests: a late attempt would re-read samples its client
  already received.

## Terms

| term | meaning |
|---|---|
| run | one launched command (a *process run*), or a *unit* inside it |
| unit | a run inside a process run, with coordinates of its own: one evaluation of one candidate on one case, say |
| experiment | the runs that may share samples (`--exp NAME`) |
| version | the program as it was: its compare coordinates and the content of the files it read |
| slot | the number of a run among the runs of its version with the same replicate and case values: 1 for the first, 2 for its first rerun, ... |
| group | a pool of samples; runs in one group share |
| occurrence | for a request, the number of times the run has sent it in its group so far; the k-th occurrence reads sample k (`x-copula-rank`) |
| ordinal | for a unit, the number of units with the same coordinates the process has opened so far |
| address | what identifies a stored sample ([When two calls are the same](identical-calls.md)) |
| phase | `search` or `holdout`; their sampled calls never share |

`copula explain` names the model a call went to and its class: `actor:s` is a sampled call to the
default model (`actor`), `actor:d` a greedy one, `actor:u` an uncached one; other served models are
`m1`, `m2`, ...

## Where the ideas come from

Numbering identical requests so that the k-th repeat reads the k-th stored sample is the rule of
[Cache Saver and Mnimi](comparison.md#occurrence-numbered-caches). Copula adds the run structure
around it: which runs share a group, decided from versions, slots and coordinates; one
concurrent, crash-safe store for every process of a project; and explanations of every miss.
