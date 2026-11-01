# Guarantees and limits

## What is guaranteed

!!! success "Each run alone sees what it would see without Copula"
    Every run, on its own, sees samples distributed as they would be without Copula: its i-th
    repeat of a request is a fresh draw that it has not seen before.

This holds under three conditions:

- **(P)** What the program does, and which cases it runs, does not depend on what is already in
  the store.
- **(C1)** No run reads one sample address twice.
- **(C2)** The next address a run asks for never depends on a sample it has not read.

Copula enforces C1 and C2: occurrences are numbered per run, a run that ended cannot send more
requests, and a client retry re-reads only an occurrence whose response it never received.
Condition P is yours: a program that, say, skips cases it finds already answered breaks it.

What sharing changes is the **joint** behaviour of several runs: two versions that read the same
samples are correlated. That is the point. Their difference has less noise than the difference of
two independent runs, and it costs less.

## What to keep in mind when you analyse results

!!! warning "Selecting on shared samples is optimistic"
    A version chosen *because* it scored best on some samples will score lower on new ones.
    Report final numbers from the [holdout phase](../guides/compare-and-repeat.md#final-numbers-the-holdout-phase),
    whose sampled calls no search run has read.

!!! warning "Versions that share are not independent replicates"
    The scores of five versions that share samples are correlated, so their spread is not the
    spread of five independent runs. Estimate variability from replicates (seeds, reruns), which
    never share.

## Where outputs can differ from native runs

- **Identical units in another order.** Units with equal coordinates are numbered in the order
  they start. If that order changes between runs, they swap samples: each still sees a fresh
  draw, but the assignment is not a byte-exact repeat. A coordinate that tells them apart
  (`case=...`) makes it stable ([`arrival_order`](../guides/why-generated.md#arrival_order)).
- **Seeded repeats inside a run.** On a server that honours seeds, two identical seeded requests
  in one run usually return the same text natively; under Copula they are two samples of the
  seed's stream ([Seeds](identical-calls.md#seeds)).
- **Sampling defaults.** Copula sends every sampling parameter explicitly. On vLLM and SGLang the
  values come from the model's `generation_config.json` when Copula finds the model files, as
  vLLM does natively; otherwise they are neutral, and a note says so
  ([Connect a model server](../guides/model-servers.md#vllm-and-sglang)).
- **`max_tokens`.** Requests that set none get `--max-tokens` (1024).
- **Text parts and roles.** A content list of text parts is sent as one string, its parts joined
  by newlines; `developer` messages are sent as `system`.
- **Streaming.** Streams are buffered: the content is the same, but it arrives at the end.
- **Unknown file contents count as unchanged.** A run whose earlier run's files cannot be read
  counts as a rerun and gets new samples. Files over 256 MB are compared by size and modification
  time.
- **Reverted edits and runs after a failure** count as reruns and get new samples
  ([Versions](runs-and-coordinates.md#versions)).

## Limits of this version

- **Requests**: text content only (no images, audio or files); no request field Copula does not
  know (`logprobs`, `logit_bias`, `best_of`, `reasoning_effort`, vLLM's `min_tokens` or
  `guided_json`, ...); no requests that rely on provider-side state or server-side tools; no
  embeddings or legacy text completions ([Refused fields](identical-calls.md#refused-fields)).
- **Model servers**: OpenAI-compatible ones (vLLM, SGLang, hosted APIs with an OpenAI-compatible
  endpoint). Anthropic clients work against them, but Anthropic's own API is not an upstream
  yet.
- **Tools**: tool calls in model requests are served; the tools themselves run in the program and
  are not cached. There is no MCP proxy yet.
- **Adapters**: DSPy and GEPA. Other frameworks run unchanged but as whole process runs, unless
  they mark units with `copula.run` or headers.
- **No replay or resume command**, and `copula.attach()` works only under `copula run`.
- **Other caches**: a framework cache that answers repeats from one stored sample hides those
  repeats from Copula. The DSPy adapter turns DSPy's off; turn off others (LangChain's
  `set_llm_cache`, LiteLLM caching, AG2's `cache_seed`, ...) yourself.
- **Calls that bypass the base URL** (a client with a hard-coded endpoint) are not served.
- **Threads**: unit context follows `ThreadPoolExecutor` workers and asyncio tasks; a thread
  started directly with `threading.Thread` carries no unit, so its calls belong to the process
  run.
- **Holdout runs**: case disjointness from the search phase is checked for units only.

## Security

- `.copula/` is private to its owner: it holds the secret behind run tokens, every prompt and
  every sample.
- The endpoint listens on 127.0.0.1 by default. With `--listen 0.0.0.0:PORT` it serves other
  hosts over plain HTTP, guarded by per-run tokens and the server token: use it on a trusted
  network only.
- Copula never forwards a program's key to the model server, and never hands the server's key to
  a program. `copula run` does not remove keys the program's environment already had.
- Stored Responses objects are served to any valid run token of the project
  (`GET /v1/responses/ID`).

## When not to use Copula

- **You need an exact replay of an earlier run**, for a regression test that compares outputs
  byte for byte. Use a record-and-replay tool ([Comparison](comparison.md#record-and-replay)).
- **Your program's behaviour depends on what is stored** (condition P).
- **Your calls rely on provider-side state** (server-side tools, provider-side conversations):
  Copula refuses them in this version.
