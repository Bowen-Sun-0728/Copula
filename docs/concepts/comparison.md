# Comparison with other tools

Several kinds of tools store model responses. They differ in what a repeated request gets back,
and so in what they are good for.

| | a repeat of a request gets | suits |
|---|---|---|
| [Response caches](#response-caches) | the one stored response | saving cost when repeats need not be independent |
| [Record and replay](#record-and-replay) | the recorded response, in recorded order | exact, offline tests |
| [Occurrence-numbered caches](#occurrence-numbered-caches) | the k-th stored response for the k-th repeat | sharing without collapsing repeats |
| [Durable execution](#durable-execution) | the result of the same step of the same execution | resuming one execution after a failure |
| Copula | the k-th sample of its group, the group decided by the run | edits, comparisons and replicates of one program |

## Response caches

Framework caches (DSPy's, LangChain's `set_llm_cache`, AdalFlow's, AG2's) and gateway caches
when they are turned on (LiteLLM, Portkey, OpenRouter) keep one response per request key. Unless
the program puts something into the key that tells repeats apart, every repeat of a request gets
that response: the five samples of a five-sample evaluation, and an unchanged rerun meant to
measure variability, all see one response. Some caches offer such a key: DSPy's `rollout_id`
(set by some modules, not by its evaluators and optimizers), AG2's `cache_seed` (one cache per
seed value). Helicone's cache buckets keep several responses per key and return one of them at
random, so repeats can still coincide.

Copula keeps the saving on calls that an edit did not change, but numbers repeats, so a run's
repeats and its replicates stay independent. Turn framework caches off under Copula (the DSPy
adapter does it for DSPy).

## Record and replay

vcr.py, pytest-recording and similar tools record the HTTP traffic of a test once and replay it
afterwards, in order, offline. A changed request is a miss: it fails or goes to the network,
depending on the record mode. They are the right tool for exact regression tests.

Copula does not replay an unchanged program: an unchanged rerun draws new samples. It shares
between *different* versions instead, so an edit pays for what changed and is compared with its
predecessor on the same samples.

## Occurrence-numbered caches

Cache Saver and Mnimi number identical requests within a namespace, so that the k-th repeat reads
the k-th stored sample and other namespaces reuse the same list. This is the rule Copula's store
uses, and the credit for it is theirs. DataDreamer (a seed per repeat) and Inspect (a cache per
epoch) number repeats in more specific forms.

What Copula adds around the rule:

- **Which runs share is decided for you**: groups follow from experiments, versions, slots and
  coordinate classes, so edits share, reruns and replicates do not, and the holdout phase is kept
  apart.
- **One store for every process of a project**, safe under concurrency and crashes: identical
  samples needed at the same time are generated once, and a sample is written only complete.
- **Unmodified programs**, through the base-URL variables, with read tracing for versions and
  adapters for DSPy and GEPA.
- **An explanation for every generated call.**

## Durable execution

Temporal, DBOS and Restate record the results of the steps of one execution, so that the
execution can resume after a failure without repeating its side effects. They replay within one
execution. Copula shares samples across executions, between versions of a program, and keeps
reruns and replicates independent.
