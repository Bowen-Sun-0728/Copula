# Copula

Copula lets reruns, edits and comparisons of an agent program share model samples, while each run
on its own still sees what it would have seen without Copula: fresh, independent draws.

Run an unmodified program under `copula run`. The calls that an edited program makes exactly as
before are read from earlier runs. The calls that changed are generated. Repeats that are meant to
be independent, such as an unchanged rerun, another seed or another replicate, draw new samples.

```text
$ copula run --exp demo -- python agent.py          # after editing one prompt
...
copula: run r-5e77ae7d7cf040a3 done: 6 calls: 3 reused, 3 generated, 0 joined
copula:   3 generated: new request -- closest earlier request (run r-74437c2a66b94f2e): messages[0] (system): '...eply OK, or give a corrected answer.' -> '...eply OK, or explain the mistake.'
```

## Why

Developing an agent means running it again and again with small changes, and every run pays for
every model call, although most of them are the calls of the previous run. A response cache saves
that cost but changes the statistics: it returns one stored response for every repeat of a
request, inside a run, across seeds and across reruns. Repeated evaluations stop being
independent, and the spread you measure between them shrinks to zero.

Copula numbers the repeats instead. Runs that share samples are in one **group**, and within a
group the k-th identical request of a run reads the group's k-th sample. A run never reads one
sample twice, so its own repeats stay independent. Which runs share a group follows from what you
are doing: two versions of a program share, while an unchanged rerun or a replicate takes a group
of its own.

## What you get

- **No changes to the program.** OpenAI Chat Completions, OpenAI Responses and Anthropic Messages
  clients work through their base-URL variables, streaming and tool calls included. DSPy and GEPA
  programs get adapters automatically.
- **You pay for what changed.** An edit regenerates only the calls whose requests changed, and
  everything downstream of them.
- **Sound by construction.** Each run alone sees samples distributed as without Copula. The
  conditions are stated in [Guarantees and limits](concepts/guarantees.md).
- **Every generated call has a reason.** The run summary and `copula explain` say why a call was
  not reused: what changed in the request, or which coordinate kept it apart.
- **Many runs at once.** One `copula server` per project serves concurrent runs, Slurm arrays and
  launchers on other hosts. When two runs need the same sample at the same moment, it is
  generated once.
- **Standard library only.** Python 3.9 or newer on Linux or macOS; vLLM, SGLang or any
  OpenAI-compatible API as the model server.

## When not to use it

- **You want an exact replay of an earlier run.** Copula gives an unchanged rerun new samples on
  purpose. Record-and-replay tools such as vcr.py replay exactly; Copula's own replay is not in
  this version.
- **What your program does depends on what is already stored**, for example a program that skips
  work it finds in the store. The guarantee assumes it does not.
- **Your calls rely on provider-side state**, such as server-side tools (web search) or
  conversations stored by the provider. Copula refuses such requests in this version.

## Start here

- [Install](getting-started/install.md), then the [Quickstart](getting-started/quickstart.md):
  five minutes, no GPU, no API key.
- [How Copula works](concepts/how-it-works.md): versions, slots and groups in one page.
- [Command line](reference/cli.md): every command and option.
