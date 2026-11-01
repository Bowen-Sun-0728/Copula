# Copula

Copula lets reruns, edits and comparisons of an agent program share model samples, while each run
on its own still sees what it would have seen without Copula: fresh, independent draws.

Run an unmodified program under `copula run`. The calls that an edited program makes exactly as
before are read from earlier runs; the calls that changed are generated; repeats that are meant
to be independent (an unchanged rerun, another seed, another replicate) draw new samples.

```text
$ copula run --exp demo -- python agent.py          # after editing one prompt
...
copula: run r-5e77ae7d7cf040a3 done: 6 calls: 3 reused, 3 generated, 0 joined
copula:   3 generated: new request -- closest earlier request (run r-74437c2a66b94f2e): messages[0] (system): '...eply OK, or give a corrected answer.' -> '...eply OK, or explain the mistake.'
```

It is the system of the paper *Copula: Sound Sharing of Stochastic Nodes*: `copula.kernel` is the
paper's code, byte for byte (`tests/test_provenance.py` checks it).

## Install

Python 3.9 or newer on Linux or macOS; standard library only. Copula is not on PyPI yet:

```text
pip install .            # from a checkout of this repository
copula --version
```

## Quick start

Your program talks to vLLM, SGLang or an OpenAI-compatible API through `OPENAI_BASE_URL`. Run it
under Copula; `--mock` serves a test model, so you can try it without a model server:

```text
copula run --exp demo --mock -- python agent.py     # try it
copula run --exp demo -- python agent.py            # forwards to $OPENAI_BASE_URL
copula explain                                      # why the latest run's calls were generated
```

- OpenAI Chat Completions, OpenAI Responses and Anthropic Messages clients work unchanged,
  streaming and tool calls included; DSPy and GEPA programs get adapters automatically.
- With `--exp`, the files the program reads decide its version: an edited program shares with
  earlier versions, an unchanged rerun draws new samples.
- Coordinates from the command line decide what is compared and what is replicated:
  `--variant b` shares with `--variant a`, `--seed 2` shares with nothing of `--seed 1`.
- For several runs at once, start `copula server` in the project; every `copula run` finds it.

The [Quickstart](docs/getting-started/quickstart.md) takes five minutes and needs no GPU.

## Documentation

- Get started: [Install](docs/getting-started/install.md), [Quickstart](docs/getting-started/quickstart.md)
- Guides: [Run an existing program](docs/guides/existing-programs.md),
  [Connect a model server](docs/guides/model-servers.md),
  [Compare, repeat and sweep](docs/guides/compare-and-repeat.md),
  [Prompt and harness optimizers](docs/guides/optimizers.md),
  [One server for many runs](docs/guides/project-server.md),
  [Why was this call generated?](docs/guides/why-generated.md)
- Concepts: [How Copula works](docs/concepts/how-it-works.md),
  [Runs, coordinates and versions](docs/concepts/runs-and-coordinates.md),
  [When two calls are the same](docs/concepts/identical-calls.md),
  [Guarantees and limits](docs/concepts/guarantees.md),
  [Comparison with other tools](docs/concepts/comparison.md)
- Reference: [Command line](docs/reference/cli.md),
  [Coordinate names and recipes](docs/reference/coordinates.md),
  [Python API](docs/reference/python.md), [HTTP interface](docs/reference/http.md),
  [Files and environment](docs/reference/files-and-env.md), [Messages](docs/reference/messages.md)

`mkdocs serve` renders them as a site ([Development](docs/project/development.md)).

## What is guaranteed

Every run, alone, sees samples distributed as without Copula: its i-th repeat of a request is a
fresh draw it has not seen before. This holds when (P) the program and its cases do not depend on
what is in the store, (C1) no sample address is read twice and (C2) the next address never depends
on a sample the run has not read. Copula enforces C1 and C2; P is yours. Where outputs can differ
from native runs, and the limits of this version, are listed in
[Guarantees and limits](docs/concepts/guarantees.md).

## Development

```text
PYTHONPATH=src:. python -m unittest discover -s tests -t .
```

`tests/kernel` holds the paper's kernel tests and `tests/test_docs.py` runs the documentation's
examples. Tests that need openai, anthropic, DSPy or Node are skipped when they are absent.
