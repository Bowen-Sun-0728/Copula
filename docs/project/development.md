# Development

## Layout

```text
src/copula/
  launcher.py      the copula command: run, server, explain, ls
  server.py        the Copula server: store, kernel manager, proxy, discovery
  proxy.py         the HTTP endpoint: model protocols, control API, units, retries
  protocols.py     Chat Completions, Responses and Anthropic Messages, to and from chat form
  registry.py      runs and units: versions, slots, groups, decisions, holdout checks
  coords.py        coordinate classes, recipes, command-line parsing
  attrs.py         how a request is classified (sampled, greedy, uncached)
  upstream.py      model servers: vLLM, SGLang, APIs; sampling tables, identity, engine seeds
  calls.py         the call log and the reasons of copula explain
  client.py        copula.run, holdout, decision, role, headers, chat
  _attach.py       copula.attach: threads, HTTP headers, adapters
  adapters/        DSPy and GEPA
  _shim/           the start-up hook (Python read tracing) and the Node read recorder
  testing.py       the scripted mock model
  kernel/          the paper's kernel: SDK, cache manager, store, keys (byte for byte)
tests/             unit and end-to-end tests; tests/kernel holds the kernel's own tests
docs/              this documentation (MkDocs)
tools/             import_kernel.py
```

## The kernel

`copula.kernel` is a byte-for-byte copy of the kernel the paper evaluated.
`kernel/PROVENANCE.json` records the sha256 of every file, and `tests/test_provenance.py` fails if
any file differs. Changes to sharing semantics belong in the layers around the kernel. To bring in
a new frozen kernel:

```text
python tools/import_kernel.py SRC_DIR --tests TESTS_DIR
```

It copies the kernel files, rewrites `PROVENANCE.json`, and ports the kernel's tests into
`tests/kernel` (imports rewritten, static-graph tests skipped).

The comments of the kernel and of `tests/kernel` call it by its research name, `evalrt`, and cite
design notes of the research repository (`reports/...`), which are not published. The research
code's static-graph executor stays behind, so the `Session` methods for static graphs (`program`,
`graph_bound`, `execute`, `execute_sync`, `forecast`) raise `ImportError` in this package.

## Tests

```text
PYTHONPATH=src:. python -m unittest discover -s tests -t .
```

The suite needs only the standard library and runs against the mock model. Tests that need
`openai`, `anthropic`, DSPy or Node are skipped when they are missing; `COPULA_DSPY_PYTHON` and
`COPULA_ANTHROPIC_PYTHON` name interpreters that have them.

## Documentation

The pages are Markdown in `docs/`, built with MkDocs and the Material theme:

```text
pip install '.[docs]'   # MkDocs 1.6 and Material 9, pinned in pyproject.toml
mkdocs serve            # http://127.0.0.1:8000
mkdocs build --strict   # what CI runs
```

`tests/test_docs.py` keeps the pages true:

- Each page is run as a script in a fresh project: a code block with a `title="FILE"` writes that
  file, and each `$ ` command of a `console` block is run, its output compared with the lines
  below it. Ids that change from run to run (runs, versions, groups, ports, mock replies) are
  compared by pattern. HTML comments `<!-- docs-test: ... -->` add hidden steps (`run`, `bg`,
  `wait`) or mark the next block (`skip` for a real model server, `fails` for a command that
  must fail).
- The command-line reference must list exactly the parser's options, the coordinate tables must
  match `coords.py`, every `copula explain` reason must have a section, every page must be in the
  navigation, and every link must resolve.

After a change to the output, regenerate the transcripts and review the diff:

```text
COPULA_DOCS_UPDATE=1 PYTHONPATH=src:. python -m unittest tests.test_docs
```

The examples need the `openai` package (`COPULA_DOCS_PYTHON` names an interpreter that has it).
