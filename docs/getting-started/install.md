# Install

Copula needs Python 3.9 or newer on Linux or macOS and nothing else: it uses only the standard
library. Read tracing of Node programs needs Node 18.19 or newer.

Copula is not on PyPI yet. Install it from a checkout of the repository:

```text
git clone <repository-url> copula
pip install ./copula
```

The extras install the clients that the tests and examples use: `pip install './copula[openai]'`,
`[anthropic]` or `[dspy]`. Copula does not need them itself.

Check the installation:

```console
$ copula --version
copula 0.1.0.dev0
```

`python -m copula` is the same command, for environments where the `copula` script is not on
`PATH`.

## What gets installed

- The `copula` command: `run`, `server`, `explain` and `ls` ([Command line](../reference/cli.md)).
- The `copula` Python package. `import copula` needs nothing beyond the standard library, and
  programs need it only to describe their structure in more detail than Copula infers
  ([Python API](../reference/python.md)).
- `copula.kernel`, the sample store and the cache manager underneath.

Copula keeps its state in a `.copula/` directory in the project, created on the first run
([Files and environment](../reference/files-and-env.md)).
