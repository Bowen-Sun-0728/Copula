# Run an existing program

`copula run` starts a command as one **run** and serves its model calls:

```text
copula run [--exp NAME] [options] -- COMMAND [ARGS...]
```

Everything after `--` is the command, unchanged. `copula run` exits with the command's exit
status (128 + N when signal N killed it). Ctrl-C in a terminal reaches the command directly;
`kill` of the launcher (SIGTERM) is forwarded to it.

## What changes for the program

The command runs with its model endpoint pointed at Copula. These variables are set in its
environment; the full list is in [Files and environment](../reference/files-and-env.md).

| variable | value |
|---|---|
| `OPENAI_BASE_URL`, `OPENAI_API_BASE`, `LLM_BASE_URL`, `COPULA_URL` | the run's URL, `http://127.0.0.1:PORT/r/TOKEN/v1` |
| `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_BASE` | the same URL without `/v1` (the Anthropic SDK adds it) |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | `copula`, if they are not set already |
| `COPULA_MODEL` | the default model of the model server |
| `PYTHONPATH` | Copula's start-up hook first: it records the files Python reads |
| `NODE_OPTIONS` | `--require` of Copula's read recorder (with `--exp`) |

The token in the URL names the run, so any client that takes a base URL works without sending
extra headers. Copula never forwards the key the program sends: it talks to the model server
with its own ([Connect a model server](model-servers.md)). The rest of the environment is left
as it was, so a key that was already set is still visible to the program.

Clients tested with Copula: the `openai` and `anthropic` Python SDKs and DSPy 3 (through LiteLLM).
Any other client that reads one of these variables works the same way. A program that hard-codes
its base URL needs one change: take it from `OPENAI_BASE_URL`.

## Without an experiment

Without `--exp` the run is served, logged and explained, but it shares nothing with any other run.
Here a shell script calls the endpoint with curl:

```sh title="ask.sh"
curl -s "$OPENAI_BASE_URL/chat/completions" -H "Content-Type: application/json" \
  -d '{"model": "m", "messages": [{"role": "user", "content": "hi"}]}' |
  python -c 'import json, sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

```console
$ copula run --mock -- sh ask.sh
copula: run r-7b008e906a924526  exp=-  no experiment: fresh group (O1)  group g-ae4de1fb42574fb3bfd7
copula: served by embedded server (mock model)
mock:9e2429903d29e0a65fc0379f
copula: run r-7b008e906a924526 done: 1 calls: 0 reused, 1 generated, 0 joined
copula:   1 generated: new request -- no earlier run sent it
copula:   details: copula explain r-7b008e906a924526
```

With `--exp NAME`, runs of the experiment share: an edited program reads the samples of the calls
it makes as before ([Quickstart](../getting-started/quickstart.md)).

## Protocols

| endpoint | protocol |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions: tools, `response_format`, `n`, `seed`, streaming |
| `POST /v1/responses`, `GET /v1/responses/ID` | OpenAI Responses, including `previous_response_id` chains of responses Copula issued |
| `POST /v1/messages` | Anthropic Messages, including tools and streaming |
| `GET /v1/models` | the models the server serves |

Each protocol is translated into a chat request for the OpenAI-compatible model server, and the
sample is rendered back in the protocol the client used. A streaming and a non-streaming request
with the same content are the same request. Streams are buffered: the client receives keep-alives
while the sample is generated, then the whole response as stream events.

What is refused with HTTP 400 in this version:

- content parts other than text (images, audio, files);
- requests that rely on provider-side state (`previous_response_id` of a response Copula did not
  issue, `conversation`, `thread_id`) or use server-side tools such as web search;
- every request field Copula does not know, for example `logprobs`, `logit_bias`, `best_of`,
  `reasoning_effort`, or vLLM's `min_tokens` and `guided_json`, also inside `extra_body`.

Fields that do not change the output (`user`, `metadata`, `store`, `service_tier`, `timeout`,
...) are accepted and ignored ([When two calls are the same](../concepts/identical-calls.md)).

## Versions: the files the program reads

With `--exp`, Copula decides whether a run is a new version of the program or an unchanged rerun
from the content of the project files the program read
([Runs, coordinates and versions](../concepts/runs-and-coordinates.md#versions)).

- **Python** children (every Python process the command starts, at any depth) are traced by an
  audit hook: files opened for reading and the source files of imported modules.
- **Node** children are traced by a preloaded module (`require`, `import` and `fs` reads).
- **Other programs**: name the files with `--track GLOB` (repeatable, relative to the project,
  `**` allowed), for example `--track 'prompts/**' --track config.yaml`.

Only files under the project directory count (`--project`, by default the current directory).
Files the run also wrote, `.copula/`, virtual environments (`venv`, `.venv`, `env`,
`site-packages`), `node_modules`, caches and VCS directories are ignored, and so are `build/` and
`dist/` for Node.

!!! warning "A program Copula cannot trace looks unchanged forever"
    If Copula records no file reads for a run, a later run finds nothing that changed and counts
    as an unchanged rerun: it draws new samples, and edits never share. This happens for programs
    that are neither Python nor Node (a compiled binary, a shell script that does the work
    itself), and for Python started with `-I`, `-E` or `-S`, which skip Copula's start-up hook.
    Name their files with `--track`. Python and Node programs started *by* a shell script are
    traced.

## Inside a Python program

With `--exp`, Python children are **attached** to their run (`copula.attach()`), which prints one
line:

```text
copula: attach: attached to http://127.0.0.1:35135/ (threads copy context; headers on httpx, httpx2, http.client)
```

Attaching copies Copula's context into worker threads and adds it to requests sent to Copula, so
that `copula.run(...)` and the framework adapters can mark units ([Python API](../reference/python.md)).
It changes nothing else in the program. `COPULA_QUIET=1` silences the line; `--no-attach` turns
attaching off.

## Other languages

A program in any language can mark units itself by sending `X-Copula-*` headers with its
requests ([HTTP interface](../reference/http.md)). Without them, the whole command is one run.
