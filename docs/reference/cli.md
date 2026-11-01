# Command line

```text
copula [--version] {run, server, explain, ls} ...
```

`copula --version` prints the version; `copula COMMAND --help` lists a command's options.
`python -m copula` is the same command.

## `copula run`

```text
copula run [options] [model server options] -- COMMAND [ARGS...]
```

Runs `COMMAND` as one run of the project and serves its model calls: through the project's
`copula server` when one is running (found in `.copula/server.json`), through the server named by
`--server`, or else through a server embedded in this command. Exits with the command's status.

| option | default | meaning |
|---|---|---|
| `--exp NAME` | none | the experiment. Runs of one experiment share samples; a run without one shares nothing. With `--exp`, the files the program reads decide its version, and Python children are attached |
| `--project DIR` | `.` | the project: where `.copula/` lives, and the only files that count for versions |
| `--recipe NAME` | `dev` | default classes for coordinate names of a kind of workload: `dev`, `eval`, `optimize`, `ab`, `sweep`, `tts`, `sdg`, `rl` ([recipes](coordinates.md#recipes)) |
| `--phase PHASE` | `search` | `search` or `holdout`; the phases never share ([phases](../concepts/runs-and-coordinates.md#phases)) |
| `--set K=V` | | a coordinate of the run (repeatable) |
| `--compare NAME` | | class coordinate `NAME` as compare: runs that differ in it share (repeatable) |
| `--replicate NAME` | | class it as replicate: runs that differ in it never share |
| `--case NAME` | | class it as case: runs that differ in it never share |
| `--annotation NAME` | | class it as annotation: recorded, ignored |
| `--route ROLE=RULE` | `run` | the group of calls made with role `ROLE`: `run`, `global`, `fresh` or `per:COORD` ([routes](../guides/optimizers.md#routes-for-roles)) |
| `--track GLOB` | | project files whose content belongs to the version, for programs Copula does not trace (repeatable; `**` allowed) |
| `--no-attach` | | do not attach Python children: no adapters, no unit context |
| `--server URL` | | the URL of a `copula server` on another host; the token comes from `$COPULA_SERVER_TOKEN` |

The model server options below apply when this command serves the run itself. With a live
server they belong to the server, and `copula run` refuses them.

The run prints its placement before the command starts and a summary when it ends, on stderr
([Messages](messages.md)).

## `copula server`

```text
copula server [--listen HOST[:PORT]] [model server options]
```

Serves every run of the project from one process until Ctrl-C or SIGTERM
([One server for many runs](../guides/project-server.md)). Writes `.copula/server.json`, prints
a line on stderr and one JSON line `{"event": "ready", "url": ..., "pid": ...}` on stdout when it
is ready. Refuses to start if a server already runs for the project.

| option | default | meaning |
|---|---|---|
| `--project DIR` | `.` | the project |
| `--listen HOST[:PORT]` | `127.0.0.1`, any free port | the address to serve on; `0.0.0.0:PORT` serves other hosts |

## Model server options

Options of `copula run` (embedded server) and `copula server`
([Connect a model server](../guides/model-servers.md)).

| option | default | meaning |
|---|---|---|
| `--upstream URL` | `$OPENAI_BASE_URL` | the OpenAI-compatible model server to forward to |
| `--served-model NAME` | the first listed | the default model of the upstream; needed when it lists no models |
| `--model-dir DIR` | | the files of the default model: they establish its identity and give its sampling defaults |
| `--defaults JSON` | | a pinned sampling table for the default model, as JSON or `@FILE`, naming all nine sampling parameters; overrides `generation_config.json` |
| `--identity-extra JSON` | | serving settings that change outputs (dtype, quantisation), as JSON or `@FILE`; vLLM and SGLang only |
| `--engine-seed MODE` | `auto` | `address`: an engine seed per sample address; `off`: none; `auto`: `address` for vLLM, `off` otherwise |
| `--api-key-env VAR` | `OPENAI_API_KEY` | the variable holding the upstream's key |
| `--model NAME[=SERVED]` | | a client-side model name for a served model (repeatable); `'*'` takes any name |
| `--max-tokens N` | 1024 | `max_tokens` of requests that set none |
| `--max-attempts N` | 6 | attempts of a generation that fails with a transport error, a timeout, 5xx, 408, 409, 429 or 401/403/404/405/407 |
| `--backoff S` | 0.5 | base of the exponential retry backoff, seconds |
| `--timeout S` | 900 | upstream request timeout, seconds |
| `--workers N` | 64 | upstream requests in flight |
| `--inflight N` | 64 | generations in flight per model |
| `--wait-ready S` | 0 | wait up to S seconds for the upstream to answer |
| `--keepalive S` | 15 | seconds between keep-alives of a buffered stream |
| `--no-identity` | | treat every model's identity as not established: greedy calls are cached per group |
| `--mock` | | serve the mock model instead of an upstream |
| `--mock-script FILE.py:FUNC` | | script the mock's replies (`MODULE:FUNC` works too); implies `--mock` |
| `--mock-nonce TEXT` | `copula-mock` | another nonce gives the mock other replies |
| `--mock-latency S` | 0 | the mock's delay per reply, seconds |
| `--manager SOCKET` | | serve from a running kernel manager instead (`python -m copula.kernel.manager`); its model is named `actor` unless `--model` names it, and its identity counts as established |
| `--trace` | | write the kernel's trace to `.copula/trace.jsonl` |

## `copula explain`

```text
copula explain [RUN] [--calls] [--json]
```

How a run was placed (experiment, phase, version, slot, group, coordinates, the run it reruns)
and why each of its calls was generated
([Why was this call generated?](../guides/why-generated.md)).

| option | default | meaning |
|---|---|---|
| `RUN` | `last` | a run id, a unique prefix of one, a unit id (its process run is shown), `last` or `last~N` (the N-th run before the latest) |
| `--project DIR` | `.` | the project |
| `--calls` | | one line per call |
| `--json` | | JSON lines: one object per run and unit; with `--calls`, one per call too |

## `copula ls`

```text
copula ls [--exp NAME]
```

Lists the runs and units of the project, oldest first: id, experiment, phase, kind (process run or
unit), status of process runs (`running`, `done`, or `unknown` when the command failed or its
launcher died), version, slot and group.

| option | default | meaning |
|---|---|---|
| `--project DIR` | `.` | the project |
| `--exp NAME` | all | only the runs of this experiment |
