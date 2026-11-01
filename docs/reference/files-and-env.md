# Files and environment

## The `.copula/` directory

Copula keeps everything of a project in `.copula/` in the project directory, created on the first
run with permissions 0700. Deleting it forgets every sample and run of the project.

| path | content |
|---|---|
| `entries.db` | the sample store (SQLite), with the secret that engine seeds are derived with |
| `registry.db` | runs and units: versions, slots, groups, coordinates, read sets; the secret behind run tokens; the Responses objects Copula issued |
| `calls.db` | the call log: one row per answered call and each distinct request body, read by `copula explain` |
| `server.json` | the live server: URL, host, pid, token and upstream (0600; removed when the server stops) |
| `runs/<run>/reads.<pid>` | the files each process of a run read and wrote |
| `pylib/` | a link to the `copula` package, put on the children's `PYTHONPATH` |
| `m.sock` | the kernel manager's socket while a server runs (in a temporary directory if the path is too long) |
| `trace.jsonl` | the kernel trace, with `--trace` |

The call log is diagnostic only: nothing reads it to decide what a request returns.

## Variables set for the command

`copula run` sets these in the command's environment:

| variable | value |
|---|---|
| `OPENAI_BASE_URL`, `OPENAI_API_BASE`, `LLM_BASE_URL`, `COPULA_URL` | the run's URL, `http://HOST:PORT/r/TOKEN/v1` |
| `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_BASE` | the run's URL without `/v1` |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | `copula`, when not set already (the key is not used) |
| `COPULA_RUN` | the run id |
| `COPULA_EXP` | the experiment, or empty |
| `COPULA_MODEL` | the model server's default model |
| `COPULA_PROJECT` | the project directory |
| `COPULA_READS_FILE` | where Python and Node children record the files they read |
| `COPULA_ATTACH` | `1` with `--exp` (and without `--no-attach`): Python children attach |
| `PYTHONPATH` | Copula's start-up hook and `.copula/pylib`, before the existing value |
| `NODE_OPTIONS` | `--require` of Copula's read recorder, after the existing value (with `--exp`) |

`COPULA_SERVER_TOKEN` is removed from the command's environment.

## Variables Copula reads

| variable | used by | meaning |
|---|---|---|
| `OPENAI_BASE_URL` | `copula run`, `copula server` | the upstream, unless `--upstream` |
| `OPENAI_API_KEY` | `copula run`, `copula server` | the upstream's key (or the variable `--api-key-env` names) |
| `COPULA_SERVER_TOKEN` | `copula run --server` | the token of a server on another host |
| `SLURM_ARRAY_TASK_ID` | `copula run` | becomes the coordinate `slurm_array_task_id` |
| `COPULA_QUIET` | attached Python children | silences the `copula: attach:` lines |
| `HF_HUB_CACHE`, `HF_HOME` | the server | where to look for a model's Hugging Face snapshot |
