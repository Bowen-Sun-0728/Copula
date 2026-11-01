# One server for many runs

`copula run` serves its run itself when no server is running: an **embedded server** that lives
as long as the command. The sample store of a project can be open in one process only, so for
several commands at once (replicates, sweeps, a Slurm array, two terminals) start one
`copula server` for the project. Every `copula run` in the project finds it and is served by it.

```text
$ copula server &
copula: server http://127.0.0.1:42921 for /home/me/project (vllm http://gpu-node:8000/v1, engine seed per address)
{"event": "ready", "url": "http://127.0.0.1:42921", "pid": 2498623}
```

The server takes the [model server options](../reference/cli.md#model-server-options) (`--upstream`,
`--model-dir`, `--mock`, ...) and writes `.copula/server.json`, through which runs find it. It
stops on Ctrl-C or SIGTERM and removes the file; a file left behind by a server that died on this
host is removed by the next command.
One server per project: a second `copula server` refuses to start.

<!-- docs-test: bg copula server --mock -->
<!-- docs-test: wait .copula/server.json -->

```python title="agent.py"
from openai import OpenAI

client = OpenAI()
for q in ["Why is the sky blue?", "Who wrote Hamlet?"]:
    r = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct", temperature=0.7,
        messages=[{"role": "user", "content": q}])
    print(r.choices[0].message.content)
```

With the server running (here started with `--mock`), runs name no model options:

```console
$ copula run --exp demo --set rep=1 -- python agent.py
copula: run r-e6ea9df2ea1a4875  exp=demo  dev: version v-2c6ebe6a89704473, slot 1  group g-a174ef6155973219d0d1a48f
copula: served by server http://127.0.0.1:38687 (mock model)
copula: coordinates: {"replicate": ["rep"]}
copula: new version (compared with earlier versions of demo)
copula: attach: attached to http://127.0.0.1:38687/ (threads copy context; headers on httpx, httpx2, http.client)
mock:4ce416d39ab6bf39ff4c20bc
mock:4670b5a6cfb51fc28ed025df
copula: run r-e6ea9df2ea1a4875 done: 2 calls: 0 reused, 2 generated, 0 joined
copula:   2 generated: new request -- no earlier run of the experiment called this model
copula:   details: copula explain r-e6ea9df2ea1a4875
```

The model options belong to the server, and `copula run` refuses them rather than ignore them:

<!-- docs-test: fails -->
```console
$ copula run --exp demo --model-dir /models/qwen -- python agent.py
copula run: the Copula server http://127.0.0.1:38687 decides the model server, so --model-dir cannot be set here: drop it, or stop the server
```

## Concurrent runs

Runs served by one server may run at the same time. When two of them need the same sample at the
same moment (same group, same request, same occurrence), it is generated once: the second waits
for it and reads it, and its summary counts it as **joined**.

<!-- docs-test: skip -->
```console
$ for seed in 1 2 3 4; do copula run --exp sweep -- python agent.py --seed $seed & done; wait
```

Runs registered at the same moment see each other: two launches of one unchanged program take
different slots, so they draw different samples.

## Slurm and other hosts

By default the server listens on 127.0.0.1 and serves runs on its own host. To serve launchers on
other hosts, listen on a reachable address and give the launchers the URL and the server token:

<!-- docs-test: skip -->
```console
$ copula server --listen 0.0.0.0:7000 --upstream http://gpu-node:8000/v1 &
$ export COPULA_SERVER_TOKEN=$(python -c 'import json; print(json.load(open(".copula/server.json"))["token"])')
$ sbatch --array=0-9 --export=ALL --wrap 'copula run --exp sweep --server http://login1:7000 -- python agent.py'
```

- Each array task is a replicate: `SLURM_ARRAY_TASK_ID` becomes the coordinate
  `slurm_array_task_id`.
- The launcher on the other host computes the content digests of the files that decide the
  version itself, so the project should be at the same path there (a shared file system). A file
  it cannot read counts as unchanged, which makes the run a rerun: new samples, never a wrong
  replay.
- The token is a secret: with it, anyone who can reach the port can register runs and read
  stored samples. Traffic is plain HTTP, so listen on a trusted network only.

## What the server holds

Everything lives in the project's `.copula/` directory, private to its owner: the sample store,
the registry of runs, the call log that `copula explain` reads, and `server.json`
([Files and environment](../reference/files-and-env.md)). The server is the only writer of the
registry; `copula explain` and `copula ls` read it while the server runs.
