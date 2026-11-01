# Connect a model server

Copula sits between the program and an OpenAI-compatible model server, the **upstream**. By
default the upstream is the server the program would have used: the `OPENAI_BASE_URL` of the
`copula run` or `copula server` process. `--upstream URL` names another one.

<!-- docs-test: skip -->
```console
$ export OPENAI_BASE_URL=http://gpu-node:8000/v1        # what the program used before
$ copula run --exp triage -- python agent.py
```

At start Copula asks the upstream for its models (`GET /v1/models`) and tells vLLM and SGLang,
which it runs differently, from other OpenAI-compatible APIs. The banner says what it found:

```text
copula: served by embedded server (vllm http://gpu-node:8000/v1, engine seed per address)
copula:   note: sampling defaults: Qwen/Qwen2.5-7B-Instruct generation_config
```

Model options belong to whoever serves the run: `copula run` when it serves the run itself, or
`copula server` for every run of the project ([One server for many runs](project-server.md)).

## vLLM and SGLang

**Sampling defaults.** A request that leaves a sampling parameter unset gets the server's default,
and two requests that differ only in stating that default must get the same sample. So Copula
sends every sampling parameter explicitly, from a pinned table:

1. `--defaults JSON` or `--defaults @FILE`, if given, for the default model; it names all nine
   sampling parameters (`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
   `frequency_penalty`, `repetition_penalty`, `stop`, `chat_template_kwargs`);
2. otherwise the model's `generation_config.json` (`temperature`, `top_p`, `top_k`, `min_p`,
   `repetition_penalty`), which is what vLLM applies by default, so the samples are those of the
   server used natively;
3. otherwise neutral values (temperature 1, top_p 1, no top_k), with a note saying so.

Copula finds `generation_config.json` in `--model-dir DIR`, or in the directory the server reports
for the model (when the server runs on this host), or in this host's Hugging Face cache for the
model id. A LoRA adapter takes the table of its base model.

**Model identity.** Copula stores samples per model revision, so a sample of one checkpoint is
never served for another. The revision covers the served model name, the server version, the
sampling table and, when it can read them, the model's files: `config.json`, the tokenizer files
and every weight file, so a re-saved or re-sharded checkpoint is a new revision.
`--identity-extra JSON` adds serving settings that change outputs, such as dtype or quantisation.

The identity is **established** only from files the server itself loaded: `--model-dir`, or the
local directory the server reports when it runs on this host. A Hugging Face id, or a server on
another host, gives the sampling table from the local cache but no established identity, and a
note says so:

```text
copula:   note: identity of Qwen/Qwen2.5-7B-Instruct not established (model files from a cache snapshot or another host): greedy calls are cached per group (W12); pass --model-dir to establish it
```

The identity matters for greedy calls (temperature 0 or top_k 1): with an established identity
they are deterministic and cached once for every run; without one they are cached per group like
any sampled call ([When two calls are the same](../concepts/identical-calls.md)).

**Engine seed.** Copula sends vLLM a seed per sample, derived from the sample's address with a
secret of the store (`--engine-seed address`, the default for vLLM under `auto`). A sample does not
depend on the server's `--seed`, and the same address asks for the same draw again. SGLang gets no
engine seed under `auto`; `--engine-seed off` turns it off for vLLM too. The seed a client puts in
its request is never sent upstream ([Seeds](../concepts/identical-calls.md#seeds)).

## Hosted and other OpenAI-compatible APIs

<!-- docs-test: skip -->
```console
$ export OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=sk-...
$ copula run --exp triage -- python agent.py
```

- **Key.** Copula sends the key in its own `OPENAI_API_KEY` (or in the variable `--api-key-env`
  names). It never forwards a key sent by the program.
- **Parameters.** `temperature`, `top_p` and the penalties are always sent, with OpenAI's defaults
  when the client sets none; pass `--defaults` for a provider whose defaults differ. `top_k`,
  `min_p`, `repetition_penalty` and `chat_template_kwargs` are sent only when they differ from
  neutral values. `max_tokens` becomes `max_completion_tokens` for models that require it.
- **Identity.** An API model's weights are not visible, so its identity is never established:
  greedy calls are cached per group. There is no engine seed, and `--identity-extra` does not
  apply.

## Names, limits and failures

- **Model names.** Copula serves the names the upstream lists. `--model NAME=SERVED` adds a
  client-side name for a served model; `--model '*'` sends any unknown name to the default model;
  `--served-model NAME` picks the default model (it is also the only model when the upstream does
  not list its models).
- **`max_tokens`.** Requests that set none get `--max-tokens` (1024), so the sample does not
  depend on a server-side limit.
- **Failures.** Connection errors and timeouts, HTTP 5xx, 408, 409, 429 and misconfiguration
  (401, 403, 404, 405, 407) are retried with backoff (`--max-attempts`, `--backoff`) and never
  stored; when the attempts run out the client gets HTTP 502 and a later request tries again. Any
  other 4xx is caused by the request itself (a context that is too long, a malformed tool
  schema): it is stored as that sample's result, as the model server would answer it every time,
  and the client gets HTTP 400 `model_error`.
- **Timeouts and concurrency.** `--timeout` (900 s per upstream request), `--workers` (64
  upstream requests at once), `--inflight` (64 generations per model), `--wait-ready S` to wait
  for an upstream that is still starting.

## The mock model

`--mock` serves a model that needs no server: each reply is a token that is unique to the sample
(`mock:...`). The token depends only on the sample's address, so an experiment gives the same
tokens in every project (a run without `--exp` has a fresh group, and so fresh tokens). It accepts any model
name unless `--model` lists names. Use it to try Copula, and to test a program's control flow.

`--mock-script FILE.py:FUNC` makes the replies depend on the request. The function receives the
request body and the sample's token and returns text, a dict of sample fields (`content`,
`tool_calls`, `finish_reason`), or `None` for the plain token:

```python title="replies.py"
def reply(body, token):
    question = body["messages"][-1]["content"]
    if "capital" in question:
        return "Paris"            # the same text for every sample of this request
    return f"I am not sure ({token})"
```

```python title="ask_capital.py"
from openai import OpenAI

client = OpenAI()
for q in ["What is the capital of France?", "What is love?"]:
    r = client.chat.completions.create(model="any", messages=[{"role": "user", "content": q}])
    print(r.choices[0].message.content)
```

```console
$ copula run --mock --mock-script replies.py:reply -- python ask_capital.py
copula: run r-b4e2f7a08fe84b3a  exp=-  no experiment: fresh group (O1)  group g-fdecf26b0001459486e3
copula: served by embedded server (mock model)
Paris
I am not sure (mock:eddef53502ac7ab118187002)
copula: run r-b4e2f7a08fe84b3a done: 2 calls: 0 reused, 2 generated, 0 joined
copula:   2 generated: new request -- no earlier run sent it
copula:   details: copula explain r-b4e2f7a08fe84b3a
```

`--mock-script` implies `--mock`. `--mock-nonce` gives the mock other tokens, and
`--mock-latency S` delays each reply.
