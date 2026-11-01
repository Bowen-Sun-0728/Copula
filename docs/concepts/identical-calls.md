# When two calls are the same

Two calls can share a sample only if they have the same **address**. This page lists everything
that goes into it, and what does not.

## The address of a call

| call | address |
|---|---|
| sampled (the default) | group, model revision, request, occurrence |
| greedy, model identity established | model revision, request |
| uncached | none: drawn anew every time |

- **Group**: decided by the run, its coordinates and slot
  ([Runs, coordinates and versions](runs-and-coordinates.md#groups-and-slots)), a role's route,
  and the request's `seed`.
- **Model revision**: the served model, the model server's kind and version, the sampling table,
  serving settings (`--identity-extra`), and the model's files when Copula can read them: config,
  tokenizer and every weight file ([Connect a model server](../guides/model-servers.md)).
- **Request**: the request after canonicalisation (below).
- **Occurrence**: for the k-th identical request of a run in its group, k. A request with `n=3`
  takes three consecutive occurrences.

## The request

The request in the address is the chat request after canonicalisation:

- **Messages**, including tool calls and tool results. Content sent as a list of text parts is
  the texts joined by newlines (as vLLM joins them), so it is the same request as that string, and
  the role `developer` is read as `system`. Content parts other than text (images, audio, files)
  are refused in this version.
- **Every sampling parameter**: `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
  `frequency_penalty`, `repetition_penalty`, `stop`, `chat_template_kwargs`. Parameters the client
  left out are filled from the model's pinned sampling table, so a request that states a default
  and one that leaves it out are the same request.
- **`max_tokens`** (or `max_completion_tokens`), set to `--max-tokens` when the client sets none.
- **`tools`, `tool_choice`, `response_format` and `parallel_tool_calls`** when they are set.

Not part of the request, because they do not change the sample: `stream` and `stream_options`;
`user`, `metadata`, `store`, `service_tier`, `prompt_cache_key`, `safety_identifier` and
`timeout`; client headers. `n` and `seed` are not part of it either, but they decide the
occurrences and the group.

A change anywhere in the request makes it a new request, including a change you did not mean: a
timestamp in the system prompt, a random id, the iteration order of a set. `copula explain` shows
the first difference ([`new_request`](../guides/why-generated.md#new_request)).

## How a call is classified

Copula infers how to treat a call from the request, as the table says. The first matching row
applies.

| the request | treated as |
|---|---|
| refers to provider-side state: `previous_response_id` of a response Copula did not issue, `conversation`, `thread_id` | refused: HTTP 400 (404 for an unknown `previous_response_id` in the Responses API) |
| uses a server-side tool (a tool whose type is not `function`, such as web search) | refused: HTTP 400 |
| comes with a role routed `fresh` | uncached |
| sets a `seed` | sampled, in the seed's own stream |
| temperature 0 or top_k 1, and the model's identity is established | greedy: one answer for all runs, across groups, phases and experiments |
| temperature 0 or top_k 1, identity not established | sampled, per group (W12) |
| anything else | sampled |

Greedy decoding on an inference server is close to, but not exactly, deterministic, so Copula
treats a greedy call as deterministic only when it knows exactly which model answers it. A
Responses request whose `previous_response_id` names a response Copula issued is expanded into
the full conversation and treated like any other request.

## Seeds

A `seed` in the request body selects a sample stream: the group of a seeded request is derived
from the run's group and the seed. Requests with `seed=7` share samples with other `seed=7`
requests of the group, and the k-th identical seeded request of a run reads the k-th sample of the
seed's stream. A request with `seed=7` and `n=3` uses the streams 7, 8 and 9.

Copula does not send the client's seed to the model server. For vLLM it sends an engine seed of
its own, derived from the sample's address, so the draw does not depend on the client's seed
value but each seed still gets its own samples.

One difference from native behaviour: on a server that honours seeds, two identical seeded
requests inside one run usually return the same text natively. Under Copula they are two
occurrences, so they get two samples of the seed's stream. A note in the run summary says so.

## Refused fields

A request is refused with HTTP 400 (`invalid_call`) when it carries a field Copula does not know:
anything besides the messages, the sampling parameters, `max_tokens`/`max_completion_tokens`,
`tools`, `tool_choice`, `response_format`, `parallel_tool_calls`, `n`, `seed`, `stream`,
`stream_options` and the ignored fields above. This holds inside `extra_body` too. Refused, among
others:

- `logprobs`, `top_logprobs`, `logit_bias`, `best_of`, `reasoning_effort`, `prediction`,
  `modalities`, `audio`, `web_search_options`, `functions` and `function_call`;
- server-specific parameters such as vLLM's `min_tokens`, `ignore_eos`, `guided_json` or
  `skip_special_tokens`.

Copula either cannot store what they ask for, or cannot be sure that two requests that differ in
them are different requests. A field that Copula does not understand is never silently dropped.
