# HTTP interface

A Copula server speaks the model protocols on one port and a small control API next to them.
`copula run` uses both; a program needs only the model endpoints, which its base URL already
points to.

## Model endpoints

```text
POST <base>/r/<token>/v1/chat/completions     OpenAI Chat Completions
POST <base>/r/<token>/v1/responses            OpenAI Responses
GET  <base>/r/<token>/v1/responses/<id>       a response this server issued
POST <base>/r/<token>/v1/messages             Anthropic Messages
GET  <base>/r/<token>/v1/models               the models served
```

`<token>` names the run and proves that the caller may use it. The same paths without `/r/<token>`
work when the request carries the token in the header `X-Copula-Run: <token>`. Once a run has
ended its token is no longer accepted (HTTP 401), because a late attempt could re-read samples
its client already received.

### Request headers

All optional; values are percent-encoded.

| header | meaning |
|---|---|
| `X-Copula-Coords: k=v;k=v` | the request belongs to a unit of the run with these coordinates |
| `X-Copula-Unit: N` | the unit's ordinal among units with the same coordinates (default 1) |
| `X-Copula-Phase: holdout` | a holdout unit; without coordinates, the process run's one holdout unit, shared by every such request |
| `X-Copula-Phase-Source: inferred` | the phase is a guess (by an adapter): on a case overlap, fall back to the search phase instead of refusing |
| `X-Copula-Decision: ID` | the unit belongs to a decision scope |
| `X-Copula-Role: NAME` | the call has a role, placed by the run's routes |
| `x-stainless-retry-count`, `Idempotency-Key` | a client retry: re-read the occurrence whose response was lost, instead of taking the next one |

A retry re-reads an occurrence of the same request in the same run whose response was never
delivered and whose client connection is gone, within 600 s; an `Idempotency-Key` matches only a
key the run has sent before. `copula.headers()` builds the `X-Copula-*` headers for the current
Python context ([Python API](python.md)).

### Response headers

| header | meaning |
|---|---|
| `x-copula-run` | the run or unit the call belonged to |
| `x-copula-group` | its group |
| `x-copula-rule` | how the run was placed, how the call was classified, and the protocol |
| `x-copula-status` | per sample: `generated`, `hit` (stored by another run), `own` (stored earlier by this run), `waited` (joined a generation in flight). The run summary counts `hit` and `own` as *reused* and `waited` as *joined* |
| `x-copula-rank` | per sample: its occurrence in the group, or `-` for a greedy call |

A streamed response carries the status and rank as a final SSE comment
(`: x-copula-status generated; x-copula-rank 1`). Streams are buffered: keep-alive comments (an
Anthropic `ping` event for Messages) until the sample is ready, then the protocol's events.

### Errors

Errors come in the protocol's own error format. A streamed request that was admitted gets
HTTP 200 at once, so an error of its generation arrives as a stream event; a request refused
before that gets an ordinary error reply. Error replies carry `x-should-retry: false`, so the openai SDK does
not retry them.

| status | when |
|---|---|
| 400 `invalid_call`, `invalid_request` | a malformed request, or a field Copula refuses (`invalid_request` from the Responses and Anthropic adapters) |
| 400 `model_error` | the model server rejected the request itself; stored as the sample's result |
| 401 `no_run` | no valid run token, or the run has ended |
| 404 `model_not_found` | a model this server does not serve |
| 404 `not_found` | no such endpoint, or a response id this server did not issue |
| 405 | the wrong method for the endpoint |
| 409 `holdout_overlap` | a holdout unit for a case the search phase used |
| 410 `run_ended` | the run ended while the request was being admitted |
| 500 `internal` | a bug in Copula |
| 502 | the model server failed after `--max-attempts` attempts; nothing was stored |
| 503 `closing` | the server is shutting down |

Responses objects that Copula issued, and the chains behind them, are kept in the project's
registry; `GET /v1/responses/ID` returns them to any valid run token of the project.

## Control API

The control endpoints need the server token (`Authorization: Bearer <token>`, from
`.copula/server.json`). They are how `copula run` registers runs with a live server, locally or
with `--server`.

```text
GET  <base>/copula/v1/health              the server's upstream, models and state
POST <base>/copula/v1/paths               {experiment, coords, ...} -> {paths}: the files whose
                                          digests decide a new run's version
POST <base>/copula/v1/runs                {experiment, recipe, phase, coords, overrides, routes,
                                          digests, owner} -> {run, url}: register a process run
POST <base>/copula/v1/runs/<run>/end      {reads, ok} -> {stats, counts, units, notes, why}: end it
```

A run registered through the control API is served until it is ended; the launcher ends it when
its command exits.
