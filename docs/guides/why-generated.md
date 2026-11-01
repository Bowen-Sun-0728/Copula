# Why was this call generated?

Every call Copula answers is either **reused** (read from the store), **joined** (another run was
generating the same sample at that moment, and this one waited for it) or **generated**. For each
generated call, Copula records why it could not be reused. The summary at the end of a run shows
the main reasons with an example of each, and `copula explain` shows all of them.

## Look at a run

<!-- docs-test: run mkdir -p prompts && echo "You are a careful assistant." > prompts/system.txt -->

```python title="ask.py"
from openai import OpenAI

client = OpenAI()
system = open("prompts/system.txt").read()
for q in ["Why is the sky blue?", "Who wrote Hamlet?"]:
    r = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct", temperature=0.7,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": q}])
    print(r.choices[0].message.content)
```

The first run generates everything; a run after an edit of the prompt file says what changed:

```console
$ copula run --exp why --mock -- python ask.py
copula: run r-8265c7968057407d  exp=why  dev: version v-42b1ca3be5654d0e, slot 1  group g-bd24407801173232f4511b1a
copula: served by embedded server (mock model)
copula: new version (compared with earlier versions of why)
copula: attach: attached to http://127.0.0.1:45401/ (threads copy context; headers on httpx, httpx2, http.client)
mock:12f61310818743556a2c6ad1
mock:b4d0e080cd43494fde1cf9d3
copula: run r-8265c7968057407d done: 2 calls: 0 reused, 2 generated, 0 joined
copula:   2 generated: new request -- no earlier run of the experiment called this model
copula:   details: copula explain r-8265c7968057407d
$ echo "You are a concise assistant." > prompts/system.txt
$ copula run --exp why --mock -- python ask.py
copula: run r-a96e3d2645ef4bf8  exp=why  dev: version v-41fcd95223e046ae, slot 1  group g-bd24407801173232f4511b1a
copula: served by embedded server (mock model)
copula: new version (compared with earlier versions of why)
copula: attach: attached to http://127.0.0.1:46671/ (threads copy context; headers on httpx, httpx2, http.client)
mock:6af80f835c28dc15c9db65db
mock:be16a3e49738f7686657e648
copula: run r-a96e3d2645ef4bf8 done: 2 calls: 0 reused, 2 generated, 0 joined
copula:   2 generated: new request -- closest earlier request (run r-8265c7968057407d): messages[0] (system): 'You are a careful assistant.\n' -> 'You are a concise assistant.\n'
copula:   details: copula explain r-a96e3d2645ef4bf8
```

`copula explain` (the latest run by default, or `copula explain RUN`, or `last~1` for the one
before) shows how the run was placed and every reason, with an example each; `--calls` adds one
line per call:

```console
$ copula run --exp why --mock -- python ask.py
copula: run r-c2471dd7f38b487b  exp=why  dev: version v-41fcd95223e046ae, slot 2  group g-0de77ae6eb93f6cc1d5b02df
copula: served by embedded server (mock model)
copula: unchanged since run r-a96e3d2645ef4bf8: a rerun (new slot, new samples)
copula: attach: attached to http://127.0.0.1:37937/ (threads copy context; headers on httpx, httpx2, http.client)
mock:94dc0728e34e6d25d2e9b8b2
mock:2ea3115e998a0b9b3d2c64ba
copula: run r-c2471dd7f38b487b done: 2 calls: 0 reused, 2 generated, 0 joined
copula:   2 generated: unchanged rerun (new slot) -- the program is unchanged since run r-a96e3d2645ef4bf8, so this run is a rerun: it takes a new slot (2; that run: 1) and draws new samples by design
copula:   details: copula explain r-c2471dd7f38b487b
$ copula explain --calls
run r-c2471dd7f38b487b  experiment why  phase search  status done
  dev: version v-41fcd95223e046ae, slot 2; group g-0de77ae6eb93f6cc1d5b02df; rerun of r-a96e3d2645ef4bf8
  files read: 2; units: 0
  calls: 2: 0 reused, 2 generated, 0 joined, 0 errors
      2 generated: unchanged rerun (new slot)
          e.g. the program is unchanged since run r-a96e3d2645ef4bf8, so this run is a rerun: it takes a new slot (2; that run: 1) and draws new samples by design
  #5      r-c2471dd7f38b487b       actor:s    generated generated  rerun: the program is unchanged since run r-a96e3d2645ef4bf8, so this run is a rerun: it takes a new slot (2; that run: 1) and draws new samples by design
  #6      r-c2471dd7f38b487b       actor:s    generated generated  rerun: the program is unchanged since run r-a96e3d2645ef4bf8, so this run is a rerun: it takes a new slot (2; that run: 1) and draws new samples by design
```

The run line gives the run's version, slot and group, and the run it is a rerun of. Each call line
gives its sequence number in the call log, the run or unit, the node (`actor:s` is a sampled call
to the default model), the outcome and the reason. `--json` prints the same as JSON lines for
scripts: one object per run and unit, and with `--calls` one per call.

## Checklist

When calls you expected to be reused were generated:

1. **Is the run in an experiment?** Without `--exp`, a run shares nothing.
2. **Is it an unchanged rerun?** Then it draws new samples by design ([`rerun`](#rerun)). An
   edited program, or one with other compare coordinates, is a new version and shares.
3. **Do the coordinates differ?** Runs with different replicate or case values, phases or
   experiments never share ([`replicate`](#replicate)). Unknown names are replicates: look at the
   `coordinates:` line of the banner and class a name with `--compare NAME` or
   `--annotation NAME` if it should not separate runs.
4. **Did the request change?** [`new_request`](#new_request) shows the nearest earlier request and
   where it differs. Timestamps, random ids, unordered sets or unseeded shuffles in a prompt make
   every request new.
5. **Did the model change?** A new checkpoint, server version or sampling table is a new model
   revision ([`model`](#model)).

## Reasons

Copula reports the first reason that applies. It checks `uncached`, `after_error`, `new_request`
and `model` first. If an earlier run stored the request in this run's own group, the reason is
`more_repeats` (or `unexplained`). Otherwise the groups differ, and it checks what keeps them
apart: `experiment`, `phase`, `granularity`, `replicate`, `case`, `seed`, `route`, then the
unit's ordinal (`more_repeats` or `arrival_order`), `rerun` and finally `slot`.

### `uncached`

The call is not cached: its role is routed `fresh` (`--route ROLE=fresh`), so it is drawn anew
every time. *Fix:* nothing to fix if that is the route you mean. (Requests that rely on
provider-side state are refused rather than passed through.)

### `after_error`

Earlier runs sent this request, but every earlier attempt failed (the upstream refused it or was
unreachable), so there was no sample to reuse. Misconfiguration and transport errors are never
stored, so the request is generated again once the upstream answers.

### `new_request`

No earlier run of the experiment sent this request to this model. The explanation names the
closest earlier request and the first place where they differ:

```text
3 generated: new request -- closest earlier request (run r-74437c2a66b94f2e): messages[0] (system): '...eply OK, or give a corrected answer.' -> '...eply OK, or explain the mistake.'
```

The path (`messages[0]`, `temperature`, `tools[1].function.description`, ...) says which field
changed. A call whose input contains the output of a call that was itself generated is a new
request too: its prompt holds a new sample.

*Fix:* if the change is not one you made, look for volatile values in the prompt: the current
date or time, request ids, `set` or `dict` iteration order, unseeded `random`. Make them stable,
or keep them out of the request.

### `model`

Earlier runs sent the same request to another revision of the model: other weights, another
server version, another sampling table or other serving settings
([Model identity](model-servers.md#vllm-and-sglang)). Samples are never served across revisions.

### `more_repeats`

The request was sent more times in this run's group than earlier runs sent it: occurrence 4 when
three are stored, say. The first three were reused, the rest generated. For units, it also covers
a run with more identical units (same coordinates) than earlier runs had.

### `rerun`

The program and every file it read are unchanged since an earlier run, so this run is a rerun: a
new replicate, which takes a new slot and draws new samples. *Fix:* if you wanted the earlier
outputs, you wanted a replay, which this version does not provide.

### `arrival_order`

Units with equal coordinates are numbered in the order they start. This unit and the one that
stored the sample have the same coordinates but other ordinals, because they started in another
order: in a parallel evaluation, say, or when a batch holds one example twice. Each run still sees
fresh, independent samples; they are only assigned differently. *Fix:* a coordinate that tells
the units apart (`case=...`) makes the assignment stable.

### `replicate`

Earlier runs sent the request with another value of a replicate coordinate (`seed=1` here,
`seed=2` there, or unset). Replicates never share. *Fix:* if the name is not a replicate index,
class it: `--compare NAME` or `--annotation NAME`.

### `case`

Earlier runs sent the request for another case (a coordinate of the case class). Cases are kept
apart.

### `phase`

Earlier runs sent it in the other phase: search and holdout never share.

### `seed`

Earlier runs sent it with another `seed` in the request body; each seed is its own sample stream.

### `route`

Earlier runs sent it with another role, which `--route` places in another group.

### `experiment`

Earlier runs of another experiment sent it. Experiments never share.

### `granularity`

Earlier runs sent it from a whole process run and this run from a unit, or the other way around.
A process run and a unit never share, so keep the unit structure of an experiment the same across
its runs.

### `slot`

Earlier runs stored it in another slot of the same version and coordinates, for example
replicates launched at the same moment.

### `unexplained`

The sample was stored in this run's group by a run that generated it at the same time, or the
records needed to explain it are gone. Rare; not an error.
