# Compare, repeat and sweep

Which runs share samples is decided by their **coordinates**: names and values Copula reads from
the command line (`--variant strict`, `seed=1`), from `--set k=v` and from framework adapters.
Each name has a **class** ([Runs, coordinates and versions](../concepts/runs-and-coordinates.md)):

- **compare**: what you are comparing (`variant`, `prompt`, `config`, `candidate`, ...). Runs that
  differ in it share samples.
- **replicate**: an index of independent repeats (`seed`, `trial`, `rep`, ...). Runs that differ
  in it never share. Every name Copula does not know is a replicate, the safe direction.
- **case**: the input (`case`, `example`, `task_id`, ...). Runs that differ in it never share.
- **annotation**: recorded and ignored (`output_dir`, `log_dir`, `run_name`, ...).

`--compare NAME`, `--replicate NAME`, `--case NAME` and `--annotation NAME` set the class of a
name; the built-in names are listed in [Coordinate names and recipes](../reference/coordinates.md).

The examples on this page use this agent:

```python title="agent.py"
import argparse

from openai import OpenAI

CHECKS = {
    "short": "Check the answer. Reply OK, or give a corrected answer.",
    "strict": "Check every claim in the answer. Reply OK only if all are correct.",
}
parser = argparse.ArgumentParser()
parser.add_argument("--variant", default="short", choices=CHECKS)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
client = OpenAI()


def ask(system, user):
    r = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct", temperature=0.7,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content


for q in ["Why is the sky blue?", "What is a prime number?", "Who wrote Hamlet?"]:
    answer = ask("Answer the question in one sentence.", q)
    verdict = ask(CHECKS[args.variant], f"Q: {q}\nA: {answer}")
    print(f"{answer}  {verdict}")
```

## A/B comparisons

`variant` is a compare coordinate, so the two variants share every call they make alike: here the
answers. Only the checks differ.

```console
$ copula run --exp ab --mock -- python agent.py --variant short
copula: run r-f52adb9e00fc4088  exp=ab  dev: version v-6175fa344b71410b, slot 1  group g-3d125924e786103afdc55bed
copula: served by embedded server (mock model)
copula: coordinates: {"compare": ["variant"]}
copula: new version (compared with earlier versions of ab)
copula: attach: attached to http://127.0.0.1:36177/ (threads copy context; headers on httpx, httpx2, http.client)
mock:1d47afd04f8f956c09abaf78  mock:868ecc712c23fa8abd889090
mock:b4983cb31c399714413314bb  mock:7d1ee14d56c173bd433a31b8
mock:67126eac7b1bf02a7972a06d  mock:02210d6cf0e1b0b298ec8044
copula: run r-f52adb9e00fc4088 done: 6 calls: 0 reused, 6 generated, 0 joined
copula:   6 generated: new request -- no earlier run of the experiment called this model
copula:   details: copula explain r-f52adb9e00fc4088
$ copula run --exp ab --mock -- python agent.py --variant strict
copula: run r-3ca58a27af214790  exp=ab  dev: version v-cc4212cc9a064229, slot 1  group g-3d125924e786103afdc55bed
copula: served by embedded server (mock model)
copula: coordinates: {"compare": ["variant"]}
copula: new version (compared with earlier versions of ab)
copula: attach: attached to http://127.0.0.1:34249/ (threads copy context; headers on httpx, httpx2, http.client)
mock:1d47afd04f8f956c09abaf78  mock:78b0d5f2b00b8f42cdfbfa1c
mock:b4983cb31c399714413314bb  mock:981b10136935a8c0e39b49c1
mock:67126eac7b1bf02a7972a06d  mock:cb5c8d03cdeb27130e0ec840
copula: run r-3ca58a27af214790 done: 6 calls: 3 reused, 3 generated, 0 joined
copula:   3 generated: new request -- closest earlier request (run r-f52adb9e00fc4088): messages[0] (system): 'Check the answer. Reply OK, or give ...' -> 'Check every claim in the answer. Rep...'
copula:   details: copula explain r-3ca58a27af214790
```

Both variants are scored on the same three answers. Their difference is the difference of the
check prompts alone, without the noise of drawing other answers, and the second variant paid for
three calls instead of six. This is what an A/B test or a regression check of an agent wants:
each variant sees samples distributed as it would alone, and the pair shares what it can.

## Replicates

To measure how much the outputs vary, run replicates. `seed` is a replicate coordinate, so
`--seed 1` shares nothing with the runs above:

```console
$ copula run --exp ab --mock -- python agent.py --variant strict --seed 1
copula: run r-fc1d340c12344e21  exp=ab  dev: version v-cc4212cc9a064229, slot 1  group g-5c5a74844e8fdec80caafdb7
copula: served by embedded server (mock model)
copula: coordinates: {"compare": ["variant"], "replicate": ["seed"]}
copula: unchanged since run r-3ca58a27af214790; first run of this version with these replicate or case values (new group, new samples)
copula: attach: attached to http://127.0.0.1:46095/ (threads copy context; headers on httpx, httpx2, http.client)
mock:dc36249c43780c6c8c9b5c99  mock:a246b870209afd4bdd982d58
mock:d78d3d6fca494b8ffed3a21f  mock:fb71b08da736d5d707e3f53b
mock:77794fd593fed936603f7d8c  mock:54bc494d8cd68639b2b5203c
copula: run r-fc1d340c12344e21 done: 6 calls: 0 reused, 6 generated, 0 joined
copula:   3 generated: other replicate -- seed=1 here, seed=(unset) in run r-3ca58a27af214790: replicates never share
copula:   3 generated: new request -- closest earlier request (run r-3ca58a27af214790): messages[1] (user): '...ue?\nA: mock:1d47afd04f8f956c09abaf78' -> '...ue?\nA: mock:dc36249c43780c6c8c9b5c99'
copula:   details: copula explain r-fc1d340c12344e21
```

Within one seed the variants still share. Run `--variant short --seed 1` and it reuses the answers
of `--variant strict --seed 1`, not those of seed 0.

A program without a seed argument gets replicates from `--set`: `--set rep=2` is a coordinate
that Copula sees and the program does not.

```console
$ copula run --exp ab --mock --set rep=2 -- python agent.py --variant strict
copula: run r-da99b76f638e490f  exp=ab  dev: version v-cc4212cc9a064229, slot 1  group g-d49c4962105af7b4f63e6c18
copula: served by embedded server (mock model)
copula: coordinates: {"compare": ["variant"], "replicate": ["rep"]}
copula: unchanged since run r-fc1d340c12344e21; first run of this version with these replicate or case values (new group, new samples)
copula: attach: attached to http://127.0.0.1:35603/ (threads copy context; headers on httpx, httpx2, http.client)
mock:214f3225d6e855b539a68240  mock:81eb24437e64ec9e644bd92f
mock:ae588b19e31caf1156800f25  mock:65a9bbd8c7a3650c2b304180
mock:be1f45c4394191cbc40bbc64  mock:1a1fe73c88c5f6a650616f5c
copula: run r-da99b76f638e490f done: 6 calls: 0 reused, 6 generated, 0 joined
copula:   3 generated: other replicate -- rep=2, seed=(unset) here, rep=(unset), seed=1 in run r-fc1d340c12344e21: replicates never share
copula:   3 generated: new request -- closest earlier request (run r-3ca58a27af214790): messages[1] (user): '...ue?\nA: mock:1d47afd04f8f956c09abaf78' -> '...ue?\nA: mock:214f3225d6e855b539a68240'
copula:   details: copula explain r-da99b76f638e490f
```

An unchanged rerun is a replicate too. It draws new samples without any coordinate, by taking a
new slot ([Quickstart](../getting-started/quickstart.md#4-run-it-again-without-changes)).

## Repeated calls inside a run

A run that sends the same request several times gets a new sample each time: the k-th identical
request of a run reads the k-th sample of its group. Repeats inside a run are independent, as they
are without Copula.

```python title="vote.py"
import sys
from collections import Counter

from openai import OpenAI

client = OpenAI()
n = int(sys.argv[1].split("=")[1])            # n=3
votes = Counter()
for _ in range(n):
    r = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct", temperature=1.0,
        messages=[{"role": "user", "content": "Is 91 a prime? Answer yes or no."}])
    votes[r.choices[0].message.content] += 1
print(len(votes), "distinct answers")
```

Larger budgets read the samples of smaller ones when the budget is a compare coordinate. `n` is
not a built-in name, so `--compare n` says that runs with different `n` are compared:

```console
$ copula run --exp vote --mock --compare n -- python vote.py n=3
copula: run r-9106ed4ed327492e  exp=vote  dev: version v-a70b962648574837, slot 1  group g-5eb7dfcfb46c433efb414d15
copula: served by embedded server (mock model)
copula: coordinates: {"compare": ["n"]}
copula: new version (compared with earlier versions of vote)
copula: attach: attached to http://127.0.0.1:43879/ (threads copy context; headers on httpx, httpx2, http.client)
3 distinct answers
copula: run r-9106ed4ed327492e done: 3 calls: 0 reused, 3 generated, 0 joined
copula:   3 generated: new request -- no earlier run of the experiment called this model
copula:   details: copula explain r-9106ed4ed327492e
$ copula run --exp vote --mock --compare n -- python vote.py n=5
copula: run r-31def7af4f784f07  exp=vote  dev: version v-4214b08839c441cb, slot 1  group g-5eb7dfcfb46c433efb414d15
copula: served by embedded server (mock model)
copula: coordinates: {"compare": ["n"]}
copula: new version (compared with earlier versions of vote)
copula: attach: attached to http://127.0.0.1:43421/ (threads copy context; headers on httpx, httpx2, http.client)
5 distinct answers
copula: run r-31def7af4f784f07 done: 5 calls: 3 reused, 2 generated, 0 joined
copula:   2 generated: more repeats than stored -- occurrence 4 of this request in its group; earlier runs stored 3 occurrence(s)
copula:   details: copula explain r-31def7af4f784f07
```

The run with `n=5` read the three samples of the `n=3` run and generated two more. Without
`--compare n`, `n` would be a replicate and the two runs would share nothing. The `tts` recipe
classes `n`, `beam_width` and `num_beams` as compare coordinates
([recipes](../reference/coordinates.md#recipes)).

## Sweeps

A sweep over a parameter compares its values within each seed:

<!-- docs-test: skip -->
```console
$ for lr in 1e-4 3e-4 1e-3; do
>   copula run --exp sweep --compare lr -- python train_agent.py --lr $lr --seed 1
> done
```

Unknown names are replicates, so without `--compare lr` every value of `lr` would be generated
from scratch: correct, only more expensive. To run the sweep in parallel, start one
[server](project-server.md) for the project first.

In a Slurm array, `SLURM_ARRAY_TASK_ID` becomes a replicate coordinate of every run on its own.

## Final numbers: the holdout phase

Runs that reuse samples are compared on the same draws, which is what you want while developing.
A version chosen *because* it did well on some samples is optimistic when it is scored on those
same samples. Score the final choice in the holdout phase, whose sampled calls share nothing with
the search phase:

<!-- docs-test: skip -->
```console
$ copula run --exp ab --phase holdout -- python agent.py --variant strict --split test
```

Inside one process, `copula.holdout()` does the same for a block of code
([Prompt and harness optimizers](optimizers.md)).

## Seeds in requests

A `seed` in the request body is a stream of its own: the requests with `seed=7` share samples
with other `seed=7` requests of the group, and the k-th identical seeded request of a run reads
the k-th sample of that stream. Copula does not send the client's seed to the model server
([Seeds](../concepts/identical-calls.md#seeds)).
