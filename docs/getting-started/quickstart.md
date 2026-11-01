# Quickstart

This page takes about five minutes and needs no GPU and no API key: Copula's mock model stands in
for a model server until the last step. You need Copula ([Install](install.md)) and the `openai`
package (`pip install openai`).

## 1. A small agent

The agent answers three questions, then asks the model to check each answer. It is an ordinary
OpenAI client program: nothing in it knows about Copula.

```python title="agent.py"
from openai import OpenAI

ANSWER = "Answer the question in one sentence."
CHECK = "Check the answer. Reply OK, or give a corrected answer."
QUESTIONS = ["Why is the sky blue?", "What is a prime number?", "Who wrote Hamlet?"]

client = OpenAI()  # OPENAI_BASE_URL and OPENAI_API_KEY come from the environment


def ask(system, user):
    r = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.7,
    )
    return r.choices[0].message.content


for q in QUESTIONS:
    answer = ask(ANSWER, q)
    verdict = ask(CHECK, f"Q: {q}\nA: {answer}")
    print(f"{answer}  {verdict}")
```

## 2. Run it under Copula

`copula run` starts the program with its model endpoint pointed at Copula. `--exp demo` names the
experiment: runs of one experiment can share samples. `--mock` serves Copula's mock model, whose
reply is a token that is unique to each sample.

```console
$ copula run --exp demo --mock -- python agent.py
copula: run r-64bcd82a96ad410d  exp=demo  dev: version v-1b298059bbfc4108, slot 1  group g-0e5d5aee9cfa7a183956942b
copula: served by embedded server (mock model)
copula: new version (compared with earlier versions of demo)
copula: attach: attached to http://127.0.0.1:41219/ (threads copy context; headers on httpx, httpx2, http.client)
mock:a2ea4ab3b827641bb0f6e35c  mock:b9bd584e3aa940edbfb3c232
mock:931a3f9b9ced931fd91106e0  mock:ab3a1fec6eefb886b044a396
mock:08a4c8313a48b2234005563d  mock:28d7ed1f7420908c9fc192db
copula: run r-64bcd82a96ad410d done: 6 calls: 0 reused, 6 generated, 0 joined
copula:   6 generated: new request -- no earlier run of the experiment called this model
copula:   details: copula explain r-64bcd82a96ad410d
```

The program printed an answer and a verdict per question. The first line says how Copula placed
the run: a new **version** of the experiment (there was none before), **slot** 1, and the
**group** whose samples it reads and writes. The last lines are the summary: all six calls were
generated, because no earlier run had made them.

## 3. Edit a prompt and run again

Change the check prompt, the way you would while developing:

```python
CHECK = "Check the answer. Reply OK, or explain the mistake."
```
<!-- docs-test: run sed -i 's/Reply OK, or give a corrected answer./Reply OK, or explain the mistake./' agent.py -->

```console
$ copula run --exp demo --mock -- python agent.py
copula: run r-fe7557a3a3f54eff  exp=demo  dev: version v-52dffde4430a45d5, slot 1  group g-0e5d5aee9cfa7a183956942b
copula: served by embedded server (mock model)
copula: new version (compared with earlier versions of demo)
copula: attach: attached to http://127.0.0.1:33231/ (threads copy context; headers on httpx, httpx2, http.client)
mock:a2ea4ab3b827641bb0f6e35c  mock:c484e3cf1ad4ee562ac3fc19
mock:931a3f9b9ced931fd91106e0  mock:0e4ae7fc82d80e7d6edda1d6
mock:08a4c8313a48b2234005563d  mock:11889e9f5bf2ca419c960fd0
copula: run r-fe7557a3a3f54eff done: 6 calls: 3 reused, 3 generated, 0 joined
copula:   3 generated: new request -- closest earlier request (run r-64bcd82a96ad410d): messages[0] (system): '...eply OK, or give a corrected answer.' -> '...eply OK, or explain the mistake.'
copula:   details: copula explain r-fe7557a3a3f54eff
```

Three calls were reused and three generated. The answers (the first column) are the ones of the
first run: the edited program made those calls exactly as before, so they were read from the
store. The checks changed, so they were generated, and the summary shows what changed in the
request.

This is the point of Copula. The two versions are compared on the same answers, so the comparison
is paired and less noisy, and the second run paid only for the calls the edit changed.

## 4. Run it again without changes

```console
$ copula run --exp demo --mock -- python agent.py
copula: run r-0d453dd20c9e4bb8  exp=demo  dev: version v-52dffde4430a45d5, slot 2  group g-99de249e733e13798f798660
copula: served by embedded server (mock model)
copula: unchanged since run r-fe7557a3a3f54eff: a rerun (new slot, new samples)
copula: attach: attached to http://127.0.0.1:37947/ (threads copy context; headers on httpx, httpx2, http.client)
mock:6bec6bcd45de1d933642e841  mock:c5eb9ea273c29ccff825b5fd
mock:3d2c61f3e177f99dcdcf6c63  mock:942f9f67d753c68356accaea
mock:f1fc0cb4b70c95b5bb328299  mock:c2cf7782fdfdc47ff0b753d4
copula: run r-0d453dd20c9e4bb8 done: 6 calls: 0 reused, 6 generated, 0 joined
copula:   3 generated: unchanged rerun (new slot) -- the program is unchanged since run r-fe7557a3a3f54eff, so this run is a rerun: it takes a new slot (2; that run: 1) and draws new samples by design
copula:   3 generated: new request -- closest earlier request (run r-fe7557a3a3f54eff): messages[1] (user): '...ue?\nA: mock:a2ea4ab3b827641bb0f6e35c' -> '...ue?\nA: mock:6bec6bcd45de1d933642e841'
copula:   details: copula explain r-0d453dd20c9e4bb8
```

Nothing was reused. The program and the files it read are unchanged, so this run is a **rerun**,
and a rerun is a new replicate: it takes a new slot and draws new samples. Copula never gives a
rerun the sampled outputs of the run it repeats, because a rerun is how you measure how much the
outputs vary. The answers are new, so the checks of those answers are new requests too.

## 5. Ask why

`copula explain` shows how the latest run was placed and why its calls were generated:

```console
$ copula explain
run r-0d453dd20c9e4bb8  experiment demo  phase search  status done
  dev: version v-52dffde4430a45d5, slot 2; group g-99de249e733e13798f798660; rerun of r-fe7557a3a3f54eff
  files read: 1; units: 0
  calls: 6: 0 reused, 6 generated, 0 joined, 0 errors
      3 generated: unchanged rerun (new slot)
          e.g. the program is unchanged since run r-fe7557a3a3f54eff, so this run is a rerun: it takes a new slot (2; that run: 1) and draws new samples by design
      3 generated: new request
          e.g. closest earlier request (run r-fe7557a3a3f54eff): messages[1] (user): '...ue?\nA: mock:a2ea4ab3b827641bb0f6e35c' -> '...ue?\nA: mock:6bec6bcd45de1d933642e841'
```

`copula ls` lists the runs of the project:

```console
$ copula ls
RUN                    EXPERIMENT   PHASE    KIND     STATUS   VERSION             SLOT  GROUP
r-64bcd82a96ad410d     demo         search   process  done     v-1b298059bbfc4108  1     g-0e5d5aee9cfa7a183956942b
r-fe7557a3a3f54eff     demo         search   process  done     v-52dffde4430a45d5  1     g-0e5d5aee9cfa7a183956942b
r-0d453dd20c9e4bb8     demo         search   process  done     v-52dffde4430a45d5  2     g-99de249e733e13798f798660
```

Versions 1 and 2 hold slot 1 of the same group, so they share; the rerun holds slot 2 of another
group. [Why was this call generated?](../guides/why-generated.md) explains every reason
`copula explain` can give.

## 6. Use your own model server

Drop `--mock`. Copula then forwards to the server your program would have used, the one in
`OPENAI_BASE_URL`:

<!-- docs-test: skip -->
```console
$ vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 &
$ export OPENAI_BASE_URL=http://localhost:8000/v1
$ copula run --exp demo-qwen -- python agent.py
```

Samples are stored per model revision, so the mock model's samples are never served for Qwen.
[Connect a model server](../guides/model-servers.md) covers vLLM, SGLang and hosted APIs.

## Next

- [Run an existing program](../guides/existing-programs.md): what `copula run` changes for the
  program, and the clients and languages it supports.
- [Compare, repeat and sweep](../guides/compare-and-repeat.md): A/B tests, replicates and seeds.
- [How Copula works](../concepts/how-it-works.md): versions, slots and groups.
