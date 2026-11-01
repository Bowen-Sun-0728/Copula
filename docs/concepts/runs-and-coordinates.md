# Runs, coordinates and versions

## Runs and units

A **run** is one launched command: `copula run -- COMMAND` registers a *process run*. Inside it,
**units** are smaller runs, each with coordinates of its own, typically one evaluation of one
candidate on one case. Units come from:

- framework adapters (DSPy module calls, [Prompt and harness optimizers](../guides/optimizers.md));
- `with copula.run(candidate=..., case=...):` in Python ([Python API](../reference/python.md));
- `X-Copula-Coords` headers on requests, from any language ([HTTP interface](../reference/http.md)).

A unit's coordinates are merged over its process run's. The k-th unit with the same coordinates
in one process is a different unit from the first (ordinal k), so it reads new samples, like a
repeated request.

## Coordinates

Coordinates are names and values that describe a run:

- **The command line.** Every `--name value`, `--name=value`, `-o value` and Hydra-style
  `name=value` (also `+name=value`) of the command's arguments; a lone `--` ends the options.
  Positional words give none. A flag followed by a word that is not an option takes that word as
  its value, so `python -u agent.py` gives `u=agent.py` and `--verbose data.csv` gives
  `verbose=data.csv`: harmless replicates while the command line stays the same. `sh -c SCRIPT`
  gives the coordinate `c`, the whole script, so put the program directly after `--` when its
  arguments matter.
- **`--set name=value`**, for coordinates the command line does not carry.
- **`SLURM_ARRAY_TASK_ID`**, as `slurm_array_task_id`.
- **Adapters**, such as `candidate`, `case` and `optimizer_seed` of the DSPy adapter.

Names are compared in lower case, without leading dashes and with `-` read as `_`: `--Output-Dir`
is `output_dir`. Values are strings, so `--seed 0` and a run that sets no seed differ.

## Classes

The class of a name decides what it does to sharing:

| class | meaning | effect |
|---|---|---|
| **compare** | what you compare: variants, configs, candidates, budgets | left out of the group: runs that differ in it share; it makes a run a new version |
| **replicate** | an index of independent repeats: seeds, trials | part of the group: runs that differ in it never share |
| **case** | the input | part of the group: runs that differ in it never share |
| **annotation** | output and log paths, run names | recorded only |

Every name that is not built in is a replicate (`log_level` and `verbosity` too): an unknown name
can only cost samples, never make two runs share that should not. Note that `version` is a
built-in compare coordinate, unrelated to the versions Copula derives from files. The built-in names, and the names each `--recipe` adds, are in
[Coordinate names and recipes](../reference/coordinates.md). `--compare`, `--replicate`, `--case`
and `--annotation` override the class of a name for one run.

## Phases

A run is in the **search** phase unless `--phase holdout` (or `copula.holdout()`, or an adapter)
puts it in the **holdout** phase, which has groups of its own: the sampled calls of the two
phases never share. (Greedy calls to a model with an established identity are global, in every
phase.) A holdout unit whose case the search phase of the experiment already used is refused
with HTTP 409 (W5a); for whole process runs in the holdout phase the cases are not checked. Report
final numbers from the holdout phase: a version selected on shared search samples is scored there
on samples the search never saw.

## Versions

With `--exp`, every process run records the project files it read and their content
([Run an existing program](../guides/existing-programs.md#versions-the-files-the-program-reads)).
A new run is an **unchanged rerun** of the most recent earlier run of the experiment that has the
same compare values and whose files all still have the content they had. Otherwise it is a
**new version**.

When Copula cannot know, it counts the files as unchanged, so the run is a rerun and draws new
samples, the safe direction:

- an earlier run is still running, so its read set is not known yet;
- an earlier run failed (non-zero exit), so its read set was not recorded, and no successful run
  with the same compare values came after it;
- an earlier run read no project file at all;
- a file cannot be read now (deleted, or not visible on this host).

Files larger than 256 MB are compared by size and modification time instead of content.

Three consequences are worth knowing:

- **Reverting an edit** makes the program equal to an earlier version again, so the run is a
  rerun of that version and draws new samples.
- **After a failed run**, the next run is counted as a rerun of the failed one (its files are
  unknown), so it shares nothing with earlier versions. Once a run succeeds, the following runs
  are matched against it again.
- **A program Copula does not trace** records no reads, so every later run counts as unchanged;
  name its files with `--track`
  ([Run an existing program](../guides/existing-programs.md#versions-the-files-the-program-reads)).

## Groups and slots

For a process run, with `H` a hash and the **counter** the key under which slots are numbered:

```text
counter = H(experiment, phase, granularity, replicate values, case values, version)
slot    = the smallest slot not yet used under that counter
group   = H(experiment, phase, granularity, replicate values, case values, slot)
```

The granularity is *process* or *unit*. Units use the same rule, with the unit's version derived
from its process run's version and its own compare values. The version is in the counter but not in the group: every new version takes
slot 1 of its counter, and slot 1 is one group for all versions. An unchanged rerun finds slot 1
used and takes slot 2: a new group. Matching the version and claiming the slot happen in one
transaction, so two runs launched at the same moment never take the same slot.

A run outside any experiment (no `--exp`) gets a fresh group of its own.

## Decisions

Inside `copula.decision()`, a unit reads the slot of the decision's first unit on the same case,
unless this unit's counter has already used that slot; then it takes its own next slot. A parent and a child
compared in one decision then read the same samples where their calls agree. The GEPA adapter
opens one decision per proposal.

## Roles and routes

A call made with a role (`copula.role(name)` or `X-Copula-Role`) is placed by the run's routes:
`--route ROLE=run` (the default: the run's group), `global` (one group per experiment, phase and
role), `per:COORD` (one group per value of a coordinate) or `fresh` (uncached). Routes need an
experiment; without `--exp` they are ignored. Routes let, say, the
proposer of an optimizer share across runs of one seed while evaluations stay per run.
