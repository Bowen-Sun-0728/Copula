# Coordinate names and recipes

The class of a coordinate name decides what it does to sharing
([Runs, coordinates and versions](../concepts/runs-and-coordinates.md#classes)). This page lists
the names Copula knows. Every other name is a **replicate**.

## Built-in names

Names are compared after normalisation: lower case, leading dashes removed, `-` read as `_`.

| class | names |
|---|---|
| **compare** | `variant`, `version`, `candidate`, `config`, `arm`, `method`, `budget`, `prompt` |
| **replicate** | `seed`, `trial`, `rep`, `repeat`, `replicate`, `attempt`, `epoch`, `rollout`, `shard`, `chunk`, `fold`, `slurm_array_task_id`, and every name not listed on this page |
| **case** | `case`, `example`, `sample_id`, `doc_id`, `task_id`, `instance`, `problem` |
| **annotation** | `iteration`, `metric_key`, `workflow_name`, `group_id`, `trace_id`, `session_id`, `thread_id`, `run_name`, `name`, `o`, `out`, `outdir`, `output`, `output_dir`, `log_dir`, `logdir`, `save_dir`, `savedir` |

Also annotations: every name of the form `o`, `out`, `output`, `log`, `logs`, `save` or `ckpt`
followed by `dir`, `path`, `file` or `prefix`, with or without `_` (`output_path`, `log_file`,
`ckptdir`, ...).

## Recipes

`--recipe NAME` adds names to these defaults for a kind of workload. A recipe never turns an
unknown name into anything but a replicate except through its own list.

| recipe | adds |
|---|---|
| `dev` | nothing: the defaults above (the default recipe) |
| `eval` | compare: `task`, `task_identifier`, `model_config`, `solver`; case: `sample`, `test`, `test_idx` |
| `optimize` | annotation: `parent` |
| `tts` | compare: `n`, `beam_width`, `num_beams` |
| `ab` | nothing yet |
| `sweep` | nothing yet |
| `sdg` | nothing yet |
| `rl` | nothing yet |

`ab`, `sweep`, `sdg` and `rl` are accepted and recorded with the run; in this version they class
names like `dev`. Overrides (`--compare`, `--replicate`, `--case`, `--annotation`) take
precedence over the recipe.

## Where coordinates come from

| source | example | gives |
|---|---|---|
| `--name value`, `--name=value` | `python agent.py --variant strict` | `variant=strict` |
| `-x value` | `python agent.py -o out/v2` | `o=out/v2` (an annotation) |
| Hydra-style overrides | `python train.py lr=3e-4 +model.size=7b` | `lr=3e-4`, `model.size=7b` |
| `--set` | `copula run --set rep=2 -- ...` | `rep=2` |
| Slurm | `SLURM_ARRAY_TASK_ID=3` | `slurm_array_task_id=3` |
| `copula.run(...)` | `copula.run(candidate=i, case=q)` | unit coordinates |
| `X-Copula-Coords` | `X-Copula-Coords: case=17;arm=b` | unit coordinates |
| DSPy adapter | | `candidate`, `case`, `optimizer_seed` (a replicate) |

Positional words give no coordinate, and a lone `--` ends the options. A flag followed by a word
that is not an option takes it as its value (`python -u agent.py` gives `u=agent.py`); a flag
followed by another option, or last on the line, gives none. A word after a flag that looks like
an override (`--fast lr=3e-4`) is an override, not the flag's value. Values are compared as
strings.
