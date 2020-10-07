# Tips for Large-scale Experiments

`jiant` was designed with large-scale transfer-learning experiments in mind. Here are some tips to manage and collect results from multiple experiments. 

### Aggregated results using `path_parse` 

One common format for running experiments is to run something like the following on SLURM: 

```bash
for TASK in mnli rte squad_v1; do
    for MODEL in roberta-base bert-base-cased; do
        export TASK=${TASK}
        export MODEL=${MODEL}
        export OUTPUT_PATH=/path/to/experiments/${MODEL}/${TASK}
        sbatch my_run_script.sbatch
    done
done
```
where `my_run_script.sbatch` kicks off an experiment, and where the run is saved to the output path `/path/to/experiments/${MODEL}/${TASK}`. As seen in [my_experiment_and_me.md](./my_experiment_and_me.md), the results are stored in `val_metrics.json`.

A quick was to pick up the results across the range of experiments is to run code like this:

```python
import pandas as pd
import jiant.utils.python.io as io
import jiant.utils.path_parse as path_parse

matches = path_parse.match_paths("/path/to/experiments/{model}/{task}/val_metrics.json")
for match in matches:
    match["score"] = io.read_json(match["path"])[match["task"]]["major"]
    del match["path"]
df = pd.DataFrame(matches).set_index(["model", "task"])
```

This returns a nice table of the results for each run across your range of experiments.
