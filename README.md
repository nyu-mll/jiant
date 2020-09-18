# Collecting Entailment Data for Pretraining: Experiment Scripts

This branch reflects the code used for the experiments for the paper [Collecting Entailment Data for Pretraining: New Protocols and Negative Results](https://arxiv.org/abs/2004.11997) by Samuel Bowman, Jennimaria Palomaki, Livio Baldini Soares, and Emily Pitler. The data introduced in that paper is available at https://github.com/google-research-datasets/Textual-Entailment-New-Protocols

See the readme for `jiant` 1.2 [here](https://github.com/nyu-mll/jiant/tree/80845ba417bbe6c3b2f7b1ab255a9452c4a3d780) for general context and installation instructions. 

The files specific to this paper are located in the [`data-collection-util` directory](https://github.com/nyu-mll/jiant/tree/nli-data/data-collection-util):

- `191223-paper-runs.sh`: This launches Kubernetes jobs to run the experiments whose results are reported in the paper.
- `PPMI_calculations.ipynb`: This implements the PPMI calculations presented in the paper, and is adapted from code used in [Gururangan et al. '18](https://www.aclweb.org/anthology/N18-2017/).
- `split.sh`: This splits an unordered file of MNLI-style datapoints into training and development sets.
- `results_to_table.py`: This compiles results from a jiant `results.tsv` output file into a draft LaTeX-formatted results table.

## Citation

```bibtex
@inproceedings{Bowman2020EntailmentNewProtocols,
  title={Collecting Entailment Data for Pretraining: New Protocols and Negative Results},
  author={Samuel R. Bowman and Jennimaria Palomaki and Livio Baldini Soares and Emily Pitler},
  year={2020},
  booktitle={Proceedings of EMNLP}
}
```
