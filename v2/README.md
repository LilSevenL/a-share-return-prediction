# V2 research

- [Latest progress and results](docs/V2_PROGRESS_SUMMARY.md)
- [V1/V2 comparison](docs/V1_V2_MODEL_COMPARISON.md)
- [Research index](docs/RESEARCH_DOCUMENT_INDEX.md)
- [Reproduction guide](docs/REPRODUCING.md)
- [Experiment reports](../reports/v2/)
- [Factor catalogues and experiment plan](../configs/v2/)

Code is grouped into `features/`, `models/`, `backtest/`, `experiments/`, and `tests/`. Run scripts from the repository root, for example `python v2/experiments/run_local_ols.py --help`. Data-free checks run with `python -m unittest discover -s v2/tests`.

The bootstrap in `_paths.py` locates cross-directory imports and V1 shared code. Local data remain under `<repository>/quant_data/`. This is historical development research with survivorship bias and repeatedly inspected 2017–2023 data; candidate selection and new independent validation remain outstanding.
