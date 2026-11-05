# Research configurations

`v2/research_design/factor_catalog_500.json` is the audited balanced 500-candidate design used by `v2/features/balanced_features.py`. `v2/feature_catalog_500.json` retains the earlier conditional-factor catalogue for historical context. Do not substitute one for the other. `v2/loss_experiment_plan.json` records the loss/weight design.

Some historical experiment constants remain in the archived runner scripts and are not centralized here. Optional dependencies are listed in `requirements-optional.txt`; install from the repository root with `python -m pip install -r configs/requirements-optional.txt`.
