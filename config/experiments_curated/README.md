# Curated experiment configs

This folder contains a small, manual progression of experiments aligned with the narrative in `docs/latex/main.tex`.

Order:
1. `01_base_logistic_regression.yaml` - original features only.
2. `02_categorization_logistic_regression.yaml` - adds clinical categorization.
3. `03_aggregations_logistic_regression.yaml` - adds aggregated risk scores.
4. `04_combined_logistic_regression.yaml` - combines categorization and aggregations.
5. `05_final_lightgbm_sangrado.yaml` - adds `SANGRADO_MAYOR` and switches to LightGBM.
6. `06_interpretable_decision_tree_sangrado.yaml` - interpretable tree on the same final feature set.

Use one file at a time by copying it over `config/pipeline_config.yaml` before running the pipeline.
