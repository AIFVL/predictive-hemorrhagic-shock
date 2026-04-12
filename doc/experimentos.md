# Catalogo de Experimentos Historicos

Generado automaticamente el 2026-04-11T20:34:59.

Este documento enumera los experimentos detectados en el repositorio segun el orden de aparicion en la estructura `models/` (orden lexicografico por ruta).

## Resumen

- Total de experimentos encontrados: **45**
- Nota: en varios experimentos faltan artefactos de evaluacion o analisis de umbral; se reportan como `N/A`.

| # | Version | Modelo | Timestamp entrenamiento | Muestras | Features | Accuracy test | Recall test | ROC AUC test | Metadata | Evaluacion |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | v1 | gradient_boosting | 2026-01-26T00:29:39.276272 | 1045 | 18 | 0.6107 | 0.2381 | 0.5317 | models/v1/gradient_boosting/metadata.json | output/v1/evaluation/gradient_boosting/evaluation.json |
| 2 | v1 | logistic_regression | 2026-01-26T00:29:45.814603 | 1045 | 18 | 0.3511 | 0.9286 | 0.5796 | models/v1/logistic_regression/metadata.json | output/v1/evaluation/logistic_regression/evaluation.json |
| 3 | v1 | random_forest | 2026-01-26T00:29:40.656614 | 1045 | 18 | 0.4542 | 0.7143 | 0.5321 | models/v1/random_forest/metadata.json | output/v1/evaluation/random_forest/evaluation.json |
| 4 | v10 | gradient_boosting | 2026-01-26T05:04:02.506286 | 1045 | 18 | 0.5954 | 0.2381 | 0.5433 | models/v10/gradient_boosting/metadata.json | output/v10/evaluation/gradient_boosting/evaluation.json |
| 5 | v10 | knn | 2026-01-26T05:02:07.102919 | 1045 | 18 | 0.5611 | 0.2381 | 0.4923 | models/v10/knn/metadata.json | output/v10/evaluation/knn/evaluation.json |
| 6 | v10 | lightgbm | 2026-02-19T15:32:54.160826 | 1045 | 18 | 0.5305 | 0.2619 | 0.5077 | models/v10/lightgbm/metadata.json | output/v10/evaluation/lightgbm/evaluation.json |
| 7 | v10 | logistic_regression | 2026-01-26T05:04:00.812820 | 1045 | 18 | 0.3206 | 1.0000 | 0.5898 | models/v10/logistic_regression/metadata.json | output/v10/evaluation/logistic_regression/evaluation.json |
| 8 | v10 | random_forest | 2026-01-26T05:08:56.696539 | 1045 | 18 | 0.3397 | 0.9762 | 0.5676 | models/v10/random_forest/metadata.json | output/v10/evaluation/random_forest/evaluation.json |
| 9 | v11 | lightgbm | 2026-02-19T20:36:44.256309 | 1045 | 18 | 0.5305 | 0.2619 | 0.5077 | models/v11/lightgbm/metadata.json | output/v11/evaluation/lightgbm/evaluation.json |
| 10 | v17 | decision_tree | 2026-03-05T19:03:14.432007 | 1045 | 19 | 0.5344 | 0.7976 | 0.6864 | models/v17/decision_tree/metadata.json | output/v17/evaluation/decision_tree/evaluation.json |
| 11 | v17 | lightgbm | 2026-03-05T19:03:16.247979 | 1045 | 19 | 0.4809 | 0.8095 | 0.6527 | models/v17/lightgbm/metadata.json | output/v17/evaluation/lightgbm/evaluation.json |
| 12 | v18 | decision_tree | 2026-03-07T00:24:32.863785 | 1045 | 18 | 0.5344 | 0.7857 | 0.6428 | models/v18/decision_tree/metadata.json | output/v18/evaluation/decision_tree/evaluation.json |
| 13 | v18 | lightgbm | 2026-03-07T00:24:34.820254 | 1045 | 18 | 0.4809 | 0.7976 | 0.5946 | models/v18/lightgbm/metadata.json | output/v18/evaluation/lightgbm/evaluation.json |
| 14 | v19 | decision_tree | 2026-03-07T01:01:26.106726 | 1045 | 19 | 0.5344 | 0.7976 | 0.6864 | models/v19/decision_tree/metadata.json | output/v19/evaluation/decision_tree/evaluation.json |
| 15 | v19 | lightgbm | 2026-03-07T01:02:17.126359 | 1045 | 19 | 0.5878 | 0.5595 | 0.6482 | models/v19/lightgbm/metadata.json | output/v19/evaluation/lightgbm/evaluation.json |
| 16 | v19 | logistic_regression | 2026-03-07T01:01:55.497594 | 1045 | 19 | 0.6221 | 0.5119 | 0.6533 | models/v19/logistic_regression/metadata.json | output/v19/evaluation/logistic_regression/evaluation.json |
| 17 | v19 | naive_bayes | 2026-03-07T01:01:36.882471 | 1045 | 19 | 0.7252 | 0.2143 | 0.6261 | models/v19/naive_bayes/metadata.json | output/v19/evaluation/naive_bayes/evaluation.json |
| 18 | v2 | gradient_boosting | 2026-01-26T00:32:09.158886 | 1045 | 20 | 0.6107 | 0.2381 | 0.5303 | models/v2/gradient_boosting/metadata.json | output/v2/evaluation/gradient_boosting/evaluation.json |
| 19 | v2 | logistic_regression | 2026-01-26T00:32:15.529249 | 1045 | 20 | 0.3473 | 0.9286 | 0.5798 | models/v2/logistic_regression/metadata.json | output/v2/evaluation/logistic_regression/evaluation.json |
| 20 | v2 | random_forest | 2026-01-26T00:32:10.615334 | 1045 | 20 | 0.4809 | 0.6310 | 0.5194 | models/v2/random_forest/metadata.json | output/v2/evaluation/random_forest/evaluation.json |
| 21 | v3 | gradient_boosting | 2026-01-26T00:44:36.079907 | 1045 | 22 | 0.5954 | 0.1905 | 0.5283 | models/v3/gradient_boosting/metadata.json | output/v3/evaluation/gradient_boosting/evaluation.json |
| 22 | v3 | logistic_regression | 2026-01-26T00:44:42.027163 | 1045 | 22 | 0.3511 | 0.9286 | 0.5795 | models/v3/logistic_regression/metadata.json | output/v3/evaluation/logistic_regression/evaluation.json |
| 23 | v3 | random_forest | 2026-01-26T00:44:37.440807 | 1045 | 22 | 0.4504 | 0.7024 | 0.5154 | models/v3/random_forest/metadata.json | output/v3/evaluation/random_forest/evaluation.json |
| 24 | v4 | gradient_boosting | 2026-01-26T00:46:48.892292 | 1045 | 24 | 0.6031 | 0.2143 | 0.5262 | models/v4/gradient_boosting/metadata.json | output/v4/evaluation/gradient_boosting/evaluation.json |
| 25 | v4 | logistic_regression | 2026-01-26T00:46:56.071690 | 1045 | 24 | 0.3473 | 0.9286 | 0.5800 | models/v4/logistic_regression/metadata.json | output/v4/evaluation/logistic_regression/evaluation.json |
| 26 | v4 | random_forest | 2026-01-26T00:46:49.972987 | 1045 | 24 | 0.4466 | 0.6548 | 0.5096 | models/v4/random_forest/metadata.json | output/v4/evaluation/random_forest/evaluation.json |
| 27 | v5 | gradient_boosting | 2026-01-26T02:02:40.325809 | 1045 | 18 | 0.5725 | 0.2381 | 0.5210 | models/v5/gradient_boosting/metadata.json | output/v5/evaluation/gradient_boosting/evaluation.json |
| 28 | v5 | logistic_regression | 2026-01-26T02:01:23.818565 | 1045 | 18 | 0.3206 | 1.0000 | 0.5892 | models/v5/logistic_regression/metadata.json | output/v5/evaluation/logistic_regression/evaluation.json |
| 29 | v5 | random_forest | 2026-01-26T02:02:34.457276 | 1045 | 18 | 0.3397 | 0.9643 | 0.5384 | models/v5/random_forest/metadata.json | output/v5/evaluation/random_forest/evaluation.json |
| 30 | v6 | gradient_boosting | 2026-01-26T02:11:55.852567 | 1045 | 18 | 0.6489 | 0.1548 | 0.5595 | models/v6/gradient_boosting/metadata.json | output/v6/evaluation/gradient_boosting/evaluation.json |
| 31 | v6 | logistic_regression | 2026-01-26T02:12:19.550846 | 1045 | 18 | 0.3893 | 0.9048 | 0.5841 | models/v6/logistic_regression/metadata.json | output/v6/evaluation/logistic_regression/evaluation.json |
| 32 | v6 | random_forest | 2026-01-26T02:13:05.021771 | 1045 | 18 | 0.3435 | 0.9643 | 0.5633 | models/v6/random_forest/metadata.json | output/v6/evaluation/random_forest/evaluation.json |
| 33 | v7 | gradient_boosting | 2026-01-26T02:59:03.973168 | 1045 | 18 | 0.6489 | 0.1548 | 0.5595 | models/v7/gradient_boosting/metadata.json | output/v7/evaluation/gradient_boosting/evaluation.json |
| 34 | v7 | logistic_regression | 2026-01-26T02:59:26.132467 | 1045 | 18 | 0.3893 | 0.9048 | 0.5841 | models/v7/logistic_regression/metadata.json | output/v7/evaluation/logistic_regression/evaluation.json |
| 35 | v7 | random_forest | 2026-01-26T03:00:05.683005 | 1045 | 18 | 0.3435 | 0.9643 | 0.5633 | models/v7/random_forest/metadata.json | output/v7/evaluation/random_forest/evaluation.json |
| 36 | v8 | gradient_boosting | 2026-01-26T03:34:12.579887 | 1045 | 18 | 0.5763 | 0.2381 | 0.5287 | models/v8/gradient_boosting/metadata.json | output/v8/evaluation/gradient_boosting/evaluation.json |
| 37 | v8 | knn | 2026-01-26T03:34:18.621176 | 1045 | 18 | 0.5611 | 0.2381 | 0.4923 | models/v8/knn/metadata.json | output/v8/evaluation/knn/evaluation.json |
| 38 | v8 | lightgbm | 2026-01-26T03:35:07.232667 | 1045 | 18 | 0.5687 | 0.1905 | 0.5228 | models/v8/lightgbm/metadata.json | output/v8/evaluation/lightgbm/evaluation.json |
| 39 | v8 | logistic_regression | 2026-01-26T03:33:10.525059 | 1045 | 18 | 0.3206 | 1.0000 | 0.5898 | models/v8/logistic_regression/metadata.json | output/v8/evaluation/logistic_regression/evaluation.json |
| 40 | v8 | random_forest | 2026-01-26T03:34:53.433061 | 1045 | 18 | 0.3435 | 0.9881 | 0.5737 | models/v8/random_forest/metadata.json | output/v8/evaluation/random_forest/evaluation.json |
| 41 | v9 | gradient_boosting | 2026-01-26T04:24:01.933095 | 1045 | 18 | 0.6756 | 0.0952 | 0.5716 | models/v9/gradient_boosting/metadata.json | output/v9/evaluation/gradient_boosting/evaluation.json |
| 42 | v9 | knn | 2026-01-26T04:24:07.846478 | 1045 | 18 | 0.5992 | 0.1667 | 0.5011 | models/v9/knn/metadata.json | output/v9/evaluation/knn/evaluation.json |
| 43 | v9 | lightgbm | 2026-01-26T04:24:44.363668 | 1045 | 18 | 0.6069 | 0.1310 | 0.5620 | models/v9/lightgbm/metadata.json | output/v9/evaluation/lightgbm/evaluation.json |
| 44 | v9 | logistic_regression | 2026-01-26T04:23:22.429514 | 1045 | 18 | 0.3206 | 1.0000 | 0.5898 | models/v9/logistic_regression/metadata.json | output/v9/evaluation/logistic_regression/evaluation.json |
| 45 | v9 | random_forest | 2026-01-26T04:24:25.080837 | 1045 | 18 | 0.3435 | 0.9881 | 0.5682 | models/v9/random_forest/metadata.json | output/v9/evaluation/random_forest/evaluation.json |

## Detalle por Experimento

### 1. v1 / gradient_boosting

**Identificacion**
- Version (ruta): `v1`
- Modelo: `gradient_boosting`
- Timestamp entrenamiento: `2026-01-26T00:29:39.276272`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.6220095693779905`, test_std=`0.021821539236347144`
- precision: test_mean=`0.38781410590969223`, test_std=`0.026909628296727158`
- recall: test_mean=`0.30149253731343284`, test_std=`0.05925203952620659`
- f1: test_mean=`0.3354471608487365`, test_std=`0.03832470386837334`
- roc_auc: test_mean=`0.5783582089552239`, test_std=`0.03137614529096873`
- f2: test_mean=`0.3135845749868574`, test_std=`0.051654033490690456`

**Evaluacion test (si existe)**
- accuracy: `0.6106870229007634`
- precision: `0.3448275862068966`
- recall: `0.23809523809523808`
- specificity: `0.7865168539325843`
- f1_score: `0.28169014084507044`
- roc_auc: `0.5316680042803638`
- average_precision: `0.37212678991671155`
- optimal_threshold: `0.25415867270267334`

**Artefactos detectados**
- metadata: `models/v1/gradient_boosting/metadata.json`
- modelo serializado: `models/v1/gradient_boosting/model.joblib`
- evaluacion: `output/v1/evaluation/gradient_boosting/evaluation.json`
- threshold_comparison_json: `N/A`
- threshold_comparison_csv: `N/A`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "ccp_alpha": 0.0,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.08,
  "loss": "log_loss",
  "max_depth": 6,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 4,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 150,
  "n_iter_no_change": null,
  "random_state": 42,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 2. v1 / logistic_regression

**Identificacion**
- Version (ruta): `v1`
- Modelo: `logistic_regression`
- Timestamp entrenamiento: `2026-01-26T00:29:45.814603`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.3397129186602871`, test_std=`0.018407066087723762`
- precision: test_mean=`0.3167729147566062`, test_std=`0.014671609647684696`
- recall: test_mean=`0.9194029850746267`, test_std=`0.07164179104477611`
- f1: test_mean=`0.4710752580509851`, test_std=`0.025383740286125864`
- roc_auc: test_mean=`0.5299768761824679`, test_std=`0.07398326754289614`
- f2: test_mean=`0.6657798343249508`, test_std=`0.04265001678012268`

**Evaluacion test (si existe)**
- accuracy: `0.3511450381679389`
- precision: `0.32231404958677684`
- recall: `0.9285714285714286`
- specificity: `0.07865168539325842`
- f1_score: `0.4785276073619632`
- roc_auc: `0.5795545746388443`
- average_precision: `0.42333692559725467`
- optimal_threshold: `0.6333955729639996`

**Artefactos detectados**
- metadata: `models/v1/logistic_regression/metadata.json`
- modelo serializado: `models/v1/logistic_regression/model.joblib`
- evaluacion: `output/v1/evaluation/logistic_regression/evaluation.json`
- threshold_comparison_json: `N/A`
- threshold_comparison_csv: `N/A`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "C": 0.5,
  "class_weight": {
    "0": 1,
    "1": 4
  },
  "dual": false,
  "fit_intercept": true,
  "intercept_scaling": 1,
  "l1_ratio": null,
  "max_iter": 2000,
  "multi_class": "deprecated",
  "n_jobs": null,
  "penalty": "l2",
  "random_state": 42,
  "solver": "lbfgs",
  "tol": 0.0001,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 3. v1 / random_forest

**Identificacion**
- Version (ruta): `v1`
- Modelo: `random_forest`
- Timestamp entrenamiento: `2026-01-26T00:29:40.656614`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.4985645933014354`, test_std=`0.027302091134329392`
- precision: test_mean=`0.3517999066661678`, test_std=`0.027768370213214964`
- recall: test_mean=`0.6776119402985074`, test_std=`0.08784441175858008`
- f1: test_mean=`0.46266067296074054`, test_std=`0.04420999959039783`
- roc_auc: test_mean=`0.5783161656506202`, test_std=`0.0618416530925756`
- f2: test_mean=`0.571142959817261`, test_std=`0.06436103455762103`

**Evaluacion test (si existe)**
- accuracy: `0.4541984732824427`
- precision: `0.33519553072625696`
- recall: `0.7142857142857143`
- specificity: `0.33146067415730335`
- f1_score: `0.45627376425855515`
- roc_auc: `0.5320692883895131`
- average_precision: `0.38812018054327757`
- optimal_threshold: `0.5379136483722522`

**Artefactos detectados**
- metadata: `models/v1/random_forest/metadata.json`
- modelo serializado: `models/v1/random_forest/model.joblib`
- evaluacion: `output/v1/evaluation/random_forest/evaluation.json`
- threshold_comparison_json: `N/A`
- threshold_comparison_csv: `N/A`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "bootstrap": true,
  "ccp_alpha": 0.0,
  "class_weight": {
    "0": 1,
    "1": 4
  },
  "criterion": "gini",
  "max_depth": 12,
  "max_features": "sqrt",
  "max_leaf_nodes": null,
  "max_samples": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": null,
  "n_estimators": 200,
  "n_jobs": -1,
  "oob_score": true,
  "random_state": 42,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 4. v10 / gradient_boosting

**Identificacion**
- Version (ruta): `v10`
- Modelo: `gradient_boosting`
- Timestamp entrenamiento: `2026-01-26T05:04:02.506286`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.6296650717703349`, test_std=`0.00937603729294997`
- precision: test_mean=`0.39732467118736625`, test_std=`0.021468286096553733`
- recall: test_mean=`0.31044776119402984`, test_std=`0.07976351770679038`
- f1: test_mean=`0.34451581089567596`, test_std=`0.054077409794804375`
- roc_auc: test_mean=`0.5706327517342864`, test_std=`0.022047018463613382`
- f2: test_mean=`0.3224822955190556`, test_std=`0.07025739709939542`

**Evaluacion test (si existe)**
- accuracy: `0.5954198473282443`
- precision: `0.3225806451612903`
- recall: `0.23809523809523808`
- specificity: `0.7640449438202247`
- f1_score: `0.273972602739726`
- roc_auc: `0.5433052434456929`
- average_precision: `0.34562096038327617`
- optimal_threshold: `0.2229027321324103`
- optimal_threshold_results:
  - threshold: `0.3900000000000002`
  - accuracy: `0.5419847328244275`
  - precision: `0.3125`
  - recall: `0.35714285714285715`
  - specificity: `0.6292134831460674`
  - f1_score: `0.3333333333333333`

**Artefactos detectados**
- metadata: `models/v10/gradient_boosting/metadata.json`
- modelo serializado: `models/v10/gradient_boosting/model.joblib`
- evaluacion: `output/v10/evaluation/gradient_boosting/evaluation.json`
- threshold_comparison_json: `output/v10/threshold_analysis/gradient_boosting/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v10/threshold_analysis/gradient_boosting/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "ccp_alpha": 0.0,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.08,
  "loss": "log_loss",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 200,
  "n_iter_no_change": null,
  "random_state": 42,
  "subsample": 0.8,
  "tol": 0.0001,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 5. v10 / knn

**Identificacion**
- Version (ruta): `v10`
- Modelo: `knn`
- Timestamp entrenamiento: `2026-01-26T05:02:07.102919`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.6382775119617226`, test_std=`0.03613628246602722`
- precision: test_mean=`0.4217459272266121`, test_std=`0.06944386642009977`
- recall: test_mean=`0.32238805970149254`, test_std=`0.0504821926127993`
- f1: test_mean=`0.3635573029682622`, test_std=`0.0508567912467137`
- roc_auc: test_mean=`0.5431364305234392`, test_std=`0.04599229826695984`
- f2: test_mean=`0.3373270575925243`, test_std=`0.04921723284862706`

**Evaluacion test (si existe)**
- accuracy: `0.5610687022900763`
- precision: `0.28169014084507044`
- recall: `0.23809523809523808`
- specificity: `0.7134831460674157`
- f1_score: `0.25806451612903225`
- roc_auc: `0.4922752808988764`
- average_precision: `0.3079505785243247`
- optimal_threshold: `0.22449100957806606`
- optimal_threshold_results:
  - threshold: `0.5100000000000002`
  - accuracy: `0.5648854961832062`
  - precision: `0.27941176470588236`
  - recall: `0.2261904761904762`
  - specificity: `0.7247191011235955`
  - f1_score: `0.25`

**Artefactos detectados**
- metadata: `models/v10/knn/metadata.json`
- modelo serializado: `models/v10/knn/model.joblib`
- evaluacion: `output/v10/evaluation/knn/evaluation.json`
- threshold_comparison_json: `output/v10/threshold_analysis/knn/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v10/threshold_analysis/knn/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "algorithm": "auto",
  "leaf_size": 30,
  "metric": "minkowski",
  "metric_params": null,
  "n_jobs": null,
  "n_neighbors": 5,
  "p": 2,
  "weights": "distance"
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 6. v10 / lightgbm

**Identificacion**
- Version (ruta): `v10`
- Modelo: `lightgbm`
- Timestamp entrenamiento: `2026-02-19T15:32:54.160826`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `False`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.4842105263157895`, test_std=`0.029245371846767697`
- precision: test_mean=`0.3534289395519322`, test_std=`0.015062235326003613`
- recall: test_mean=`0.7313432835820896`, test_std=`0.05895050047203433`
- f1: test_mean=`0.47583007520626114`, test_std=`0.020058529168568092`
- roc_auc: test_mean=`0.5743956274963213`, test_std=`0.033804712457453895`
- f2: test_mean=`0.6015550653288135`, test_std=`0.034551408043728395`

**Evaluacion test (si existe)**
- accuracy: `0.5305343511450382`
- precision: `0.26506024096385544`
- recall: `0.2619047619047619`
- specificity: `0.6573033707865169`
- f1_score: `0.2634730538922156`
- roc_auc: `0.5076578384162653`
- average_precision: `0.32598383986692636`
- optimal_threshold: `0.04173238808347247`
- optimal_threshold_results:
  - threshold: `0.5400000000000003`
  - accuracy: `0.5572519083969466`
  - precision: `0.2894736842105263`
  - recall: `0.2619047619047619`
  - specificity: `0.6966292134831461`
  - f1_score: `0.275`

**Artefactos detectados**
- metadata: `models/v10/lightgbm/metadata.json`
- modelo serializado: `models/v10/lightgbm/model.joblib`
- evaluacion: `output/v10/evaluation/lightgbm/evaluation.json`
- threshold_comparison_json: `output/v10/threshold_analysis/lightgbm/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v10/threshold_analysis/lightgbm/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "boosting_type": "gbdt",
  "class_weight": null,
  "colsample_bytree": 0.6,
  "importance_type": "split",
  "learning_rate": 0.03,
  "max_depth": 5,
  "min_child_samples": 20,
  "min_child_weight": 0.001,
  "min_split_gain": 0.0,
  "n_estimators": 200,
  "n_jobs": null,
  "num_leaves": 63,
  "objective": null,
  "random_state": 42,
  "reg_alpha": 0.1,
  "reg_lambda": 5.0,
  "subsample": 0.6,
  "subsample_for_bin": 200000,
  "subsample_freq": 0,
  "scale_pos_weight": 3.0
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 7. v10 / logistic_regression

**Identificacion**
- Version (ruta): `v10`
- Modelo: `logistic_regression`
- Timestamp entrenamiento: `2026-01-26T05:04:00.812820`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.3177033492822966`, test_std=`0.004879444510615098`
- precision: test_mean=`0.3179175117290245`, test_std=`0.0024156938496571385`
- recall: test_mean=`0.9850746268656716`, test_std=`0.009439634806472766`
- f1: test_mean=`0.4806952569275641`, test_std=`0.003769685845331449`
- roc_auc: test_mean=`0.5299138112255622`, test_std=`0.06424330442862414`
- f2: test_mean=`0.6938547518949338`, test_std=`0.005841489090937644`

**Evaluacion test (si existe)**
- accuracy: `0.32061068702290074`
- precision: `0.32061068702290074`
- recall: `1.0`
- specificity: `0.0`
- f1_score: `0.48554913294797686`
- roc_auc: `0.5897873194221509`
- average_precision: `0.44413596790491916`
- optimal_threshold: `0.5774600970951057`
- optimal_threshold_results:
  - threshold: `0.5400000000000003`
  - accuracy: `0.3473282442748092`
  - precision: `0.32388663967611336`
  - recall: `0.9523809523809523`
  - specificity: `0.06179775280898876`
  - f1_score: `0.48338368580060426`

**Artefactos detectados**
- metadata: `models/v10/logistic_regression/metadata.json`
- modelo serializado: `models/v10/logistic_regression/model.joblib`
- evaluacion: `output/v10/evaluation/logistic_regression/evaluation.json`
- threshold_comparison_json: `output/v10/threshold_analysis/logistic_regression/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v10/threshold_analysis/logistic_regression/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "C": 0.001,
  "class_weight": {
    "0": 1,
    "1": 3
  },
  "dual": false,
  "fit_intercept": true,
  "intercept_scaling": 1,
  "l1_ratio": null,
  "max_iter": 100,
  "multi_class": "deprecated",
  "n_jobs": null,
  "penalty": "l2",
  "random_state": 42,
  "solver": "lbfgs",
  "tol": 0.0001,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 8. v10 / random_forest

**Identificacion**
- Version (ruta): `v10`
- Modelo: `random_forest`
- Timestamp entrenamiento: `2026-01-26T05:08:56.696539`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.34258373205741627`, test_std=`0.008342390322565885`
- precision: test_mean=`0.3257155396186537`, test_std=`0.005018095136924237`
- recall: test_mean=`0.982089552238806`, test_std=`0.021935729039849364`
- f1: test_mean=`0.4891799133755302`, test_std=`0.008251288814956268`
- roc_auc: test_mean=`0.5664073996216104`, test_std=`0.057331460940250725`
- f2: test_mean=`0.6999582931518049`, test_std=`0.013334650214097047`

**Evaluacion test (si existe)**
- accuracy: `0.33969465648854963`
- precision: `0.3241106719367589`
- recall: `0.9761904761904762`
- specificity: `0.03932584269662921`
- f1_score: `0.486646884272997`
- roc_auc: `0.5676498127340824`
- average_precision: `0.3968438786510774`
- optimal_threshold: `0.6193841192689028`
- optimal_threshold_results:
  - threshold: `0.5400000000000003`
  - accuracy: `0.3511450381679389`
  - precision: `0.3252032520325203`
  - recall: `0.9523809523809523`
  - specificity: `0.06741573033707865`
  - f1_score: `0.48484848484848486`

**Artefactos detectados**
- metadata: `models/v10/random_forest/metadata.json`
- modelo serializado: `models/v10/random_forest/model.joblib`
- evaluacion: `output/v10/evaluation/random_forest/evaluation.json`
- threshold_comparison_json: `output/v10/threshold_analysis/random_forest/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v10/threshold_analysis/random_forest/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "bootstrap": true,
  "ccp_alpha": 0.0,
  "class_weight": {
    "0": 1,
    "1": 4.2
  },
  "criterion": "gini",
  "max_depth": 10,
  "max_features": "sqrt",
  "max_leaf_nodes": null,
  "max_samples": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 6,
  "min_samples_split": 6,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": null,
  "n_estimators": 500,
  "n_jobs": null,
  "oob_score": false,
  "random_state": 42,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 9. v11 / lightgbm

**Identificacion**
- Version (ruta): `v11`
- Modelo: `lightgbm`
- Timestamp entrenamiento: `2026-02-19T20:36:44.256309`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `False`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.4842105263157895`, test_std=`0.029245371846767697`
- precision: test_mean=`0.3534289395519322`, test_std=`0.015062235326003613`
- recall: test_mean=`0.7313432835820896`, test_std=`0.05895050047203433`
- f1: test_mean=`0.47583007520626114`, test_std=`0.020058529168568092`
- roc_auc: test_mean=`0.5743956274963213`, test_std=`0.033804712457453895`
- f2: test_mean=`0.6015550653288135`, test_std=`0.034551408043728395`

**Evaluacion test (si existe)**
- accuracy: `0.5305343511450382`
- precision: `0.26506024096385544`
- recall: `0.2619047619047619`
- specificity: `0.6573033707865169`
- f1_score: `0.2634730538922156`
- roc_auc: `0.5076578384162653`
- average_precision: `0.32598383986692636`
- optimal_threshold: `0.04173238808347247`
- optimal_threshold_results:
  - threshold: `0.5400000000000003`
  - accuracy: `0.5572519083969466`
  - precision: `0.2894736842105263`
  - recall: `0.2619047619047619`
  - specificity: `0.6966292134831461`
  - f1_score: `0.275`

**Artefactos detectados**
- metadata: `models/v11/lightgbm/metadata.json`
- modelo serializado: `models/v11/lightgbm/model.joblib`
- evaluacion: `output/v11/evaluation/lightgbm/evaluation.json`
- threshold_comparison_json: `output/v11/threshold_analysis/lightgbm/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v11/threshold_analysis/lightgbm/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "boosting_type": "gbdt",
  "class_weight": null,
  "colsample_bytree": 0.6,
  "importance_type": "split",
  "learning_rate": 0.03,
  "max_depth": 5,
  "min_child_samples": 20,
  "min_child_weight": 0.001,
  "min_split_gain": 0.0,
  "n_estimators": 200,
  "n_jobs": -1,
  "num_leaves": 63,
  "objective": "binary",
  "random_state": 42,
  "reg_alpha": 0.1,
  "reg_lambda": 5.0,
  "subsample": 0.6,
  "subsample_for_bin": 200000,
  "subsample_freq": 0,
  "scale_pos_weight": 3.0
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 10. v17 / decision_tree

**Identificacion**
- Version (ruta): `v17`
- Modelo: `decision_tree`
- Timestamp entrenamiento: `2026-03-05T19:03:14.432007`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `19`
- scale_features: `False`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `10`
- accuracy: test_mean=`0.545448717948718`, test_std=`0.07520975969433709`
- precision: test_mean=`0.379030709025007`, test_std=`0.0465599853828369`
- recall: test_mean=`0.5934046345811053`, test_std=`0.1572212723343332`
- f1: test_mean=`0.4495288320471517`, test_std=`0.056251440711334186`
- roc_auc: test_mean=`0.6110278426351334`, test_std=`0.07002451576106196`
- f2: test_mean=`0.521374465431105`, test_std=`0.10247020251596307`
- kappa: test_mean=`0.10342412614961863`, test_std=`0.07340903198287023`

**Evaluacion test (si existe)**
- accuracy: `0.5343511450381679`
- precision: `0.38953488372093026`
- recall: `0.7976190476190477`
- specificity: `0.4101123595505618`
- f1_score: `0.5234375`
- kappa: `0.16272003352891862`
- roc_auc: `0.6863964686998395`
- average_precision: `0.5016665954990465`
- optimal_threshold: `0.5231154064809496`
- optimal_threshold_results:
  - threshold: `0.3300000000000001`
  - accuracy: `0.4541984732824427`
  - precision: `0.3588516746411483`
  - recall: `0.8928571428571429`
  - specificity: `0.24719101123595505`
  - f1_score: `0.5119453924914675`
  - kappa: `0.10054256493974156`

**Artefactos detectados**
- metadata: `models/v17/decision_tree/metadata.json`
- modelo serializado: `models/v17/decision_tree/model.joblib`
- evaluacion: `output/v17/evaluation/decision_tree/evaluation.json`
- threshold_comparison_json: `output/v17/threshold_analysis/decision_tree/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v17/threshold_analysis/decision_tree/threshold_comparison_test.csv`
- hyperparams_best: `output/v17/hyperparameter_search/decision_tree/best_params.json`
- hyperparams_search: `output/v17/hyperparameter_search/decision_tree/search_results.json`

**Parametros del modelo (metadata.model_params)**
```json
{
  "ccp_alpha": 0.0,
  "class_weight": "balanced",
  "criterion": "gini",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 10,
  "min_samples_split": 20,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": null,
  "random_state": 42,
  "splitter": "best"
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 11. v17 / lightgbm

**Identificacion**
- Version (ruta): `v17`
- Modelo: `lightgbm`
- Timestamp entrenamiento: `2026-03-05T19:03:16.247979`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `19`
- scale_features: `False`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `10`
- accuracy: test_mean=`0.4947069597069597`, test_std=`0.04361010851887871`
- precision: test_mean=`0.36264563667187477`, test_std=`0.025227110379199977`
- recall: test_mean=`0.7523172905525847`, test_std=`0.05928283117412107`
- f1: test_mean=`0.48854854403886294`, test_std=`0.02925477987534526`
- roc_auc: test_mean=`0.6109079611357988`, test_std=`0.04640007338363833`
- f2: test_mean=`0.6181345890254023`, test_std=`0.03880407405522406`
- kappa: test_mean=`0.0986411715482976`, test_std=`0.056069493803338186`

**Evaluacion test (si existe)**
- accuracy: `0.48091603053435117`
- precision: `0.3617021276595745`
- recall: `0.8095238095238095`
- specificity: `0.3258426966292135`
- f1_score: `0.5`
- kappa: `0.1020161290322581`
- roc_auc: `0.6527220438737292`
- average_precision: `0.5602158888131079`
- optimal_threshold: `0.5363682087234453`
- optimal_threshold_results:
  - threshold: `0.5000000000000002`
  - accuracy: `0.48091603053435117`
  - precision: `0.3617021276595745`
  - recall: `0.8095238095238095`
  - specificity: `0.3258426966292135`
  - f1_score: `0.5`
  - kappa: `0.1020161290322581`

**Artefactos detectados**
- metadata: `models/v17/lightgbm/metadata.json`
- modelo serializado: `models/v17/lightgbm/model.joblib`
- evaluacion: `output/v17/evaluation/lightgbm/evaluation.json`
- threshold_comparison_json: `output/v17/threshold_analysis/lightgbm/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v17/threshold_analysis/lightgbm/threshold_comparison_test.csv`
- hyperparams_best: `output/v17/hyperparameter_search/lightgbm/best_params.json`
- hyperparams_search: `output/v17/hyperparameter_search/lightgbm/search_results.json`

**Parametros del modelo (metadata.model_params)**
```json
{
  "boosting_type": "gbdt",
  "class_weight": null,
  "colsample_bytree": 0.6,
  "importance_type": "split",
  "learning_rate": 0.03,
  "max_depth": 5,
  "min_child_samples": 20,
  "min_child_weight": 0.001,
  "min_split_gain": 0.0,
  "n_estimators": 200,
  "n_jobs": -1,
  "num_leaves": 63,
  "objective": "binary",
  "random_state": 42,
  "reg_alpha": 0.1,
  "reg_lambda": 5.0,
  "subsample": 0.6,
  "subsample_for_bin": 200000,
  "subsample_freq": 0,
  "scale_pos_weight": 3.0
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 12. v18 / decision_tree

**Identificacion**
- Version (ruta): `v18`
- Modelo: `decision_tree`
- Timestamp entrenamiento: `2026-03-07T00:24:32.863785`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `False`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `10`
- accuracy: test_mean=`0.5243864468864469`, test_std=`0.05051843573275332`
- precision: test_mean=`0.3634801538115768`, test_std=`0.03344642864610559`
- recall: test_mean=`0.6480392156862745`, test_std=`0.149742938486688`
- f1: test_mean=`0.46114466118878994`, test_std=`0.05892737019745163`
- roc_auc: test_mean=`0.5647309884261003`, test_std=`0.05847467713769571`
- f2: test_mean=`0.5553264526372628`, test_std=`0.09959244286713469`
- kappa: test_mean=`0.09329721980666966`, test_std=`0.07765230304670842`

**Evaluacion test (si existe)**
- accuracy: `0.5343511450381679`
- precision: `0.38823529411764707`
- recall: `0.7857142857142857`
- specificity: `0.4157303370786517`
- f1_score: `0.5196850393700787`
- kappa: `0.1585763925450141`
- roc_auc: `0.6428237025147137`
- average_precision: `0.448253200249485`
- optimal_threshold: `0.5455677183945001`
- optimal_threshold_results:
  - threshold: `0.3900000000000002`
  - accuracy: `0.46564885496183206`
  - precision: `0.3613861386138614`
  - recall: `0.8690476190476191`
  - specificity: `0.2752808988764045`
  - f1_score: `0.5104895104895105`
  - kappa: `0.10527856376231826`

**Artefactos detectados**
- metadata: `models/v18/decision_tree/metadata.json`
- modelo serializado: `models/v18/decision_tree/model.joblib`
- evaluacion: `output/v18/evaluation/decision_tree/evaluation.json`
- threshold_comparison_json: `output/v18/threshold_analysis/decision_tree/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v18/threshold_analysis/decision_tree/threshold_comparison_test.csv`
- hyperparams_best: `output/v18/hyperparameter_search/decision_tree/best_params.json`
- hyperparams_search: `output/v18/hyperparameter_search/decision_tree/search_results.json`

**Parametros del modelo (metadata.model_params)**
```json
{
  "ccp_alpha": 0.0,
  "class_weight": "balanced",
  "criterion": "gini",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 10,
  "min_samples_split": 20,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": null,
  "random_state": 42,
  "splitter": "best"
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 13. v18 / lightgbm

**Identificacion**
- Version (ruta): `v18`
- Modelo: `lightgbm`
- Timestamp entrenamiento: `2026-03-07T00:24:34.820254`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `False`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `10`
- accuracy: test_mean=`0.48894688644688644`, test_std=`0.04534992425006361`
- precision: test_mean=`0.3606923497844602`, test_std=`0.028946965165665913`
- recall: test_mean=`0.7639928698752227`, test_std=`0.07422888818276556`
- f1: test_mean=`0.4892857082740405`, test_std=`0.037066780490527605`
- roc_auc: test_mean=`0.5822920589490599`, test_std=`0.05361467035177729`
- f2: test_mean=`0.6233632352819969`, test_std=`0.050719839733540946`
- kappa: test_mean=`0.09562816445173457`, test_std=`0.06604312583756461`

**Evaluacion test (si existe)**
- accuracy: `0.48091603053435117`
- precision: `0.3602150537634409`
- recall: `0.7976190476190477`
- specificity: `0.33146067415730335`
- f1_score: `0.4962962962962963`
- kappa: `0.09774131469664749`
- roc_auc: `0.5946027287319422`
- average_precision: `0.42960474066280957`
- optimal_threshold: `0.5601162442982074`
- optimal_threshold_results:
  - threshold: `0.49000000000000027`
  - accuracy: `0.4732824427480916`
  - precision: `0.35638297872340424`
  - recall: `0.7976190476190477`
  - specificity: `0.3202247191011236`
  - f1_score: `0.49264705882352944`
  - kappa: `0.0888104838709678`

**Artefactos detectados**
- metadata: `models/v18/lightgbm/metadata.json`
- modelo serializado: `models/v18/lightgbm/model.joblib`
- evaluacion: `output/v18/evaluation/lightgbm/evaluation.json`
- threshold_comparison_json: `output/v18/threshold_analysis/lightgbm/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v18/threshold_analysis/lightgbm/threshold_comparison_test.csv`
- hyperparams_best: `output/v18/hyperparameter_search/lightgbm/best_params.json`
- hyperparams_search: `output/v18/hyperparameter_search/lightgbm/search_results.json`

**Parametros del modelo (metadata.model_params)**
```json
{
  "boosting_type": "gbdt",
  "class_weight": null,
  "colsample_bytree": 0.6,
  "importance_type": "split",
  "learning_rate": 0.03,
  "max_depth": 5,
  "min_child_samples": 20,
  "min_child_weight": 0.001,
  "min_split_gain": 0.0,
  "n_estimators": 200,
  "n_jobs": -1,
  "num_leaves": 63,
  "objective": "binary",
  "random_state": 42,
  "reg_alpha": 0.1,
  "reg_lambda": 5.0,
  "subsample": 0.6,
  "subsample_for_bin": 200000,
  "subsample_freq": 0,
  "scale_pos_weight": 3.0
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 14. v19 / decision_tree

**Identificacion**
- Version (ruta): `v19`
- Modelo: `decision_tree`
- Timestamp entrenamiento: `2026-03-07T01:01:26.106726`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `19`
- scale_features: `False`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `10`
- accuracy: test_mean=`0.5158974358974358`, test_std=`0.059003903026788304`
- precision: test_mean=`0.35361059749245166`, test_std=`0.0300924506109857`
- recall: test_mean=`0.6254010695187167`, test_std=`0.18776791774811658`
- f1: test_mean=`0.4411443038082773`, test_std=`0.07843205218666366`
- roc_auc: test_mean=`0.5754556752278377`, test_std=`0.052229181978670436`
- f2: test_mean=`0.5322418984286763`, test_std=`0.1319072179527414`
- kappa: test_mean=`0.07210736395365348`, test_std=`0.06732773723462065`

**Evaluacion test (si existe)**
- accuracy: `0.5343511450381679`
- precision: `0.38953488372093026`
- recall: `0.7976190476190477`
- specificity: `0.4101123595505618`
- f1_score: `0.5234375`
- kappa: `0.16272003352891862`
- roc_auc: `0.6863964686998395`
- average_precision: `0.5016665954990465`
- optimal_threshold: `0.5231154064809496`
- optimal_threshold_results:
  - threshold: `0.49000000000000027`
  - accuracy: `0.5343511450381679`
  - precision: `0.38953488372093026`
  - recall: `0.7976190476190477`
  - specificity: `0.4101123595505618`
  - f1_score: `0.5234375`
  - kappa: `0.16272003352891862`

**Artefactos detectados**
- metadata: `models/v19/decision_tree/metadata.json`
- modelo serializado: `models/v19/decision_tree/model.joblib`
- evaluacion: `output/v19/evaluation/decision_tree/evaluation.json`
- threshold_comparison_json: `output/v19/threshold_analysis/decision_tree/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v19/threshold_analysis/decision_tree/threshold_comparison_test.csv`
- hyperparams_best: `output/v19/hyperparameter_search/decision_tree/best_params.json`
- hyperparams_search: `output/v19/hyperparameter_search/decision_tree/search_results.json`

**Parametros del modelo (metadata.model_params)**
```json
{
  "ccp_alpha": 0.0,
  "class_weight": "balanced",
  "criterion": "gini",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 10,
  "min_samples_split": 20,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": null,
  "random_state": 42,
  "splitter": "best"
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 15. v19 / lightgbm

**Identificacion**
- Version (ruta): `v19`
- Modelo: `lightgbm`
- Timestamp entrenamiento: `2026-03-07T01:02:17.126359`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `19`
- scale_features: `False`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `10`
- accuracy: test_mean=`0.42865384615384616`, test_std=`0.03593163061688299`
- precision: test_mean=`0.3488223413371123`, test_std=`0.023148749595823068`
- recall: test_mean=`0.9044563279857398`, test_std=`0.07528403959453087`
- f1: test_mean=`0.5032925527781018`, test_std=`0.03481164440518977`
- roc_auc: test_mean=`0.5970180261605282`, test_std=`0.06309507315775081`
- f2: test_mean=`0.6856473566970156`, test_std=`0.051114407437890225`
- kappa: test_mean=`0.07626074579458697`, test_std=`0.060148733219135116`

**Evaluacion test (si existe)**
- accuracy: `0.5877862595419847`
- precision: `0.3983050847457627`
- recall: `0.5595238095238095`
- specificity: `0.601123595505618`
- f1_score: `0.46534653465346537`
- kappa: `0.1451359516616314`
- roc_auc: `0.6481741573033708`
- average_precision: `0.5141315761412735`
- optimal_threshold: `0.3778651532727212`
- optimal_threshold_results:
  - threshold: `0.4100000000000002`
  - accuracy: `0.5267175572519084`
  - precision: `0.375`
  - recall: `0.7142857142857143`
  - specificity: `0.43820224719101125`
  - f1_score: `0.4918032786885246`
  - kappa: `0.12308356726408987`

**Artefactos detectados**
- metadata: `models/v19/lightgbm/metadata.json`
- modelo serializado: `models/v19/lightgbm/model.joblib`
- evaluacion: `output/v19/evaluation/lightgbm/evaluation.json`
- threshold_comparison_json: `output/v19/threshold_analysis/lightgbm/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v19/threshold_analysis/lightgbm/threshold_comparison_test.csv`
- hyperparams_best: `output/v19/hyperparameter_search/lightgbm/best_params.json`
- hyperparams_search: `output/v19/hyperparameter_search/lightgbm/search_results.json`

**Parametros del modelo (metadata.model_params)**
```json
{
  "boosting_type": "gbdt",
  "class_weight": null,
  "colsample_bytree": 0.6,
  "importance_type": "split",
  "learning_rate": 0.03,
  "max_depth": 5,
  "min_child_samples": 20,
  "min_child_weight": 0.001,
  "min_split_gain": 0.0,
  "n_estimators": 200,
  "n_jobs": -1,
  "num_leaves": 63,
  "objective": "binary",
  "random_state": 42,
  "reg_alpha": 0.1,
  "reg_lambda": 5.0,
  "subsample": 0.6,
  "subsample_for_bin": 200000,
  "subsample_freq": 0,
  "scale_pos_weight": 3.0
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 16. v19 / logistic_regression

**Identificacion**
- Version (ruta): `v19`
- Modelo: `logistic_regression`
- Timestamp entrenamiento: `2026-03-07T01:01:55.497594`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `19`
- scale_features: `False`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `10`
- accuracy: test_mean=`0.575091575091575`, test_std=`0.05196994157441832`
- precision: test_mean=`0.393665809595534`, test_std=`0.05019301487791682`
- recall: test_mean=`0.5823529411764705`, test_std=`0.059175203290454004`
- f1: test_mean=`0.46862516818494104`, test_std=`0.049554552576220094`
- roc_auc: test_mean=`0.6003194747809496`, test_std=`0.05733683769590025`
- f2: test_mean=`0.5303376849497702`, test_std=`0.052226928594787156`
- kappa: test_mean=`0.13804304318614688`, test_std=`0.0886881423702224`

**Evaluacion test (si existe)**
- accuracy: `0.6221374045801527`
- precision: `0.42574257425742573`
- recall: `0.5119047619047619`
- specificity: `0.6741573033707865`
- f1_score: `0.4648648648648649`
- kappa: `0.17662370643133762`
- roc_auc: `0.6532905296950241`
- average_precision: `0.5459334066588967`
- optimal_threshold: `0.4470874313472836`
- optimal_threshold_results:
  - threshold: `0.4300000000000002`
  - accuracy: `0.5`
  - precision: `0.37566137566137564`
  - recall: `0.8452380952380952`
  - specificity: `0.33707865168539325`
  - f1_score: `0.5201465201465202`
  - kappa: `0.13707447075979295`

**Artefactos detectados**
- metadata: `models/v19/logistic_regression/metadata.json`
- modelo serializado: `models/v19/logistic_regression/model.joblib`
- evaluacion: `output/v19/evaluation/logistic_regression/evaluation.json`
- threshold_comparison_json: `output/v19/threshold_analysis/logistic_regression/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v19/threshold_analysis/logistic_regression/threshold_comparison_test.csv`
- hyperparams_best: `output/v19/hyperparameter_search/logistic_regression/best_params.json`
- hyperparams_search: `output/v19/hyperparameter_search/logistic_regression/search_results.json`

**Parametros del modelo (metadata.model_params)**
```json
{
  "C": 1.0,
  "class_weight": "balanced",
  "dual": false,
  "fit_intercept": true,
  "intercept_scaling": 1,
  "l1_ratio": null,
  "max_iter": 1000,
  "multi_class": "deprecated",
  "n_jobs": -1,
  "penalty": "l2",
  "random_state": 42,
  "solver": "saga",
  "tol": 0.0001,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 17. v19 / naive_bayes

**Identificacion**
- Version (ruta): `v19`
- Modelo: `naive_bayes`
- Timestamp entrenamiento: `2026-03-07T01:01:36.882471`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `19`
- scale_features: `False`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `10`
- accuracy: test_mean=`0.3913644688644689`, test_std=`0.038375275013292055`
- precision: test_mean=`0.3230121450200148`, test_std=`0.024568469571563598`
- recall: test_mean=`0.8208556149732621`, test_std=`0.06694348051649981`
- f1: test_mean=`0.46355521125357957`, test_std=`0.03571536227286413`
- roc_auc: test_mean=`0.5890644221837261`, test_std=`0.06671315444134128`
- f2: test_mean=`0.6273802724828089`, test_std=`0.04941919452327344`
- kappa: test_mean=`0.006697092754146772`, test_std=`0.06414396558094067`

**Evaluacion test (si existe)**
- accuracy: `0.7251908396946565`
- precision: `0.75`
- recall: `0.21428571428571427`
- specificity: `0.9662921348314607`
- f1_score: `0.3333333333333333`
- kappa: `0.22255192878338292`
- roc_auc: `0.6260700909577315`
- average_precision: `0.527789310409628`
- optimal_threshold: `0.31410114491603164`
- optimal_threshold_results:
  - threshold: `0.3000000000000001`
  - accuracy: `0.5038167938931297`
  - precision: `0.3707865168539326`
  - recall: `0.7857142857142857`
  - specificity: `0.3707865168539326`
  - f1_score: `0.5038167938931297`
  - kappa: `0.12080536912751683`

**Artefactos detectados**
- metadata: `models/v19/naive_bayes/metadata.json`
- modelo serializado: `models/v19/naive_bayes/model.joblib`
- evaluacion: `output/v19/evaluation/naive_bayes/evaluation.json`
- threshold_comparison_json: `output/v19/threshold_analysis/naive_bayes/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v19/threshold_analysis/naive_bayes/threshold_comparison_test.csv`
- hyperparams_best: `output/v19/hyperparameter_search/naive_bayes/best_params.json`
- hyperparams_search: `output/v19/hyperparameter_search/naive_bayes/search_results.json`

**Parametros del modelo (metadata.model_params)**
```json
{
  "priors": null,
  "var_smoothing": 1e-09
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 18. v2 / gradient_boosting

**Identificacion**
- Version (ruta): `v2`
- Modelo: `gradient_boosting`
- Timestamp entrenamiento: `2026-01-26T00:32:09.158886`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `20`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.6181818181818182`, test_std=`0.025390429017074837`
- precision: test_mean=`0.3717480777928539`, test_std=`0.03765530351374934`
- recall: test_mean=`0.2656716417910448`, test_std=`0.05772859583824991`
- f1: test_mean=`0.3056680503391659`, test_std=`0.037974748978741764`
- roc_auc: test_mean=`0.57280849274753`, test_std=`0.029821227340131994`
- f2: test_mean=`0.2796771681917658`, test_std=`0.050440262828299734`

**Evaluacion test (si existe)**
- accuracy: `0.6106870229007634`
- precision: `0.3448275862068966`
- recall: `0.23809523809523808`
- specificity: `0.7865168539325843`
- f1_score: `0.28169014084507044`
- roc_auc: `0.5303303905831995`
- average_precision: `0.37150428760372467`
- optimal_threshold: `0.2541566528289464`

**Artefactos detectados**
- metadata: `models/v2/gradient_boosting/metadata.json`
- modelo serializado: `models/v2/gradient_boosting/model.joblib`
- evaluacion: `output/v2/evaluation/gradient_boosting/evaluation.json`
- threshold_comparison_json: `N/A`
- threshold_comparison_csv: `N/A`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "ccp_alpha": 0.0,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.08,
  "loss": "log_loss",
  "max_depth": 6,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 4,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 150,
  "n_iter_no_change": null,
  "random_state": 42,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 19. v2 / logistic_regression

**Identificacion**
- Version (ruta): `v2`
- Modelo: `logistic_regression`
- Timestamp entrenamiento: `2026-01-26T00:32:15.529249`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `20`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.34258373205741627`, test_std=`0.021312016699827834`
- precision: test_mean=`0.316034474009088`, test_std=`0.018014288622065937`
- recall: test_mean=`0.9074626865671641`, test_std=`0.0841134495737634`
- f1: test_mean=`0.468653840518739`, test_std=`0.030917201172343145`
- roc_auc: test_mean=`0.5263821736388479`, test_std=`0.06732057378563293`
- f2: test_mean=`0.6600527813640441`, test_std=`0.05116634930176238`

**Evaluacion test (si existe)**
- accuracy: `0.3473282442748092`
- precision: `0.32098765432098764`
- recall: `0.9285714285714286`
- specificity: `0.07303370786516854`
- f1_score: `0.47706422018348627`
- roc_auc: `0.5798220973782772`
- average_precision: `0.41958916744799524`
- optimal_threshold: `0.6312084341522401`

**Artefactos detectados**
- metadata: `models/v2/logistic_regression/metadata.json`
- modelo serializado: `models/v2/logistic_regression/model.joblib`
- evaluacion: `output/v2/evaluation/logistic_regression/evaluation.json`
- threshold_comparison_json: `N/A`
- threshold_comparison_csv: `N/A`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "C": 0.5,
  "class_weight": {
    "0": 1,
    "1": 4
  },
  "dual": false,
  "fit_intercept": true,
  "intercept_scaling": 1,
  "l1_ratio": null,
  "max_iter": 2000,
  "multi_class": "deprecated",
  "n_jobs": null,
  "penalty": "l2",
  "random_state": 42,
  "solver": "lbfgs",
  "tol": 0.0001,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 20. v2 / random_forest

**Identificacion**
- Version (ruta): `v2`
- Modelo: `random_forest`
- Timestamp entrenamiento: `2026-01-26T00:32:10.615334`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `20`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.5435406698564593`, test_std=`0.03267630828509833`
- precision: test_mean=`0.3754978987052316`, test_std=`0.029242528044247415`
- recall: test_mean=`0.6417910447761194`, test_std=`0.08065227514093312`
- f1: test_mean=`0.4728242408142028`, test_std=`0.04219631513459696`
- roc_auc: test_mean=`0.5836661761614463`, test_std=`0.04788025699102089`
- f2: test_mean=`0.5610252401060629`, test_std=`0.059626677698900406`

**Evaluacion test (si existe)**
- accuracy: `0.48091603053435117`
- precision: `0.33544303797468356`
- recall: `0.6309523809523809`
- specificity: `0.4101123595505618`
- f1_score: `0.4380165289256198`
- roc_auc: `0.5193619582664526`
- average_precision: `0.3597511351161173`
- optimal_threshold: `0.5058378658910544`

**Artefactos detectados**
- metadata: `models/v2/random_forest/metadata.json`
- modelo serializado: `models/v2/random_forest/model.joblib`
- evaluacion: `output/v2/evaluation/random_forest/evaluation.json`
- threshold_comparison_json: `N/A`
- threshold_comparison_csv: `N/A`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "bootstrap": true,
  "ccp_alpha": 0.0,
  "class_weight": {
    "0": 1,
    "1": 4
  },
  "criterion": "gini",
  "max_depth": 12,
  "max_features": "sqrt",
  "max_leaf_nodes": null,
  "max_samples": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": null,
  "n_estimators": 200,
  "n_jobs": -1,
  "oob_score": true,
  "random_state": 42,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 21. v3 / gradient_boosting

**Identificacion**
- Version (ruta): `v3`
- Modelo: `gradient_boosting`
- Timestamp entrenamiento: `2026-01-26T00:44:36.079907`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `22`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.6296650717703349`, test_std=`0.021096083909137416`
- precision: test_mean=`0.3947228888764861`, test_std=`0.03583895628553429`
- recall: test_mean=`0.2835820895522388`, test_std=`0.05970149253731344`
- f1: test_mean=`0.3263663763996471`, test_std=`0.04041158635653156`
- roc_auc: test_mean=`0.5691402144208535`, test_std=`0.019586963309761243`
- f2: test_mean=`0.2986303142376616`, test_std=`0.052316607136790906`

**Evaluacion test (si existe)**
- accuracy: `0.5954198473282443`
- precision: `0.2962962962962963`
- recall: `0.19047619047619047`
- specificity: `0.7865168539325843`
- f1_score: `0.2318840579710145`
- roc_auc: `0.5283239700374531`
- average_precision: `0.36106486674758576`
- optimal_threshold: `0.23492698011381408`

**Artefactos detectados**
- metadata: `models/v3/gradient_boosting/metadata.json`
- modelo serializado: `models/v3/gradient_boosting/model.joblib`
- evaluacion: `output/v3/evaluation/gradient_boosting/evaluation.json`
- threshold_comparison_json: `N/A`
- threshold_comparison_csv: `N/A`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "ccp_alpha": 0.0,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.08,
  "loss": "log_loss",
  "max_depth": 6,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 4,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 150,
  "n_iter_no_change": null,
  "random_state": 42,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 22. v3 / logistic_regression

**Identificacion**
- Version (ruta): `v3`
- Modelo: `logistic_regression`
- Timestamp entrenamiento: `2026-01-26T00:44:42.027163`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `22`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.3397129186602871`, test_std=`0.018407066087723762`
- precision: test_mean=`0.3167729147566062`, test_std=`0.014671609647684696`
- recall: test_mean=`0.9194029850746267`, test_std=`0.07164179104477611`
- f1: test_mean=`0.4710752580509851`, test_std=`0.025383740286125864`
- roc_auc: test_mean=`0.5297456380071475`, test_std=`0.07391414433987076`
- f2: test_mean=`0.6657798343249508`, test_std=`0.04265001678012268`

**Evaluacion test (si existe)**
- accuracy: `0.3511450381679389`
- precision: `0.32231404958677684`
- recall: `0.9285714285714286`
- specificity: `0.07865168539325842`
- f1_score: `0.4785276073619632`
- roc_auc: `0.5794876939539861`
- average_precision: `0.422290501106456`
- optimal_threshold: `0.6337479394477785`

**Artefactos detectados**
- metadata: `models/v3/logistic_regression/metadata.json`
- modelo serializado: `models/v3/logistic_regression/model.joblib`
- evaluacion: `output/v3/evaluation/logistic_regression/evaluation.json`
- threshold_comparison_json: `N/A`
- threshold_comparison_csv: `N/A`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "C": 0.5,
  "class_weight": {
    "0": 1,
    "1": 4
  },
  "dual": false,
  "fit_intercept": true,
  "intercept_scaling": 1,
  "l1_ratio": null,
  "max_iter": 2000,
  "multi_class": "deprecated",
  "n_jobs": null,
  "penalty": "l2",
  "random_state": 42,
  "solver": "lbfgs",
  "tol": 0.0001,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 23. v3 / random_forest

**Identificacion**
- Version (ruta): `v3`
- Modelo: `random_forest`
- Timestamp entrenamiento: `2026-01-26T00:44:37.440807`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `22`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.491866028708134`, test_std=`0.025925774515108264`
- precision: test_mean=`0.3458448966640558`, test_std=`0.027009316177962154`
- recall: test_mean=`0.6656716417910448`, test_std=`0.09654305450027602`
- f1: test_mean=`0.45449028360232974`, test_std=`0.04536752172334375`
- roc_auc: test_mean=`0.5727454277906243`, test_std=`0.061285412305804865`
- f2: test_mean=`0.5609378904415646`, test_std=`0.06873417323381496`

**Evaluacion test (si existe)**
- accuracy: `0.45038167938931295`
- precision: `0.33146067415730335`
- recall: `0.7023809523809523`
- specificity: `0.33146067415730335`
- f1_score: `0.45038167938931295`
- roc_auc: `0.5154159978598181`
- average_precision: `0.37147027591155457`
- optimal_threshold: `0.5286536080150994`

**Artefactos detectados**
- metadata: `models/v3/random_forest/metadata.json`
- modelo serializado: `models/v3/random_forest/model.joblib`
- evaluacion: `output/v3/evaluation/random_forest/evaluation.json`
- threshold_comparison_json: `N/A`
- threshold_comparison_csv: `N/A`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "bootstrap": true,
  "ccp_alpha": 0.0,
  "class_weight": {
    "0": 1,
    "1": 4
  },
  "criterion": "gini",
  "max_depth": 12,
  "max_features": "sqrt",
  "max_leaf_nodes": null,
  "max_samples": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": null,
  "n_estimators": 200,
  "n_jobs": -1,
  "oob_score": true,
  "random_state": 42,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 24. v4 / gradient_boosting

**Identificacion**
- Version (ruta): `v4`
- Modelo: `gradient_boosting`
- Timestamp entrenamiento: `2026-01-26T00:46:48.892292`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `24`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.6325358851674642`, test_std=`0.016408065262662436`
- precision: test_mean=`0.3998573717070592`, test_std=`0.036800278000090024`
- recall: test_mean=`0.2955223880597015`, test_std=`0.06833148144632595`
- f1: test_mean=`0.33665267579867075`, test_std=`0.049668874347912735`
- roc_auc: test_mean=`0.5721988648307756`, test_std=`0.02010082214825701`
- f2: test_mean=`0.3101178157421673`, test_std=`0.06119733812452479`

**Evaluacion test (si existe)**
- accuracy: `0.6030534351145038`
- precision: `0.32142857142857145`
- recall: `0.21428571428571427`
- specificity: `0.7865168539325843`
- f1_score: `0.2571428571428571`
- roc_auc: `0.5261837881219904`
- average_precision: `0.35918802923058923`
- optimal_threshold: `0.2333114005215314`

**Artefactos detectados**
- metadata: `models/v4/gradient_boosting/metadata.json`
- modelo serializado: `models/v4/gradient_boosting/model.joblib`
- evaluacion: `output/v4/evaluation/gradient_boosting/evaluation.json`
- threshold_comparison_json: `N/A`
- threshold_comparison_csv: `N/A`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "ccp_alpha": 0.0,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.08,
  "loss": "log_loss",
  "max_depth": 6,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 4,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 150,
  "n_iter_no_change": null,
  "random_state": 42,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 25. v4 / logistic_regression

**Identificacion**
- Version (ruta): `v4`
- Modelo: `logistic_regression`
- Timestamp entrenamiento: `2026-01-26T00:46:56.071690`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `24`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.34258373205741627`, test_std=`0.021312016699827834`
- precision: test_mean=`0.316034474009088`, test_std=`0.018014288622065937`
- recall: test_mean=`0.9074626865671641`, test_std=`0.0841134495737634`
- f1: test_mean=`0.468653840518739`, test_std=`0.030917201172343145`
- roc_auc: test_mean=`0.5264872819003574`, test_std=`0.06718051084399819`
- f2: test_mean=`0.6600527813640441`, test_std=`0.05116634930176238`

**Evaluacion test (si existe)**
- accuracy: `0.3473282442748092`
- precision: `0.32098765432098764`
- recall: `0.9285714285714286`
- specificity: `0.07303370786516854`
- f1_score: `0.47706422018348627`
- roc_auc: `0.5800227394328519`
- average_precision: `0.4192739614548335`
- optimal_threshold: `0.6312863617336899`

**Artefactos detectados**
- metadata: `models/v4/logistic_regression/metadata.json`
- modelo serializado: `models/v4/logistic_regression/model.joblib`
- evaluacion: `output/v4/evaluation/logistic_regression/evaluation.json`
- threshold_comparison_json: `N/A`
- threshold_comparison_csv: `N/A`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "C": 0.5,
  "class_weight": {
    "0": 1,
    "1": 4
  },
  "dual": false,
  "fit_intercept": true,
  "intercept_scaling": 1,
  "l1_ratio": null,
  "max_iter": 2000,
  "multi_class": "deprecated",
  "n_jobs": null,
  "penalty": "l2",
  "random_state": 42,
  "solver": "lbfgs",
  "tol": 0.0001,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 26. v4 / random_forest

**Identificacion**
- Version (ruta): `v4`
- Modelo: `random_forest`
- Timestamp entrenamiento: `2026-01-26T00:46:49.972987`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `24`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.5387559808612441`, test_std=`0.03562585735691156`
- precision: test_mean=`0.3733038073038073`, test_std=`0.032421063473135565`
- recall: test_mean=`0.6507462686567165`, test_std=`0.08368863145597723`
- f1: test_mean=`0.47367814076893194`, test_std=`0.04678836580546962`
- roc_auc: test_mean=`0.585820895522388`, test_std=`0.053024570896073056`
- f2: test_mean=`0.5657128740824392`, test_std=`0.06412299250609102`

**Evaluacion test (si existe)**
- accuracy: `0.44656488549618323`
- precision: `0.3216374269005848`
- recall: `0.6547619047619048`
- specificity: `0.34831460674157305`
- f1_score: `0.43137254901960786`
- roc_auc: `0.5095973782771535`
- average_precision: `0.37581727671567666`
- optimal_threshold: `0.5445566813085339`

**Artefactos detectados**
- metadata: `models/v4/random_forest/metadata.json`
- modelo serializado: `models/v4/random_forest/model.joblib`
- evaluacion: `output/v4/evaluation/random_forest/evaluation.json`
- threshold_comparison_json: `N/A`
- threshold_comparison_csv: `N/A`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "bootstrap": true,
  "ccp_alpha": 0.0,
  "class_weight": {
    "0": 1,
    "1": 4
  },
  "criterion": "gini",
  "max_depth": 12,
  "max_features": "sqrt",
  "max_leaf_nodes": null,
  "max_samples": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": null,
  "n_estimators": 200,
  "n_jobs": -1,
  "oob_score": true,
  "random_state": 42,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 27. v5 / gradient_boosting

**Identificacion**
- Version (ruta): `v5`
- Modelo: `gradient_boosting`
- Timestamp entrenamiento: `2026-01-26T02:02:40.325809`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.6038277511961723`, test_std=`0.0344231688968516`
- precision: test_mean=`0.3656959064327485`, test_std=`0.048460384624405856`
- recall: test_mean=`0.31044776119402984`, test_std=`0.06360380821690303`
- f1: test_mean=`0.33244368446642375`, test_std=`0.04520787125174656`
- roc_auc: test_mean=`0.5651040571788942`, test_std=`0.018427321891961747`
- f2: test_mean=`0.31815282934466105`, test_std=`0.054921047083212104`

**Evaluacion test (si existe)**
- accuracy: `0.5725190839694656`
- precision: `0.29411764705882354`
- recall: `0.23809523809523808`
- specificity: `0.7303370786516854`
- f1_score: `0.2631578947368421`
- roc_auc: `0.521033975387908`
- average_precision: `0.3242453453340006`
- optimal_threshold: `0.10878439404051966`

**Artefactos detectados**
- metadata: `models/v5/gradient_boosting/metadata.json`
- modelo serializado: `models/v5/gradient_boosting/model.joblib`
- evaluacion: `output/v5/evaluation/gradient_boosting/evaluation.json`
- threshold_comparison_json: `N/A`
- threshold_comparison_csv: `N/A`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "ccp_alpha": 0.0,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.05,
  "loss": "log_loss",
  "max_depth": 8,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 200,
  "n_iter_no_change": null,
  "random_state": 42,
  "subsample": 0.9,
  "tol": 0.0001,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 28. v5 / logistic_regression

**Identificacion**
- Version (ruta): `v5`
- Modelo: `logistic_regression`
- Timestamp entrenamiento: `2026-01-26T02:01:23.818565`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.32057416267942584`, test_std=`0.0`
- precision: test_mean=`0.32057416267942584`, test_std=`0.0`
- recall: test_mean=`1.0`, test_std=`0.0`
- f1: test_mean=`0.48550724637681164`, test_std=`5.551115123125783e-17`
- roc_auc: test_mean=`0.5299138112255625`, test_std=`0.06521757491443621`
- f2: test_mean=`0.7023060796645703`, test_std=`0.0`

**Evaluacion test (si existe)**
- accuracy: `0.32061068702290074`
- precision: `0.32061068702290074`
- recall: `1.0`
- specificity: `0.0`
- f1_score: `0.48554913294797686`
- roc_auc: `0.589185393258427`
- average_precision: `0.4418175883382409`
- optimal_threshold: `0.694351608942547`

**Artefactos detectados**
- metadata: `models/v5/logistic_regression/metadata.json`
- modelo serializado: `models/v5/logistic_regression/model.joblib`
- evaluacion: `output/v5/evaluation/logistic_regression/evaluation.json`
- threshold_comparison_json: `N/A`
- threshold_comparison_csv: `N/A`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "C": 0.001,
  "class_weight": {
    "0": 1,
    "1": 5
  },
  "dual": false,
  "fit_intercept": true,
  "intercept_scaling": 1,
  "l1_ratio": null,
  "max_iter": 2000,
  "multi_class": "deprecated",
  "n_jobs": null,
  "penalty": "l2",
  "random_state": 42,
  "solver": "saga",
  "tol": 0.0001,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 29. v5 / random_forest

**Identificacion**
- Version (ruta): `v5`
- Modelo: `random_forest`
- Timestamp entrenamiento: `2026-01-26T02:02:34.457276`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.34258373205741627`, test_std=`0.014385929548682208`
- precision: test_mean=`0.3224929316123346`, test_std=`0.009263035197637012`
- recall: test_mean=`0.955223880597015`, test_std=`0.036559548399748905`
- f1: test_mean=`0.4821831611644348`, test_std=`0.014998093499372409`
- roc_auc: test_mean=`0.5601744797141055`, test_std=`0.05450705072574333`
- f2: test_mean=`0.6860091615280646`, test_std=`0.023445671731824026`

**Evaluacion test (si existe)**
- accuracy: `0.33969465648854963`
- precision: `0.32270916334661354`
- recall: `0.9642857142857143`
- specificity: `0.0449438202247191`
- f1_score: `0.4835820895522388`
- roc_auc: `0.5383560727661851`
- average_precision: `0.38606645630900693`
- optimal_threshold: `0.6403805846776333`

**Artefactos detectados**
- metadata: `models/v5/random_forest/metadata.json`
- modelo serializado: `models/v5/random_forest/model.joblib`
- evaluacion: `output/v5/evaluation/random_forest/evaluation.json`
- threshold_comparison_json: `N/A`
- threshold_comparison_csv: `N/A`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "bootstrap": true,
  "ccp_alpha": 0.0,
  "class_weight": {
    "0": 1,
    "1": 5
  },
  "criterion": "gini",
  "max_depth": 10,
  "max_features": "sqrt",
  "max_leaf_nodes": null,
  "max_samples": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 4,
  "min_samples_split": 6,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": null,
  "n_estimators": 300,
  "n_jobs": null,
  "oob_score": false,
  "random_state": 42,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 30. v6 / gradient_boosting

**Identificacion**
- Version (ruta): `v6`
- Modelo: `gradient_boosting`
- Timestamp entrenamiento: `2026-01-26T02:11:55.852567`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.6411483253588516`, test_std=`0.012838667813395906`
- precision: test_mean=`0.3896718335428013`, test_std=`0.03373342786568236`
- recall: test_mean=`0.21791044776119403`, test_std=`0.07468654330266748`
- f1: test_mean=`0.27393004016818656`, test_std=`0.060150602706105824`
- roc_auc: test_mean=`0.5791675425688461`, test_std=`0.024692492201911893`
- f2: test_mean=`0.23657338548917312`, test_std=`0.07020678354732061`

**Evaluacion test (si existe)**
- accuracy: `0.648854961832061`
- precision: `0.38235294117647056`
- recall: `0.15476190476190477`
- specificity: `0.8820224719101124`
- f1_score: `0.22033898305084745`
- roc_auc: `0.5595238095238095`
- average_precision: `0.38176227638016635`
- optimal_threshold: `0.25884611788775613`

**Artefactos detectados**
- metadata: `models/v6/gradient_boosting/metadata.json`
- modelo serializado: `models/v6/gradient_boosting/model.joblib`
- evaluacion: `output/v6/evaluation/gradient_boosting/evaluation.json`
- threshold_comparison_json: `N/A`
- threshold_comparison_csv: `N/A`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "ccp_alpha": 0.0,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.08,
  "loss": "log_loss",
  "max_depth": 4,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 6,
  "min_samples_split": 6,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 150,
  "n_iter_no_change": null,
  "random_state": 42,
  "subsample": 0.8,
  "tol": 0.0001,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 31. v6 / logistic_regression

**Identificacion**
- Version (ruta): `v6`
- Modelo: `logistic_regression`
- Timestamp entrenamiento: `2026-01-26T02:12:19.550846`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.3770334928229665`, test_std=`0.01187528578563719`
- precision: test_mean=`0.32550852139091346`, test_std=`0.012087373032335052`
- recall: test_mean=`0.8835820895522388`, test_std=`0.0708915945494801`
- f1: test_mean=`0.4755353970780189`, test_std=`0.023047623891670806`
- roc_auc: test_mean=`0.5305234391423166`, test_std=`0.07088813481688717`
- f2: test_mean=`0.6576078245927093`, test_std=`0.041202121239979704`

**Evaluacion test (si existe)**
- accuracy: `0.3893129770992366`
- precision: `0.3333333333333333`
- recall: `0.9047619047619048`
- specificity: `0.14606741573033707`
- f1_score: `0.48717948717948717`
- roc_auc: `0.5841024612092027`
- average_precision: `0.4405078896234444`
- optimal_threshold: `0.5692138470728314`

**Artefactos detectados**
- metadata: `models/v6/logistic_regression/metadata.json`
- modelo serializado: `models/v6/logistic_regression/model.joblib`
- evaluacion: `output/v6/evaluation/logistic_regression/evaluation.json`
- threshold_comparison_json: `N/A`
- threshold_comparison_csv: `N/A`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "C": 0.01,
  "class_weight": {
    "0": 1,
    "1": 3
  },
  "dual": false,
  "fit_intercept": true,
  "intercept_scaling": 1,
  "l1_ratio": null,
  "max_iter": 2000,
  "multi_class": "deprecated",
  "n_jobs": null,
  "penalty": "l2",
  "random_state": 42,
  "solver": "lbfgs",
  "tol": 0.0001,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 32. v6 / random_forest

**Identificacion**
- Version (ruta): `v6`
- Modelo: `random_forest`
- Timestamp entrenamiento: `2026-01-26T02:13:05.021771`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.3435406698564593`, test_std=`0.0063476072542687076`
- precision: test_mean=`0.3253513110104981`, test_std=`0.004160705377727019`
- recall: test_mean=`0.9761194029850746`, test_std=`0.01791044776119403`
- f1: test_mean=`0.4880315319856384`, test_std=`0.006893980259640881`
- roc_auc: test_mean=`0.5624973722934622`, test_std=`0.0518484405140467`
- f2: test_mean=`0.6972013599513713`, test_std=`0.011089475840796925`

**Evaluacion test (si existe)**
- accuracy: `0.3435114503816794`
- precision: `0.324`
- recall: `0.9642857142857143`
- specificity: `0.05056179775280899`
- f1_score: `0.48502994011976047`
- roc_auc: `0.5633025682182986`
- average_precision: `0.39645059106196934`
- optimal_threshold: `0.6123920085552895`

**Artefactos detectados**
- metadata: `models/v6/random_forest/metadata.json`
- modelo serializado: `models/v6/random_forest/model.joblib`
- evaluacion: `output/v6/evaluation/random_forest/evaluation.json`
- threshold_comparison_json: `N/A`
- threshold_comparison_csv: `N/A`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "bootstrap": true,
  "ccp_alpha": 0.0,
  "class_weight": {
    "0": 1,
    "1": 4
  },
  "criterion": "gini",
  "max_depth": 8,
  "max_features": "sqrt",
  "max_leaf_nodes": null,
  "max_samples": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 6,
  "min_samples_split": 6,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": null,
  "n_estimators": 300,
  "n_jobs": null,
  "oob_score": false,
  "random_state": 42,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 33. v7 / gradient_boosting

**Identificacion**
- Version (ruta): `v7`
- Modelo: `gradient_boosting`
- Timestamp entrenamiento: `2026-01-26T02:59:03.973168`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.6411483253588516`, test_std=`0.012838667813395906`
- precision: test_mean=`0.3896718335428013`, test_std=`0.03373342786568236`
- recall: test_mean=`0.21791044776119403`, test_std=`0.07468654330266748`
- f1: test_mean=`0.27393004016818656`, test_std=`0.060150602706105824`
- roc_auc: test_mean=`0.5791675425688461`, test_std=`0.024692492201911893`
- f2: test_mean=`0.23657338548917312`, test_std=`0.07020678354732061`

**Evaluacion test (si existe)**
- accuracy: `0.648854961832061`
- precision: `0.38235294117647056`
- recall: `0.15476190476190477`
- specificity: `0.8820224719101124`
- f1_score: `0.22033898305084745`
- roc_auc: `0.5595238095238095`
- average_precision: `0.38176227638016635`
- optimal_threshold: `0.25884611788775613`
- optimal_threshold_results:
  - threshold: `0.27`
  - accuracy: `0.5076335877862596`
  - precision: `0.35294117647058826`
  - recall: `0.6428571428571429`
  - specificity: `0.4438202247191011`
  - f1_score: `0.45569620253164556`

**Artefactos detectados**
- metadata: `models/v7/gradient_boosting/metadata.json`
- modelo serializado: `models/v7/gradient_boosting/model.joblib`
- evaluacion: `output/v7/evaluation/gradient_boosting/evaluation.json`
- threshold_comparison_json: `output/v7/threshold_analysis/gradient_boosting/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v7/threshold_analysis/gradient_boosting/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "ccp_alpha": 0.0,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.08,
  "loss": "log_loss",
  "max_depth": 4,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 6,
  "min_samples_split": 6,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 150,
  "n_iter_no_change": null,
  "random_state": 42,
  "subsample": 0.8,
  "tol": 0.0001,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 34. v7 / logistic_regression

**Identificacion**
- Version (ruta): `v7`
- Modelo: `logistic_regression`
- Timestamp entrenamiento: `2026-01-26T02:59:26.132467`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.3770334928229665`, test_std=`0.01187528578563719`
- precision: test_mean=`0.32550852139091346`, test_std=`0.012087373032335052`
- recall: test_mean=`0.8835820895522388`, test_std=`0.0708915945494801`
- f1: test_mean=`0.4755353970780189`, test_std=`0.023047623891670806`
- roc_auc: test_mean=`0.5305234391423166`, test_std=`0.07088813481688717`
- f2: test_mean=`0.6576078245927093`, test_std=`0.041202121239979704`

**Evaluacion test (si existe)**
- accuracy: `0.3893129770992366`
- precision: `0.3333333333333333`
- recall: `0.9047619047619048`
- specificity: `0.14606741573033707`
- f1_score: `0.48717948717948717`
- roc_auc: `0.5841024612092027`
- average_precision: `0.4405078896234444`
- optimal_threshold: `0.5692138470728314`
- optimal_threshold_results:
  - threshold: `0.5000000000000001`
  - accuracy: `0.3893129770992366`
  - precision: `0.3333333333333333`
  - recall: `0.9047619047619048`
  - specificity: `0.14606741573033707`
  - f1_score: `0.48717948717948717`

**Artefactos detectados**
- metadata: `models/v7/logistic_regression/metadata.json`
- modelo serializado: `models/v7/logistic_regression/model.joblib`
- evaluacion: `output/v7/evaluation/logistic_regression/evaluation.json`
- threshold_comparison_json: `output/v7/threshold_analysis/logistic_regression/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v7/threshold_analysis/logistic_regression/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "C": 0.01,
  "class_weight": {
    "0": 1,
    "1": 3
  },
  "dual": false,
  "fit_intercept": true,
  "intercept_scaling": 1,
  "l1_ratio": null,
  "max_iter": 2000,
  "multi_class": "deprecated",
  "n_jobs": null,
  "penalty": "l2",
  "random_state": 42,
  "solver": "lbfgs",
  "tol": 0.0001,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 35. v7 / random_forest

**Identificacion**
- Version (ruta): `v7`
- Modelo: `random_forest`
- Timestamp entrenamiento: `2026-01-26T03:00:05.683005`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.3435406698564593`, test_std=`0.0063476072542687076`
- precision: test_mean=`0.3253513110104981`, test_std=`0.004160705377727019`
- recall: test_mean=`0.9761194029850746`, test_std=`0.01791044776119403`
- f1: test_mean=`0.4880315319856384`, test_std=`0.006893980259640881`
- roc_auc: test_mean=`0.5624973722934622`, test_std=`0.0518484405140467`
- f2: test_mean=`0.6972013599513713`, test_std=`0.011089475840796925`

**Evaluacion test (si existe)**
- accuracy: `0.3435114503816794`
- precision: `0.324`
- recall: `0.9642857142857143`
- specificity: `0.05056179775280899`
- f1_score: `0.48502994011976047`
- roc_auc: `0.5633025682182986`
- average_precision: `0.39645059106196934`
- optimal_threshold: `0.6123920085552895`
- optimal_threshold_results:
  - threshold: `0.5900000000000002`
  - accuracy: `0.4732824427480916`
  - precision: `0.3532608695652174`
  - recall: `0.7738095238095238`
  - specificity: `0.33146067415730335`
  - f1_score: `0.48507462686567165`

**Artefactos detectados**
- metadata: `models/v7/random_forest/metadata.json`
- modelo serializado: `models/v7/random_forest/model.joblib`
- evaluacion: `output/v7/evaluation/random_forest/evaluation.json`
- threshold_comparison_json: `output/v7/threshold_analysis/random_forest/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v7/threshold_analysis/random_forest/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "bootstrap": true,
  "ccp_alpha": 0.0,
  "class_weight": {
    "0": 1,
    "1": 4
  },
  "criterion": "gini",
  "max_depth": 8,
  "max_features": "sqrt",
  "max_leaf_nodes": null,
  "max_samples": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 6,
  "min_samples_split": 6,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": null,
  "n_estimators": 300,
  "n_jobs": null,
  "oob_score": false,
  "random_state": 42,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 36. v8 / gradient_boosting

**Identificacion**
- Version (ruta): `v8`
- Modelo: `gradient_boosting`
- Timestamp entrenamiento: `2026-01-26T03:34:12.579887`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.6248803827751196`, test_std=`0.014064055939424967`
- precision: test_mean=`0.3847841849947113`, test_std=`0.0389882101496856`
- recall: test_mean=`0.30149253731343284`, test_std=`0.08925323216552868`
- f1: test_mean=`0.33394607270076604`, test_std=`0.06650752482102269`
- roc_auc: test_mean=`0.5651250788311961`, test_std=`0.01826748322621685`
- f2: test_mean=`0.3129140169090957`, test_std=`0.08064397622527847`

**Evaluacion test (si existe)**
- accuracy: `0.5763358778625954`
- precision: `0.29850746268656714`
- recall: `0.23809523809523808`
- specificity: `0.7359550561797753`
- f1_score: `0.26490066225165565`
- roc_auc: `0.5286583734617442`
- average_precision: `0.3341848312637077`
- optimal_threshold: `0.22057940093759776`
- optimal_threshold_results:
  - threshold: `0.38000000000000006`
  - accuracy: `0.5267175572519084`
  - precision: `0.3`
  - recall: `0.35714285714285715`
  - specificity: `0.6067415730337079`
  - f1_score: `0.32608695652173914`

**Artefactos detectados**
- metadata: `models/v8/gradient_boosting/metadata.json`
- modelo serializado: `models/v8/gradient_boosting/model.joblib`
- evaluacion: `output/v8/evaluation/gradient_boosting/evaluation.json`
- threshold_comparison_json: `output/v8/threshold_analysis/gradient_boosting/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v8/threshold_analysis/gradient_boosting/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "ccp_alpha": 0.0,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.08,
  "loss": "log_loss",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 200,
  "n_iter_no_change": null,
  "random_state": 42,
  "subsample": 0.7,
  "tol": 0.0001,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 37. v8 / knn

**Identificacion**
- Version (ruta): `v8`
- Modelo: `knn`
- Timestamp entrenamiento: `2026-01-26T03:34:18.621176`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.6382775119617226`, test_std=`0.03613628246602722`
- precision: test_mean=`0.4217459272266121`, test_std=`0.06944386642009977`
- recall: test_mean=`0.32238805970149254`, test_std=`0.0504821926127993`
- f1: test_mean=`0.3635573029682622`, test_std=`0.0508567912467137`
- roc_auc: test_mean=`0.5431364305234392`, test_std=`0.04599229826695984`
- f2: test_mean=`0.3373270575925243`, test_std=`0.04921723284862706`

**Evaluacion test (si existe)**
- accuracy: `0.5610687022900763`
- precision: `0.28169014084507044`
- recall: `0.23809523809523808`
- specificity: `0.7134831460674157`
- f1_score: `0.25806451612903225`
- roc_auc: `0.4922752808988764`
- average_precision: `0.3079505785243247`
- optimal_threshold: `0.22449100957806606`
- optimal_threshold_results:
  - threshold: `0.5100000000000001`
  - accuracy: `0.5648854961832062`
  - precision: `0.27941176470588236`
  - recall: `0.2261904761904762`
  - specificity: `0.7247191011235955`
  - f1_score: `0.25`

**Artefactos detectados**
- metadata: `models/v8/knn/metadata.json`
- modelo serializado: `models/v8/knn/model.joblib`
- evaluacion: `output/v8/evaluation/knn/evaluation.json`
- threshold_comparison_json: `output/v8/threshold_analysis/knn/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v8/threshold_analysis/knn/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "algorithm": "auto",
  "leaf_size": 30,
  "metric": "minkowski",
  "metric_params": null,
  "n_jobs": null,
  "n_neighbors": 5,
  "p": 2,
  "weights": "distance"
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 38. v8 / lightgbm

**Identificacion**
- Version (ruta): `v8`
- Modelo: `lightgbm`
- Timestamp entrenamiento: `2026-01-26T03:35:07.232667`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.6267942583732057`, test_std=`0.018654151856093724`
- precision: test_mean=`0.39035087719298245`, test_std=`0.0449358964336079`
- recall: test_mean=`0.31044776119402984`, test_std=`0.09024606840496105`
- f1: test_mean=`0.3419960277350728`, test_std=`0.0698762068143287`
- roc_auc: test_mean=`0.5725352112676056`, test_std=`0.029575425304387778`
- f2: test_mean=`0.32166180358331226`, test_std=`0.08270029308862906`

**Evaluacion test (si existe)**
- accuracy: `0.5687022900763359`
- precision: `0.26229508196721313`
- recall: `0.19047619047619047`
- specificity: `0.7471910112359551`
- f1_score: `0.2206896551724138`
- roc_auc: `0.5227728731942215`
- average_precision: `0.34038718281298386`
- optimal_threshold: `0.21187721946774885`
- optimal_threshold_results:
  - threshold: `0.35000000000000003`
  - accuracy: `0.5190839694656488`
  - precision: `0.3055555555555556`
  - recall: `0.39285714285714285`
  - specificity: `0.5786516853932584`
  - f1_score: `0.34375`

**Artefactos detectados**
- metadata: `models/v8/lightgbm/metadata.json`
- modelo serializado: `models/v8/lightgbm/model.joblib`
- evaluacion: `output/v8/evaluation/lightgbm/evaluation.json`
- threshold_comparison_json: `output/v8/threshold_analysis/lightgbm/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v8/threshold_analysis/lightgbm/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "boosting_type": "gbdt",
  "class_weight": null,
  "colsample_bytree": 0.7,
  "importance_type": "split",
  "learning_rate": 0.05,
  "max_depth": -1,
  "min_child_samples": 10,
  "min_child_weight": 0.001,
  "min_split_gain": 0.0,
  "n_estimators": 300,
  "n_jobs": null,
  "num_leaves": 31,
  "objective": null,
  "random_state": 42,
  "reg_alpha": 0.0,
  "reg_lambda": 0.0,
  "subsample": 0.8,
  "subsample_for_bin": 200000,
  "subsample_freq": 0
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 39. v8 / logistic_regression

**Identificacion**
- Version (ruta): `v8`
- Modelo: `logistic_regression`
- Timestamp entrenamiento: `2026-01-26T03:33:10.525059`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.3177033492822966`, test_std=`0.004879444510615098`
- precision: test_mean=`0.3179175117290245`, test_std=`0.0024156938496571385`
- recall: test_mean=`0.9850746268656716`, test_std=`0.009439634806472766`
- f1: test_mean=`0.4806952569275641`, test_std=`0.003769685845331449`
- roc_auc: test_mean=`0.5299138112255622`, test_std=`0.06424330442862414`
- f2: test_mean=`0.6938547518949338`, test_std=`0.005841489090937644`

**Evaluacion test (si existe)**
- accuracy: `0.32061068702290074`
- precision: `0.32061068702290074`
- recall: `1.0`
- specificity: `0.0`
- f1_score: `0.48554913294797686`
- roc_auc: `0.5897873194221509`
- average_precision: `0.44413596790491916`
- optimal_threshold: `0.5774600970951057`
- optimal_threshold_results:
  - threshold: `0.5500000000000002`
  - accuracy: `0.3549618320610687`
  - precision: `0.3206751054852321`
  - recall: `0.9047619047619048`
  - specificity: `0.09550561797752809`
  - f1_score: `0.4735202492211838`

**Artefactos detectados**
- metadata: `models/v8/logistic_regression/metadata.json`
- modelo serializado: `models/v8/logistic_regression/model.joblib`
- evaluacion: `output/v8/evaluation/logistic_regression/evaluation.json`
- threshold_comparison_json: `output/v8/threshold_analysis/logistic_regression/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v8/threshold_analysis/logistic_regression/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "C": 0.001,
  "class_weight": {
    "0": 1,
    "1": 3
  },
  "dual": false,
  "fit_intercept": true,
  "intercept_scaling": 1,
  "l1_ratio": null,
  "max_iter": 100,
  "multi_class": "deprecated",
  "n_jobs": null,
  "penalty": "l2",
  "random_state": 42,
  "solver": "lbfgs",
  "tol": 0.0001,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 40. v8 / random_forest

**Identificacion**
- Version (ruta): `v8`
- Modelo: `random_forest`
- Timestamp entrenamiento: `2026-01-26T03:34:53.433061`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.337799043062201`, test_std=`0.006490267926435676`
- precision: test_mean=`0.3241059265007415`, test_std=`0.004305733771931798`
- recall: test_mean=`0.982089552238806`, test_std=`0.021935729039849364`
- f1: test_mean=`0.48735970020841357`, test_std=`0.007444291957121163`
- roc_auc: test_mean=`0.5631490435148203`, test_std=`0.056470332757741465`
- f2: test_mean=`0.6984624304400014`, test_std=`0.012681596790485644`

**Evaluacion test (si existe)**
- accuracy: `0.3435114503816794`
- precision: `0.32677165354330706`
- recall: `0.9880952380952381`
- specificity: `0.03932584269662921`
- f1_score: `0.4911242603550296`
- roc_auc: `0.5737025147137507`
- average_precision: `0.4031055467814172`
- optimal_threshold: `0.6095725781243557`
- optimal_threshold_results:
  - threshold: `0.5900000000000002`
  - accuracy: `0.4770992366412214`
  - precision: `0.3612565445026178`
  - recall: `0.8214285714285714`
  - specificity: `0.3146067415730337`
  - f1_score: `0.5018181818181818`

**Artefactos detectados**
- metadata: `models/v8/random_forest/metadata.json`
- modelo serializado: `models/v8/random_forest/model.joblib`
- evaluacion: `output/v8/evaluation/random_forest/evaluation.json`
- threshold_comparison_json: `output/v8/threshold_analysis/random_forest/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v8/threshold_analysis/random_forest/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "bootstrap": true,
  "ccp_alpha": 0.0,
  "class_weight": {
    "0": 1,
    "1": 4
  },
  "criterion": "gini",
  "max_depth": 8,
  "max_features": "sqrt",
  "max_leaf_nodes": null,
  "max_samples": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 8,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": null,
  "n_estimators": 500,
  "n_jobs": null,
  "oob_score": false,
  "random_state": 42,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 41. v9 / gradient_boosting

**Identificacion**
- Version (ruta): `v9`
- Modelo: `gradient_boosting`
- Timestamp entrenamiento: `2026-01-26T04:24:01.933095`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.6555023923444976`, test_std=`0.010036448307848328`
- precision: test_mean=`0.3492548159614723`, test_std=`0.08707361001053189`
- recall: test_mean=`0.10746268656716418`, test_std=`0.055364886540272856`
- f1: test_mean=`0.16090546528650612`, test_std=`0.07233904982881888`
- roc_auc: test_mean=`0.5808072314483919`, test_std=`0.025567368021238965`
- f2: test_mean=`0.12376185016193149`, test_std=`0.06117339725361996`

**Evaluacion test (si existe)**
- accuracy: `0.6755725190839694`
- precision: `0.47058823529411764`
- recall: `0.09523809523809523`
- specificity: `0.949438202247191`
- f1_score: `0.15841584158415842`
- roc_auc: `0.5715623327982878`
- average_precision: `0.4040211728512777`
- optimal_threshold: `0.28066880615685696`
- optimal_threshold_results:
  - threshold: `0.30000000000000004`
  - accuracy: `0.5267175572519084`
  - precision: `0.36666666666666664`
  - recall: `0.6547619047619048`
  - specificity: `0.46629213483146065`
  - f1_score: `0.4700854700854701`

**Artefactos detectados**
- metadata: `models/v9/gradient_boosting/metadata.json`
- modelo serializado: `models/v9/gradient_boosting/model.joblib`
- evaluacion: `output/v9/evaluation/gradient_boosting/evaluation.json`
- threshold_comparison_json: `output/v9/threshold_analysis/gradient_boosting/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v9/threshold_analysis/gradient_boosting/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "ccp_alpha": 0.0,
  "criterion": "friedman_mse",
  "init": null,
  "learning_rate": 0.03,
  "loss": "log_loss",
  "max_depth": 5,
  "max_features": null,
  "max_leaf_nodes": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 100,
  "n_iter_no_change": null,
  "random_state": 42,
  "subsample": 0.8,
  "tol": 0.0001,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 42. v9 / knn

**Identificacion**
- Version (ruta): `v9`
- Modelo: `knn`
- Timestamp entrenamiento: `2026-01-26T04:24:07.846478`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.6421052631578947`, test_std=`0.02763546235060987`
- precision: test_mean=`0.41023594371618693`, test_std=`0.06505308627146252`
- recall: test_mean=`0.24477611940298508`, test_std=`0.03960149003409433`
- f1: test_mean=`0.30429716266416407`, test_std=`0.040079919480735346`
- roc_auc: test_mean=`0.5497372293462266`, test_std=`0.04119668128705378`
- f2: test_mean=`0.2652968059900661`, test_std=`0.039434619242947166`

**Evaluacion test (si existe)**
- accuracy: `0.5992366412213741`
- precision: `0.2857142857142857`
- recall: `0.16666666666666666`
- specificity: `0.8033707865168539`
- f1_score: `0.21052631578947367`
- roc_auc: `0.5010700909577314`
- average_precision: `0.31429951592433975`
- optimal_threshold: `0.26057676681408637`
- optimal_threshold_results:
  - threshold: `0.5100000000000001`
  - accuracy: `0.6106870229007634`
  - precision: `0.29545454545454547`
  - recall: `0.15476190476190477`
  - specificity: `0.8258426966292135`
  - f1_score: `0.203125`

**Artefactos detectados**
- metadata: `models/v9/knn/metadata.json`
- modelo serializado: `models/v9/knn/model.joblib`
- evaluacion: `output/v9/evaluation/knn/evaluation.json`
- threshold_comparison_json: `output/v9/threshold_analysis/knn/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v9/threshold_analysis/knn/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "algorithm": "auto",
  "leaf_size": 30,
  "metric": "minkowski",
  "metric_params": null,
  "n_jobs": null,
  "n_neighbors": 10,
  "p": 2,
  "weights": "distance"
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 43. v9 / lightgbm

**Identificacion**
- Version (ruta): `v9`
- Modelo: `lightgbm`
- Timestamp entrenamiento: `2026-01-26T04:24:44.363668`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.6574162679425837`, test_std=`0.012695214508537415`
- precision: test_mean=`0.42162433109801534`, test_std=`0.031461697872078906`
- recall: test_mean=`0.1761194029850746`, test_std=`0.0643004753090687`
- f1: test_mean=`0.24125757962967267`, test_std=`0.06073450966650437`
- roc_auc: test_mean=`0.5773176371662812`, test_std=`0.03563872887972926`
- f2: test_mean=`0.19688211399089578`, test_std=`0.06454645601192616`

**Evaluacion test (si existe)**
- accuracy: `0.6068702290076335`
- precision: `0.2682926829268293`
- recall: `0.13095238095238096`
- specificity: `0.8314606741573034`
- f1_score: `0.176`
- roc_auc: `0.5619983948635634`
- average_precision: `0.3742168140790628`
- optimal_threshold: `0.26300011521745226`
- optimal_threshold_results:
  - threshold: `0.24000000000000005`
  - accuracy: `0.4847328244274809`
  - precision: `0.35428571428571426`
  - recall: `0.7380952380952381`
  - specificity: `0.3651685393258427`
  - f1_score: `0.47876447876447875`

**Artefactos detectados**
- metadata: `models/v9/lightgbm/metadata.json`
- modelo serializado: `models/v9/lightgbm/model.joblib`
- evaluacion: `output/v9/evaluation/lightgbm/evaluation.json`
- threshold_comparison_json: `output/v9/threshold_analysis/lightgbm/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v9/threshold_analysis/lightgbm/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "boosting_type": "gbdt",
  "class_weight": null,
  "colsample_bytree": 0.7,
  "importance_type": "split",
  "learning_rate": 0.03,
  "max_depth": -1,
  "min_child_samples": 40,
  "min_child_weight": 0.001,
  "min_split_gain": 0.0,
  "n_estimators": 300,
  "n_jobs": null,
  "num_leaves": 63,
  "objective": null,
  "random_state": 42,
  "reg_alpha": 0.0,
  "reg_lambda": 0.0,
  "subsample": 0.7,
  "subsample_for_bin": 200000,
  "subsample_freq": 0
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 44. v9 / logistic_regression

**Identificacion**
- Version (ruta): `v9`
- Modelo: `logistic_regression`
- Timestamp entrenamiento: `2026-01-26T04:23:22.429514`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.3177033492822966`, test_std=`0.004879444510615098`
- precision: test_mean=`0.3179175117290245`, test_std=`0.0024156938496571385`
- recall: test_mean=`0.9850746268656716`, test_std=`0.009439634806472766`
- f1: test_mean=`0.4806952569275641`, test_std=`0.003769685845331449`
- roc_auc: test_mean=`0.5299138112255622`, test_std=`0.06424330442862414`
- f2: test_mean=`0.6938547518949338`, test_std=`0.005841489090937644`

**Evaluacion test (si existe)**
- accuracy: `0.32061068702290074`
- precision: `0.32061068702290074`
- recall: `1.0`
- specificity: `0.0`
- f1_score: `0.48554913294797686`
- roc_auc: `0.5897873194221509`
- average_precision: `0.44413596790491916`
- optimal_threshold: `0.5774600970951057`
- optimal_threshold_results:
  - threshold: `0.5500000000000002`
  - accuracy: `0.3549618320610687`
  - precision: `0.3206751054852321`
  - recall: `0.9047619047619048`
  - specificity: `0.09550561797752809`
  - f1_score: `0.4735202492211838`

**Artefactos detectados**
- metadata: `models/v9/logistic_regression/metadata.json`
- modelo serializado: `models/v9/logistic_regression/model.joblib`
- evaluacion: `output/v9/evaluation/logistic_regression/evaluation.json`
- threshold_comparison_json: `output/v9/threshold_analysis/logistic_regression/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v9/threshold_analysis/logistic_regression/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "C": 0.001,
  "class_weight": {
    "0": 1,
    "1": 3
  },
  "dual": false,
  "fit_intercept": true,
  "intercept_scaling": 1,
  "l1_ratio": null,
  "max_iter": 100,
  "multi_class": "deprecated",
  "n_jobs": null,
  "penalty": "l2",
  "random_state": 42,
  "solver": "lbfgs",
  "tol": 0.0001,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

### 45. v9 / random_forest

**Identificacion**
- Version (ruta): `v9`
- Modelo: `random_forest`
- Timestamp entrenamiento: `2026-01-26T04:24:25.080837`
- Pipeline version (metadata): `N/A`
- Dataset version (metadata): `N/A`

**Datos de entrenamiento**
- n_samples: `1045`
- n_features: `18`
- scale_features: `True`
- class_distribution: `{'0.0': 710, '1.0': 335}`

**Validacion cruzada (si existe en metadata)**
- n_folds: `5`
- accuracy: test_mean=`0.3397129186602871`, test_std=`0.006052206048169143`
- precision: test_mean=`0.32510257644101803`, test_std=`0.003652633220896147`
- recall: test_mean=`0.9850746268656717`, test_std=`0.01887926961294555`
- f1: test_mean=`0.48885498058872673`, test_std=`0.006277803359216144`
- roc_auc: test_mean=`0.5642526802606685`, test_std=`0.05448080281442663`
- f2: test_mean=`0.7005978422017349`, test_std=`0.010758107691467353`

**Evaluacion test (si existe)**
- accuracy: `0.3435114503816794`
- precision: `0.32677165354330706`
- recall: `0.9880952380952381`
- specificity: `0.03932584269662921`
- f1_score: `0.4911242603550296`
- roc_auc: `0.5681514178705189`
- average_precision: `0.3968682708312702`
- optimal_threshold: `0.6095258386270828`
- optimal_threshold_results:
  - threshold: `0.5900000000000002`
  - accuracy: `0.4351145038167939`
  - precision: `0.336734693877551`
  - recall: `0.7857142857142857`
  - specificity: `0.2696629213483146`
  - f1_score: `0.4714285714285714`

**Artefactos detectados**
- metadata: `models/v9/random_forest/metadata.json`
- modelo serializado: `models/v9/random_forest/model.joblib`
- evaluacion: `output/v9/evaluation/random_forest/evaluation.json`
- threshold_comparison_json: `output/v9/threshold_analysis/random_forest/threshold_comparison_test.json`
- threshold_comparison_csv: `output/v9/threshold_analysis/random_forest/threshold_comparison_test.csv`
- hyperparams_best: `N/A`
- hyperparams_search: `N/A`

**Parametros del modelo (metadata.model_params)**
```json
{
  "bootstrap": true,
  "ccp_alpha": 0.0,
  "class_weight": {
    "0": 1,
    "1": 4
  },
  "criterion": "gini",
  "max_depth": 10,
  "max_features": "sqrt",
  "max_leaf_nodes": null,
  "max_samples": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 8,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": null,
  "n_estimators": 300,
  "n_jobs": null,
  "oob_score": false,
  "random_state": 42,
  "verbose": 0,
  "warm_start": false
}
```

**Informacion para replicacion**
- Usar el `version` y `dataset_version` registrados en metadata cuando existan.
- Recuperar features desde `feature_names` en metadata y parametros desde `model_params`.
- Si falta metadata de pipeline/dataset, inferir desde ruta y confirmar contra `config/pipeline_config.yaml` historico del commit correspondiente.

