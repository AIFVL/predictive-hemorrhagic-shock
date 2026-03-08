# Predicción de Shock Hemorrágico

Pipeline de machine learning para la predicción de shock hemorrágico. Orquestado con Apache Airflow, configurable mediante un único archivo YAML y empaquetado con Docker.

## Estructura del proyecto

```
pdg-shock/
├── config/
│   ├── airflow.cfg               # Configuración de Airflow
│   └── pipeline_config.yaml      # Configuración central del pipeline
├── dags/
│   └── shock_prediction_dag.py   # Definición del DAG de Airflow
├── src/
│   ├── data/                     # Carga, validación y limpieza de datos
│   ├── datasets/                 # Construcción del dataset de entrenamiento
│   ├── features/                 # Ingeniería de características
│   ├── models/                   # Entrenamiento y evaluación de modelos
│   ├── reports/                  # Generación de reportes
│   ├── utils/                    # Utilidades (config manager, logger, I/O)
│   └── visualization/            # Generación de gráficas
├── data/
│   ├── raw/<dataset_version>/    # Datos crudos de entrada
│   ├── processed/<dataset_version>/  # Datos limpios y dataset de entrenamiento
│   └── splits/<dataset_version>/ # Particiones train/test
├── output/
│   └── <version>/                # Todo lo generado por el pipeline para esa versión
│       ├── models/<model>/       # Modelos entrenados y metadatos
│       ├── evaluation/<model>/   # Métricas de evaluación en test
│       ├── threshold_analysis/<model>/
│       ├── model_plots/<model>/
│       ├── hyperparameter_search/<model>/
│       └── eda_plots/
├── logs/                         # Logs de Airflow
├── Dockerfile                    # Imagen Docker basada en apache/airflow
├── docker-compose.yaml           # Orquestación de servicios (Airflow + PostgreSQL)
├── start_airflow.sh              # Script de inicio
└── requirements.txt              # Dependencias Python
```

## Puesta en marcha

### Requisitos

- Docker y Docker Compose
- Sistema operativo Linux, macOS o Windows con WSL

### Pasos

1. Colocar el archivo de datos en la ruta esperada según la `dataset_version` configurada:

   ```
   data/raw/<dataset_version>/shock.csv
   ```

   Por defecto `dataset_version: "v1"`, por lo que la ruta es `data/raw/v1/shock.csv`.

2. Construir la imagen Docker e iniciar los servicios:

   ```bash
   docker compose build
   bash start_airflow.sh
   ```

   El script `start_airflow.sh` prepara los permisos de los directorios compartidos y levanta los contenedores en segundo plano.

3. Abrir la interfaz web de Airflow en `http://localhost:8080` y activar el DAG `shock_prediction_pipeline`.

## Pipeline

El DAG `shock_prediction_pipeline` ejecuta los siguientes pasos de forma secuencial para los datos compartidos y en paralelo para cada modelo habilitado:

| Paso | Tarea | Descripción |
|------|-------|-------------|
| 1 | `validate_raw_data` | Valida el CSV crudo contra las reglas definidas en config |
| 2 | `clean_data` | Elimina columnas de fuga e identificadores |
| 3 | `create_training_dataset` | Aplica ingeniería de características y genera particiones train/test |
| 4 | `generate_eda_plots` | Genera gráficas exploratorias de los datos limpios |
| 5A | `optimize_hyperparams_<model>` | Búsqueda de hiperparámetros (si está habilitada) |
| 5 | `train_model_<model>` | Entrena el modelo con validación cruzada y guarda el artefacto |
| 5B | `optimize_threshold_<model>` | Optimiza el umbral de clasificación sobre el conjunto de entrenamiento |
| 6 | `evaluate_model_<model>` | Evalúa el modelo sobre el conjunto de test con el umbral óptimo |
| 6B | `compare_thresholds_<model>` | Compara distintos umbrales sobre test (solo reporte) |
| 7 | `generate_plots_<model>` | Genera gráficas de evaluación por modelo |
| 8 | `generate_comparison_plots` | Genera gráficas comparativas entre todos los modelos |

Los pasos 1–4 son comunes a todos los modelos. Los pasos 5A–7 se ejecutan una vez por cada modelo habilitado y en paralelo entre sí. El paso 8 espera a que todos los modelos hayan finalizado.

## Configuración

Toda la configuración del pipeline se encuentra en `config/pipeline_config.yaml`. Es la única fuente de verdad.

### Versiones

```yaml
version: "v24"          # Versión del pipeline — controla rutas de modelos y salidas
dataset_version: "v1"   # Versión del dataset — controla qué datos se leen
```

Los artefactos del pipeline (modelos, evaluaciones, gráficas) se guardan bajo `models/<version>/` y `output/<version>/`. Los datos de entrada se leen desde `data/raw/<dataset_version>/` y `data/processed/<dataset_version>/`. Ambas versiones pueden ser distintas, lo que permite reutilizar un mismo dataset con distintas configuraciones del pipeline.

### Modelos

Cada modelo tiene una sección propia con los siguientes atributos:

```yaml
models:
  lightgbm:
    enabled: true           # false para excluir del pipeline completamente
    module: "lightgbm"
    class: "LGBMClassifier"
    target_recall: 0.85     # restricción mínima de recall para la optimización de umbral
    params:                 # hiperparámetros estáticos (usados cuando la búsqueda está deshabilitada)
      n_estimators: 400
      ...
    search_space:           # espacio de búsqueda (usado cuando hyperparameter_search.enabled: true)
      n_estimators: [400, 200, 800]
      ...
```

Los modelos disponibles son `lightgbm`, `decision_tree`, `logistic_regression` y `naive_bayes`. Cualquiera puede desactivarse con `enabled: false` sin modificar ningún otro archivo.

### Características

Las características se definen en el YAML. Hay tres tipos:

- `numerical_features` y `binary_features`: listas de columnas del CSV original.
- `aggregated_features`: características derivadas por combinación de columnas base (suma, producto, inverso binario). Cada una tiene un atributo `enabled`.
- `categorical_features`: variables discretizadas por umbrales sobre una columna fuente. Cada una tiene un atributo `enabled`.

Desactivar una característica con `enabled: false` la excluye del dataset de entrenamiento sin necesidad de modificar código.

### Búsqueda de hiperparámetros

```yaml
hyperparameter_search:
  enabled: false   # true para activar RandomizedSearchCV
  n_iter: 70
  cv_folds: 10
  scoring: "f2"
  n_jobs: -1
```

Cuando está desactivada, cada modelo usa los valores de su sección `params`.

### Validación cruzada

```yaml
cross_validation:
  n_folds: 10
  scale_features: true
  n_jobs: -1
```

### Optimización de umbral

```yaml
evaluation:
  threshold_optimization:
    search_thresholds: [0.20, 0.55, 0.01]   # [inicio, fin, paso]
    test_thresholds: [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    permutation_n: 200
```

El `target_recall` mínimo se configura por modelo (ver sección Modelos).

### Limpieza de datos

```yaml
cleaning:
  leakage_variables:
    - MUERTE.1
  identifier_columns:
    - CODIGO
```

### Poda de características

```yaml
feature_pruning:
  enabled: true
  min_total_ones: 10   # elimina columnas binarias con muy baja prevalencia
```

## Artefactos generados

Por cada ejecución del pipeline se generan los siguientes artefactos:

| Ruta | Contenido |
|------|-----------|
| `output/<version>/validation_report.json` | Reporte de validación del CSV crudo |
| `output/<version>/cleaning_report.json` | Reporte de limpieza |
| `output/<version>/eda_plots/` | Gráficas exploratorias |
| `output/<version>/models/<model>/model.joblib` | Modelo entrenado |
| `output/<version>/models/<model>/metadata.json` | Metadatos del entrenamiento (parámetros, CV, versiones) |
| `output/<version>/evaluation/<model>/evaluation.json` | Métricas de evaluación en test |
| `output/<version>/threshold_analysis/<model>/` | Comparación de umbrales en test |
| `output/<version>/model_plots/<model>/` | Gráficas de evaluación por modelo |
| `output/<version>/hyperparameter_search/<model>/` | Mejores parámetros encontrados (si la búsqueda está habilitada) |

## Dependencias

Las dependencias se gestionan con `pip` y están declaradas en `requirements.txt`. Se instalan automáticamente durante la construcción de la imagen Docker (`docker compose build`).