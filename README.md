# Shock - MLflow Machine Learning Project

Proyecto profesional para predicción de shock hemorrágico usando MLflow para automatizar el entrenamiento y seguimiento de modelos, siguiendo las mejores prácticas de ingeniería de software.

**Gestión de dependencias:** Poetry (estándar moderno de Python)

## Estructura del Proyecto

```bash
shock/
├── src/                          # Código fuente
│   ├── analysis/                 # Análisis de datos
│   │   ├── __init__.py
│   │   └── example_analysis.py  # Ejemplo de análisis
│   ├── config/                   # Configuraciones
│   │   ├── __init__.py
│   │   └── settings.py          # Configuración de la aplicación
│   ├── etl/                     # Pipelines ETL
│   │   ├── __init__.py
│   │   └── example_etl.py       # Ejemplo de pipeline ETL
│   ├── evaluation/              # Evaluación de modelos
│   │   ├── __init__.py
│   │   └── metrics.py          # Métricas de evaluación
│   ├── models/                  # Modelos de ML
│   │   ├── __init__.py
│   │   └── shock_classifier.py # Clasificador de shock
│   ├── pipelines/               # Pipelines ML
│   │   ├── __init__.py
│   │   ├── shock_pipeline.py   # Pipeline de shock
│   │   ├── eda_pipeline.py     # Pipeline EDA
│   │   └── eda_aggregated_pipeline.py # Pipeline EDA agregado
│   ├── preprocessing/           # Preprocesamiento
│   │   ├── __init__.py
│   │   └── feature_engineering.py # Ingeniería de características
│   ├── utils/                   # Utilidades
│   │   ├── __init__.py
│   │   └── helpers.py          # Funciones auxiliares
│   └── visualization/           # Visualización
│       ├── __init__.py
│       └── plots.py            # Gráficos y visualizaciones
├── data/                        # Datos (no versionados)
│   ├── raw/                     # Datos crudos
│   ├── processed/               # Datos procesados
│   └── output/                  # Datos de salida
├── models/                      # Modelos entrenados
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Tests unitarios
│   ├── __init__.py
│   ├── conftest.py             # Fixtures de pytest
│   └── test_shock_pipeline.py  # Tests de pipeline de shock
├── scripts/                     # Scripts de utilidad
│   └── create_sample_data.py   # Crear datos de ejemplo
├── logs/                        # Logs de la aplicación
├── doc/                         # Documentación
├── .env                         # Variables de entorno
├── .env.example                 # Ejemplo de variables de entorno
├── .gitignore                   # Archivos ignorados por git
├── pyproject.toml              # Configuración de Poetry y herramientas
├── requirements.txt             # Dependencias (backup para pip)
├── scripts/                     # Scripts de utilidad
│   ├── run_shock_pipeline.py   # Script principal de shock
│   ├── run_eda_pipeline.py     # Script EDA
│   ├── run_eda_strict_pipeline.py # Script EDA estricto
│   └── run_eda_strict_aggregated_pipeline.py # Script EDA estricto agregado
└── README.md                    # Este archivo
```

## Características

- **Poetry**: Gestión moderna de dependencias y entorno virtual
- **MLflow**: Seguimiento y automatización de modelos de ML
- **Sistema de logging robusto** con Loguru
- **Carga y preprocesamiento de datos** desde múltiples fuentes
- **Validación y chequeo de calidad** de datos
- **Pipelines de ML completos** con evaluación y registro de métricas
- **Ejemplos de análisis de datos**
- **Tests unitarios** con pytest
- **Linting moderno** con Ruff (más rápido que Flake8)
- **Documentación completa**

## Requisitos

- Python 3.11+ (recomendado Python 3.13)
- Poetry (se instala automáticamente con el script de setup)

## Instalación

### Opción 1: Setup Automático

*Nota: El script setup_env.sh no existe actualmente, seguir las instrucciones manuales*

### Opción 2: Setup Manual con Poetry

#### 1. Instalar Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### 2. Configurar Poetry

```bash
# Configurar para crear .venv en el proyecto
poetry config virtualenvs.in-project true
```

#### 3. Instalar dependencias

```bash
# Con dependencias de desarrollo, notebooks y MLflow
poetry install --with dev,notebook

# Solo dependencias de producción
poetry install
```

#### 4. Configurar .env

```bash
cp .env.example .env
# Edita tus configuraciones según sea necesario
```

### Opción 3: Instalación con pip (No recomendado)

Si prefieres no usar Poetry:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Uso Rápido

### 1. Activar el entorno

```bash
# Opción A: Shell de Poetry (recomendado)
poetry env activate

# Opción B: Ejecutar comandos con 'poetry run'
poetry run <comando>
```

### 2. Ejecutar los pipelines de ML

```bash
# Pipeline de predicción de shock hemorrágico
poetry run python scripts/run_shock_pipeline.py

# Pipeline de EDA
poetry run python scripts/run_eda_pipeline.py

# Ejecución con parámetros específicos
poetry run python scripts/run_shock_pipeline.py --data data/processed/shock.csv --output reports/shock_model
```

### 3. Iniciar MLflow UI para visualizar experimentos

```bash
# Iniciar el servidor MLflow
poetry run mlflow ui --host 0.0.0.0 --port 5000
```

## Comandos Principales

### Pipelines y Scripts

```bash
# Ejecutar pipeline de predicción de shock
poetry run python scripts/run_shock_pipeline.py

# Ejecutar pipeline de EDA
poetry run python scripts/run_eda_pipeline.py

# Ejecutar otros pipelines de EDA
poetry run python scripts/run_eda_strict_pipeline.py
poetry run python scripts/run_eda_strict_aggregated_pipeline.py

# Iniciar MLflow UI para visualizar experimentos
poetry run mlflow ui

# Registrar modelo en MLflow
poetry run python -c "import mlflow; mlflow.register_model(model_uri='ruta/al/modelo', name='shock_model')"
```

### Testing y Calidad de Código

```bash
# Ejecutar tests
poetry run pytest tests/ -v

# Tests con coverage
poetry run pytest tests/ --cov=src --cov-report=html

# Formatear código con black
poetry run black src tests scripts

# Linting con ruff (rápido!)
poetry run ruff check src tests

# Type checking con mypy
poetry run mypy src
```

### Gestión de Dependencias

```bash
# Agregar nueva dependencia
poetry add nombre-paquete

# Agregar dependencia de desarrollo
poetry add --group dev nombre-paquete

# Agregar dependencia para notebooks
poetry add --group notebook nombre-paquete

# Actualizar todas las dependencias
poetry update

# Actualizar una dependencia específica
poetry update nombre-paquete

# Ver dependencias instaladas
poetry show

# Ver árbol de dependencias
poetry show --tree

# Eliminar dependencia
poetry remove nombre-paquete
```

### Jupyter Notebooks

```bash
# Iniciar Jupyter
poetry run jupyter notebook

# O si estás en poetry shell
jupyter notebook
```

## Configuración de MLflow

MLflow se configura principalmente a través de variables de entorno y el código se encuentra en [src/pipelines/shock_pipeline.py](src/pipelines/shock_pipeline.py). Se personaliza con variables de entorno en `.env`:

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db  # URI de seguimiento
MLFLOW_S3_ENDPOINT_URL=                  # Endpoint S3 si se usa almacenamiento remoto
AWS_ACCESS_KEY_ID=                       # Credenciales AWS si se usan
AWS_SECRET_ACCESS_KEY=                   # Credenciales AWS si se usan
```

## Buenas Prácticas Implementadas

### 1. Gestión de Dependencias Moderna

- Poetry para gestión de dependencias
- Lock file (poetry.lock) para reproducibilidad
- Grupos de dependencias (dev, notebook)

### 2. Organización del Código

- Separación clara entre ETL, análisis y utilidades
- Módulos reutilizables
- Configuración centralizada

### 3. Gestión de Datos

- Separación de datos por etapas (raw, processed, output)
- Carga y preprocesamiento con pandas/scikit-learn
- Validación de calidad de datos

### 4. Calidad de Código

- Type hints en funciones
- Docstrings completos
- Linting con Ruff (más rápido que Flake8)
- Formateo automático con Black

### 5. Testing

- Tests unitarios con pytest
- Fixtures reutilizables
- Cobertura de código

### 6. Logging

- Sistema de logging robusto con Loguru
- Rotación de logs
- Múltiples niveles de log

### 7. MLflow para experimentación

- Seguimiento de experimentos y métricas
- Versionado de modelos
- Comparación de resultados
- Registro de artefactos (gráficos, modelos, resultados)

## Ejemplos de Código

### Cargar y procesar datos

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing.data_cleaner import ShockDataCleaner

# Cargar datos
df = pd.read_csv("data/raw/shock.csv")

# Limpiar y preparar datos
cleaner = ShockDataCleaner()
X, y = cleaner.prepare_features(df)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Entrenar modelo con MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Empezar un experimento de MLflow
mlflow.set_experiment("shock_prediction_experiments")

with mlflow.start_run():
    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Registrar parámetros
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # Registrar métricas
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    mlflow.log_metric("train_accuracy", train_score)
    mlflow.log_metric("test_accuracy", test_score)

    # Registrar modelo
    mlflow.sklearn.log_model(model, "model")
```

### Validar calidad de datos

```python
from src.utils.helpers import validate_data_quality

# Validar calidad de datos
quality_report = validate_data_quality(df)

# Verificar valores nulos
null_percentage = quality_report.get_null_percentages()
```

## Desarrollo

### Agregar nuevos pipelines

1. Crear nuevo archivo en `src/pipelines/`
2. Usar las utilidades existentes
3. Agregar tests en `tests/`
4. Registrar en `pyproject.toml` bajo `[tool.poetry.scripts]` si es necesario

### Agregar nuevas utilidades

1. Crear nuevo módulo en `src/utils/`
2. Agregar tests unitarios
3. Actualizar documentación

## Troubleshooting

### Poetry no encontrado después de instalación

```bash
# Agregar Poetry al PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Error de dependencias

Si hay conflictos de dependencias:

```bash
# Limpiar y reinstalar entorno
rm -rf .venv
poetry install --with dev,notebook
```

### MLflow UI no inicia

Comprobar que los puertos estén disponibles:

```bash
# Iniciar MLflow en otro puerto
poetry run mlflow ui --port 8080
```

### Recrear el entorno virtual

```bash
# Eliminar entorno actual
rm -rf .venv

# Reinstalar
poetry install --with dev,notebook
```

## Recursos

- [Poetry Documentation](https://python-poetry.org/docs/)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [Ruff Linter](https://docs.astral.sh/ruff/)

## Licencia

Este proyecto es para uso educativo - Curso de Procesamiento de Grandes Datos, ICESI.

## Autor

Creado para el curso de PDG - 9no Semestre, ICESI

## Por qué Poetry?

Poetry es el estándar moderno de Python (2025) porque:

- ✅ Gestión de dependencias + virtualenv en una herramienta
- ✅ Lock file automático para reproducibilidad
- ✅ Resolución inteligente de dependencias
- ✅ Scripts personalizados fáciles
- ✅ Build y publicación simplificados
- ✅ Más rápido y confiable que pip
- ✅ Usado por empresas modernas y proyectos open-source
