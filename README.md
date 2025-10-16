# Shock - PySpark Data Analysis Project

Proyecto profesional para análisis de datos con PySpark siguiendo las mejores prácticas de ingeniería de software.

**Gestión de dependencias:** Poetry (estándar moderno de Python)

## Estructura del Proyecto

```bash
shock/
├── src/                          # Código fuente
│   ├── config/                   # Configuraciones
│   │   ├── __init__.py
│   │   ├── spark_config.py      # Configuración de Spark
│   │   └── settings.py          # Configuración de la aplicación
│   ├── etl/                     # Pipelines ETL
│   │   ├── __init__.py
│   │   └── example_etl.py       # Ejemplo de pipeline ETL
│   ├── analysis/                # Análisis de datos
│   │   ├── __init__.py
│   │   └── example_analysis.py  # Ejemplo de análisis
│   └── utils/                   # Utilidades
│       ├── __init__.py
│       ├── logger.py            # Configuración de logging
│       ├── data_loader.py       # Carga de datos
│       └── data_quality.py      # Validación de calidad
├── data/                        # Datos (no versionados)
│   ├── raw/                     # Datos crudos
│   ├── processed/               # Datos procesados
│   ├── staging/                 # Datos en staging
│   └── output/                  # Datos de salida
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Tests unitarios
│   ├── __init__.py
│   ├── conftest.py             # Fixtures de pytest
│   ├── test_data_loader.py     # Tests de data loader
│   └── test_data_quality.py    # Tests de calidad de datos
├── scripts/                     # Scripts de utilidad
│   ├── create_sample_data.py   # Crear datos de ejemplo
│   ├── run_pipeline.sh         # Ejecutar pipeline completo
│   └── setup_env.sh            # Setup automático
├── logs/                        # Logs de la aplicación
├── .env                         # Variables de entorno
├── .env.example                 # Ejemplo de variables de entorno
├── .gitignore                   # Archivos ignorados por git
├── pyproject.toml              # Configuración de Poetry y herramientas
├── requirements.txt             # Dependencias (backup para pip)
└── README.md                    # Este archivo
```

## Características

- **Poetry**: Gestión moderna de dependencias y entorno virtual
- **Configuración modular de Spark** con optimizaciones
- **Sistema de logging robusto** con Loguru
- **Carga de datos** desde múltiples fuentes (CSV, Parquet, JSON, JDBC)
- **Validación y chequeo de calidad** de datos
- **Pipeline ETL completo**
- **Ejemplos de análisis de datos**
- **Tests unitarios** con pytest
- **Linting moderno** con Ruff (más rápido que Flake8)
- **Documentación completa**

## Requisitos

- Python 3.8+
- Java 8 o 11 (requerido por Spark)
- Poetry (se instala automáticamente con el script de setup)

## Instalación

### Opción 1: Setup Automático

```bash
bash scripts/setup_env.sh
```

Este script:

- ✓ Verifica Python y Java
- ✓ Instala Poetry automáticamente si no está instalado
- ✓ Configura el entorno virtual
- ✓ Instala todas las dependencias
- ✓ Crea el archivo .env

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
# Con dependencias de desarrollo y notebooks
poetry install --with dev,notebook

# Solo dependencias de producción
poetry install
```

#### 4. Configurar .env

```bash
cp .env.example .env
nano .env  # Edita tus configuraciones
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
poetry env

# Opción B: Ejecutar comandos con 'poetry run'
poetry run <comando>
```

### 2. Ejecutar el pipeline completo

```bash
bash scripts/run_pipeline.sh
```

Esto ejecuta:

1. Creación de datos de ejemplo
2. Pipeline ETL
3. Análisis de datos

## Comandos Principales

### Pipelines y Scripts

```bash
# Crear datos de ejemplo
poetry run shock-create-data

# Ejecutar pipeline ETL
poetry run shock-etl

# Ejecutar análisis
poetry run shock-analysis

# Pipeline completo
bash scripts/run_pipeline.sh
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

## Configuración de Spark

La configuración se encuentra en [src/config/spark_config.py](src/config/spark_config.py) y se personaliza con variables de entorno en `.env`:

```bash
SPARK_MASTER=local[*]              # Master URL
SPARK_APP_NAME=shock_data_analysis # Nombre de la aplicación
SPARK_DRIVER_MEMORY=4g             # Memoria del driver
SPARK_EXECUTOR_MEMORY=4g           # Memoria del executor
SPARK_EXECUTOR_CORES=2             # Cores por executor
SPARK_SQL_SHUFFLE_PARTITIONS=200   # Particiones para shuffles
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

- Separación de datos por etapas (raw, processed, staging, output)
- Uso de formatos eficientes (Parquet)
- Particionamiento de datos

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

### 7. Optimizaciones de Spark

- Adaptive Query Execution habilitado
- Serialización Kryo
- Configuración de particiones dinámicas

## Ejemplos de Código

### Cargar datos CSV

```python
from config.spark_config import get_spark_session
from utils.data_loader import DataLoader

spark = get_spark_session()
loader = DataLoader(spark)

df = loader.load_csv(
    path="data/raw/example.csv",
    header=True,
    infer_schema=True
)
```

### Validar calidad de datos

```python
from utils.data_quality import DataQualityChecker

checker = DataQualityChecker()

# Chequear valores nulos
null_counts = checker.check_nulls(df)

# Chequear duplicados
duplicates = checker.check_duplicates(df)

# Generar reporte completo
report = checker.get_quality_report(df)
```

### Guardar datos en Parquet

```python
from utils.data_loader import DataWriter

DataWriter.write_parquet(
    df=df,
    path="data/processed/output",
    mode="overwrite",
    partition_by=["year", "month"]
)
```

## Desarrollo

### Agregar nuevos pipelines ETL

1. Crear nuevo archivo en `src/etl/`
2. Usar las utilidades existentes
3. Agregar tests en `tests/`
4. (Opcional) Registrar en `pyproject.toml` bajo `[tool.poetry.scripts]`

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

### Error de Java

```bash
# Ubuntu/Debian
sudo apt install openjdk-11-jdk

# Fedora
sudo dnf install java-11-openjdk

# macOS
brew install openjdk@11
```

### Error de memoria

Ajusta las configuraciones en `.env`:

```bash
SPARK_DRIVER_MEMORY=2g
SPARK_EXECUTOR_MEMORY=2g
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
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [Spark SQL Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [Ruff Linter](https://docs.astral.sh/ruff/)

## Documentación Adicional

- [QUICKSTART.md](QUICKSTART.md) - Guía rápida de 5 minutos
- [ARCHITECTURE.md](ARCHITECTURE.md) - Arquitectura y diseño del proyecto

## Licencia

Este proyecto es para uso educativo - Curso de Procesamiento de Grandes Datos, ICESI.

## Autor

Creado para el curso de PDG - 8vo Semestre, ICESI

## Por qué Poetry?

Poetry es el estándar moderno de Python (2025) porque:

- ✅ Gestión de dependencias + virtualenv en una herramienta
- ✅ Lock file automático para reproducibilidad
- ✅ Resolución inteligente de dependencias
- ✅ Scripts personalizados fáciles
- ✅ Build y publicación simplificados
- ✅ Más rápido y confiable que pip
- ✅ Usado por empresas modernas y proyectos open-source
