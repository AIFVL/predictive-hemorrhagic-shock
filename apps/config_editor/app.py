"""
Streamlit App - Pipeline Configuration Editor

Interfaz visual para editar el archivo pipeline_config.yaml de forma segura.
Incluye validaciones y guías para usuarios no expertos.
"""

import streamlit as st
import yaml
from pathlib import Path
from typing import Any, Dict, List
from copy import deepcopy
import re
import os

st.set_page_config(
    page_title="Editor de Configuración - Predicción Shock Hemorrágico",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ruta del archivo de configuración (adaptable a Docker y local)
if os.path.exists("/config/pipeline_config.yaml"):
    # Ruta en contenedor Docker
    CONFIG_PATH = Path("/config/pipeline_config.yaml")
else:
    # Ruta local (desarrollo)
    CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "pipeline_config.yaml"


class ConfigValidator:
    """Validador de configuraciones del pipeline."""

    @staticmethod
    def validate_version(version: str) -> tuple[bool, str]:
        """Valida formato de versión (ej: v1, v2, v10)."""
        if not version:
            return False, "La versión no puede estar vacía"
        if not re.match(r'^v\d+$', version):
            return False, "Formato inválido. Debe ser 'v' seguido de un número (ej: v1, v2)"
        return True, ""

    @staticmethod
    def validate_random_seed(seed: int) -> tuple[bool, str]:
        """Valida semilla aleatoria."""
        if not isinstance(seed, int):
            return False, "La semilla debe ser un número entero"
        if seed < 0:
            return False, "La semilla debe ser un número positivo"
        return True, ""

    @staticmethod
    def validate_age_range(min_age: float, max_age: float) -> tuple[bool, str]:
        """Valida rango de edad."""
        if min_age < 0 or min_age > 150:
            return False, "Edad mínima debe estar entre 0 y 150"
        if max_age < 0 or max_age > 150:
            return False, "Edad máxima debe estar entre 0 y 150"
        if min_age >= max_age:
            return False, "La edad mínima debe ser menor que la máxima"
        return True, ""

    @staticmethod
    def validate_hemoglobin_range(min_hb: float, max_hb: float) -> tuple[bool, str]:
        """Valida rango de hemoglobina."""
        if min_hb < 0 or min_hb > 30:
            return False, "Hemoglobina mínima debe estar entre 0 y 30 g/dL"
        if max_hb < 0 or max_hb > 30:
            return False, "Hemoglobina máxima debe estar entre 0 y 30 g/dL"
        if min_hb >= max_hb:
            return False, "La hemoglobina mínima debe ser menor que la máxima"
        return True, ""

    @staticmethod
    def validate_test_size(test_size: float) -> tuple[bool, str]:
        """Valida tamaño del conjunto de prueba."""
        if test_size <= 0 or test_size >= 1:
            return False, "El tamaño de prueba debe estar entre 0 y 1 (exclusivo)"
        return True, ""

    @staticmethod
    def validate_cv_folds(n_folds: int) -> tuple[bool, str]:
        """Valida número de folds para validación cruzada."""
        if n_folds < 2:
            return False, "Debe haber al menos 2 folds"
        if n_folds > 20:
            return False, "No se recomienda más de 20 folds (muy costoso computacionalmente)"
        return True, ""

    @staticmethod
    def validate_threshold(threshold: float) -> tuple[bool, str]:
        """Valida umbral de clasificación."""
        if threshold < 0 or threshold > 1:
            return False, "El umbral debe estar entre 0 y 1"
        return True, ""

    @staticmethod
    def validate_target_recall(recall: float) -> tuple[bool, str]:
        """Valida recall objetivo."""
        if recall < 0 or recall > 1:
            return False, "El recall objetivo debe estar entre 0 y 1"
        if recall < 0.5:
            return False, "Un recall menor a 0.5 no es recomendable para predicción médica"
        return True, ""


def load_config() -> Dict[str, Any]:
    """Carga el archivo de configuración YAML."""
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo de configuración en: {CONFIG_PATH}")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"Error al leer el archivo YAML: {e}")
        st.stop()


def save_config(config: Dict[str, Any]) -> bool:
    """Guarda el archivo de configuración YAML."""
    try:
        # Crear backup
        backup_path = CONFIG_PATH.with_suffix('.yaml.backup')
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                backup_content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(backup_content)

        # Guardar nueva configuración
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        return True
    except Exception as e:
        st.error(f"Error al guardar la configuración: {e}")
        return False


def show_general_config(config: Dict[str, Any]):
    """Muestra y permite editar la configuración general."""
    st.header("Configuración General")

    general = config.get('general_config', {})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Versiones")
        version = st.text_input(
            "Versión del Pipeline",
            value=general.get('version', 'v1'),
            help="Controla las rutas de salida de modelos y resultados. Formato: v1, v2, v3...",
            key="pipeline_version"
        )
        valid, msg = ConfigValidator.validate_version(version)
        if not valid:
            st.error(msg)
        else:
            general['version'] = version

        dataset_version = st.text_input(
            "Versión del Dataset",
            value=general.get('dataset_version', 'v1'),
            help="Controla qué datos crudos/procesados se leen. Formato: v1, v2, v3...",
            key="dataset_version"
        )
        valid, msg = ConfigValidator.validate_version(dataset_version)
        if not valid:
            st.error(msg)
        else:
            general['dataset_version'] = dataset_version

    with col2:
        st.subheader("Reproducibilidad")
        random_seed = st.number_input(
            "Semilla Aleatoria",
            value=general.get('random_seed', 42),
            min_value=0,
            step=1,
            help="Garantiza resultados reproducibles. Recomendado: 42",
            key="random_seed"
        )
        general['random_seed'] = int(random_seed)

    with col3:
        st.subheader("Visualización")
        viz = config.get('visualization', {})
        dpi = st.number_input(
            "DPI de Gráficas",
            value=viz.get('dpi', 150),
            min_value=72,
            max_value=300,
            step=10,
            help="Resolución de las gráficas guardadas",
            key="dpi"
        )
        viz['dpi'] = int(dpi)
        config['visualization'] = viz


def show_features_config(config: Dict[str, Any]):
    """Muestra y permite editar las características."""
    st.header("Características (Features)")

    features = config.get('features', {})

    st.subheader("Variable Objetivo")
    target = st.text_input(
        "Nombre de la variable objetivo",
        value=features.get('target_name', 'SHOCK'),
        help="Columna que indica si hubo shock hemorrágico (1) o no (0)",
        key="target_name"
    )
    features['target_name'] = target

    st.divider()

    st.subheader("Features Numéricas")
    st.info("Estas son las variables numéricas del dataset que se usarán para el modelo")
    numerical = features.get('numerical_features', [])
    numerical_text = st.text_area(
        "Variables Numéricas (una por línea)",
        value="\n".join(numerical),
        height=100,
        help="Ingrese cada variable en una línea nueva",
        key="numerical_features"
    )
    features['numerical_features'] = [f.strip() for f in numerical_text.split('\n') if f.strip()]

    st.divider()

    st.subheader("Features Binarias")
    st.info("Variables binarias (0/1) que representan condiciones presentes o ausentes")
    binary = features.get('binary_features', [])
    binary_text = st.text_area(
        "Variables Binarias (una por línea)",
        value="\n".join(binary),
        height=200,
        help="Ingrese cada variable en una línea nueva. Deben tener valores 0 o 1",
        key="binary_features"
    )
    features['binary_features'] = [f.strip() for f in binary_text.split('\n') if f.strip()]

    st.divider()

    st.subheader("Features Agregadas")
    st.info("Variables derivadas que se calculan combinando otras variables (sumas, productos, etc.)")

    aggregated = features.get('aggregated_features', {})

    for feat_name, feat_config in aggregated.items():
        with st.expander(f"{feat_name} - {feat_config.get('description', '')}", expanded=False):
            enabled = st.checkbox(
                "Habilitar esta feature",
                value=feat_config.get('enabled', False),
                key=f"agg_enabled_{feat_name}"
            )
            feat_config['enabled'] = enabled

            if enabled:
                feat_type = st.selectbox(
                    "Tipo de agregación",
                    options=['sum', 'product', 'inverse_binary'],
                    index=['sum', 'product', 'inverse_binary'].index(feat_config.get('type', 'sum')),
                    help="sum: suma de componentes, product: producto, inverse_binary: inverso binario",
                    key=f"agg_type_{feat_name}"
                )
                feat_config['type'] = feat_type

                st.text(f"Componentes: {', '.join(feat_config.get('components', []))}")

    st.divider()

    st.subheader("Features Categóricas")
    st.info("Variables discretizadas usando umbrales sobre variables numéricas")

    categorical = features.get('categorical_features', {})

    for feat_name, feat_config in categorical.items():
        with st.expander(f"{feat_name} - {feat_config.get('description', '')}", expanded=False):
            enabled = st.checkbox(
                "Habilitar esta feature",
                value=feat_config.get('enabled', False),
                key=f"cat_enabled_{feat_name}"
            )
            feat_config['enabled'] = enabled

            if enabled:
                st.text(f"Variable fuente: {feat_config.get('source_variable', '')}")
                st.text(f"Umbrales: {feat_config.get('thresholds', [])}")
                st.json(feat_config.get('categories', []))


def show_cleaning_config(config: Dict[str, Any]):
    """Muestra y permite editar la configuración de limpieza."""
    st.header("Limpieza y Validación de Datos")

    cleaning = config.get('cleaning', {})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Columnas a Excluir")
        st.info("Columnas que se eliminarán del dataset (ej: identificadores, datos de fuga)")
        exclude = cleaning.get('exclude_columns', [])
        exclude_text = st.text_area(
            "Columnas a Excluir (una por línea)",
            value="\n".join(exclude),
            height=100,
            help="Variables que no deben usarse en el modelo",
            key="exclude_columns"
        )
        cleaning['exclude_columns'] = [c.strip() for c in exclude_text.split('\n') if c.strip()]

    with col2:
        st.subheader("Validación de Valores Binarios")
        valid_rules = cleaning.get('validation_rules', {})
        binary_vals = valid_rules.get('binary_values', {}).get('valid', [0, 1])
        st.text_input(
            "Valores válidos para variables binarias",
            value=", ".join(map(str, binary_vals)),
            disabled=True,
            help="Los valores binarios deben ser 0 o 1",
            key="binary_values"
        )

    st.divider()

    st.subheader("Rangos Válidos")

    valid_ranges = valid_rules.get('valid_ranges', {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Edad**")
        edad_range = valid_ranges.get('EDAD', {})
        edad_min = st.number_input(
            "Edad Mínima",
            value=float(edad_range.get('min', 18)),
            min_value=0.0,
            max_value=150.0,
            step=1.0,
            key="edad_min"
        )
        edad_max = st.number_input(
            "Edad Máxima",
            value=float(edad_range.get('max', 120)),
            min_value=0.0,
            max_value=150.0,
            step=1.0,
            key="edad_max"
        )

        valid, msg = ConfigValidator.validate_age_range(edad_min, edad_max)
        if not valid:
            st.error(msg)
        else:
            valid_ranges['EDAD'] = {'min': edad_min, 'max': edad_max}

    with col2:
        st.markdown("**Hemoglobina Pre-Quirúrgica**")
        hb_range = valid_ranges.get('HB_PREQX', {})
        hb_min = st.number_input(
            "Hemoglobina Mínima (g/dL)",
            value=float(hb_range.get('min', 3.0)),
            min_value=0.0,
            max_value=30.0,
            step=0.1,
            key="hb_min"
        )
        hb_max = st.number_input(
            "Hemoglobina Máxima (g/dL)",
            value=float(hb_range.get('max', 20.0)),
            min_value=0.0,
            max_value=30.0,
            step=0.1,
            key="hb_max"
        )

        valid, msg = ConfigValidator.validate_hemoglobin_range(hb_min, hb_max)
        if not valid:
            st.error(msg)
        else:
            valid_ranges['HB_PREQX'] = {'min': hb_min, 'max': hb_max}

    valid_rules['valid_ranges'] = valid_ranges
    cleaning['validation_rules'] = valid_rules


def show_data_split_config(config: Dict[str, Any]):
    """Muestra y permite editar la configuración de partición de datos."""
    st.header("Partición de Datos (Train/Test Split)")

    split_config = config.get('data_split', {})

    col1, col2, col3 = st.columns(3)

    with col1:
        test_size = st.slider(
            "Proporción del Conjunto de Prueba",
            min_value=0.1,
            max_value=0.5,
            value=split_config.get('test_size', 0.2),
            step=0.05,
            help="Porcentaje de datos para prueba (test set)",
            key="test_size"
        )
        valid, msg = ConfigValidator.validate_test_size(test_size)
        if not valid:
            st.error(msg)
        else:
            split_config['test_size'] = test_size
            st.info(f"Train: {(1-test_size)*100:.0f}% | Test: {test_size*100:.0f}%")

    with col2:
        shuffle = st.checkbox(
            "Mezclar Datos",
            value=split_config.get('shuffle', True),
            help="Aleatorizar el orden antes de particionar",
            key="shuffle"
        )
        split_config['shuffle'] = shuffle

    with col3:
        stratify = st.checkbox(
            "Partición Estratificada",
            value=split_config.get('stratify', True),
            help="Mantener la misma proporción de clases en train y test",
            key="stratify"
        )
        split_config['stratify'] = stratify
        if stratify:
            st.success("Recomendado para datos desbalanceados")


def show_cv_config(config: Dict[str, Any]):
    """Muestra y permite editar la configuración de validación cruzada."""
    st.header("Validación Cruzada (Cross-Validation)")

    cv_config = config.get('cross_validation', {})

    col1, col2, col3 = st.columns(3)

    with col1:
        n_folds = st.number_input(
            "Número de Folds",
            value=cv_config.get('n_folds', 10),
            min_value=2,
            max_value=20,
            step=1,
            help="Particiones para validación cruzada",
            key="cv_n_folds"
        )
        valid, msg = ConfigValidator.validate_cv_folds(n_folds)
        if not valid:
            st.error(msg)
        else:
            cv_config['n_folds'] = int(n_folds)

    with col2:
        scale_features = st.checkbox(
            "Escalar Features",
            value=cv_config.get('scale_features', True),
            help="Normalizar variables numéricas (recomendado)",
            key="scale_features"
        )
        cv_config['scale_features'] = scale_features

    with col3:
        n_jobs = st.number_input(
            "Procesamiento Paralelo",
            value=cv_config.get('n_jobs', -1),
            min_value=-1,
            max_value=16,
            step=1,
            help="-1 usa todos los cores disponibles",
            key="cv_n_jobs"
        )
        cv_config['n_jobs'] = int(n_jobs)


def show_models_config(config: Dict[str, Any]):
    """Muestra y permite editar la configuración de modelos."""
    st.header("Configuración de Modelos")

    models = config.get('models', {})

    model_descriptions = {
        'lightgbm': 'Gradient Boosting - Alto rendimiento, ideal para datos tabulares',
        'decision_tree': 'Árbol de Decisión - Interpretable y simple',
        'logistic_regression': 'Regresión Logística - Modelo lineal clásico',
        'naive_bayes': 'Naive Bayes - Probabilístico, rápido'
    }

    for model_name, model_config in models.items():
        status = 'ACTIVO' if model_config.get('enabled') else 'DESACTIVADO'
        with st.expander(
            f"{model_descriptions.get(model_name, model_name)} [{status}]",
            expanded=model_config.get('enabled', False)
        ):
            col1, col2 = st.columns([1, 3])

            with col1:
                enabled = st.checkbox(
                    "Habilitar Modelo",
                    value=model_config.get('enabled', True),
                    key=f"model_enabled_{model_name}"
                )
                model_config['enabled'] = enabled

            with col2:
                if enabled:
                    target_recall = st.slider(
                        "Recall Objetivo Mínimo",
                        min_value=0.5,
                        max_value=1.0,
                        value=model_config.get('target_recall', 0.80),
                        step=0.05,
                        help="Sensibilidad mínima requerida (importante en contexto médico)",
                        key=f"target_recall_{model_name}"
                    )
                    valid, msg = ConfigValidator.validate_target_recall(target_recall)
                    if not valid:
                        st.error(msg)
                    else:
                        model_config['target_recall'] = target_recall

            if enabled:
                st.info(f"Módulo: {model_config.get('module')} | Clase: {model_config.get('class')}")

                st.markdown("**Parámetros Principales:**")
                params = model_config.get('params', {})

                if model_name == 'lightgbm':
                    pcol1, pcol2, pcol3 = st.columns(3)
                    with pcol1:
                        params['n_estimators'] = st.number_input(
                            "N° Estimadores",
                            value=params.get('n_estimators', 400),
                            min_value=50,
                            max_value=1000,
                            step=50,
                            key=f"lgb_n_est_{model_name}"
                        )
                    with pcol2:
                        params['learning_rate'] = st.number_input(
                            "Learning Rate",
                            value=params.get('learning_rate', 0.03),
                            min_value=0.001,
                            max_value=0.3,
                            step=0.01,
                            format="%.3f",
                            key=f"lgb_lr_{model_name}"
                        )
                    with pcol3:
                        params['max_depth'] = st.number_input(
                            "Profundidad Máxima",
                            value=params.get('max_depth', 7),
                            min_value=-1,
                            max_value=20,
                            step=1,
                            help="-1 = sin límite",
                            key=f"lgb_depth_{model_name}"
                        )

                elif model_name == 'decision_tree':
                    pcol1, pcol2 = st.columns(2)
                    with pcol1:
                        params['max_depth'] = st.number_input(
                            "Profundidad Máxima",
                            value=params.get('max_depth', 5),
                            min_value=1,
                            max_value=20,
                            step=1,
                            key=f"dt_depth_{model_name}"
                        )
                    with pcol2:
                        params['min_samples_split'] = st.number_input(
                            "Mín. Muestras para Split",
                            value=params.get('min_samples_split', 20),
                            min_value=2,
                            max_value=100,
                            step=5,
                            key=f"dt_split_{model_name}"
                        )

                elif model_name == 'logistic_regression':
                    pcol1, pcol2 = st.columns(2)
                    with pcol1:
                        params['C'] = st.number_input(
                            "C (Regularización Inversa)",
                            value=params.get('C', 1.0),
                            min_value=0.001,
                            max_value=100.0,
                            step=0.1,
                            key=f"lr_c_{model_name}"
                        )
                    with pcol2:
                        params['penalty'] = st.selectbox(
                            "Tipo de Penalización",
                            options=['l1', 'l2', 'elasticnet'],
                            index=['l1', 'l2', 'elasticnet'].index(params.get('penalty', 'l2')),
                            key=f"lr_pen_{model_name}"
                        )


def show_hyperparameter_search_config(config: Dict[str, Any]):
    """Muestra y permite editar la configuración de búsqueda de hiperparámetros."""
    st.header("Búsqueda de Hiperparámetros")

    hp_config = config.get('hyperparameter_search', {})

    col1, col2 = st.columns([1, 3])

    with col1:
        enabled = st.checkbox(
            "Habilitar Búsqueda",
            value=hp_config.get('enabled', False),
            help="Buscar automáticamente los mejores hiperparámetros (aumenta tiempo de ejecución)",
            key="hp_enabled"
        )
        hp_config['enabled'] = enabled

    with col2:
        if enabled:
            st.warning("La búsqueda de hiperparámetros aumentará significativamente el tiempo de entrenamiento")

    if enabled:
        col1, col2, col3 = st.columns(3)

        with col1:
            n_iter = st.number_input(
                "N° Iteraciones",
                value=hp_config.get('n_iter', 70),
                min_value=10,
                max_value=200,
                step=10,
                help="Combinaciones de parámetros a probar",
                key="hp_n_iter"
            )
            hp_config['n_iter'] = int(n_iter)

        with col2:
            cv_folds = st.number_input(
                "Folds para CV",
                value=hp_config.get('cv_folds', 10),
                min_value=3,
                max_value=20,
                step=1,
                key="hp_cv_folds"
            )
            hp_config['cv_folds'] = int(cv_folds)

        with col3:
            scoring = st.selectbox(
                "Métrica de Optimización",
                options=['f1', 'f2', 'recall', 'precision', 'roc_auc'],
                index=['f1', 'f2', 'recall', 'precision', 'roc_auc'].index(hp_config.get('scoring', 'f2')),
                help="F2 prioriza recall sobre precisión",
                key="hp_scoring"
            )
            hp_config['scoring'] = scoring


def show_threshold_optimization_config(config: Dict[str, Any]):
    """Muestra y permite editar la configuración de optimización de umbral."""
    st.header("Optimización de Umbral de Clasificación")

    thresh_config = config.get('threshold_optimization', {})

    st.info("El umbral determina a partir de qué probabilidad se clasifica como shock positivo")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Búsqueda de Umbral Óptimo")
        search = thresh_config.get('search_thresholds', [0.20, 0.55, 0.01])

        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            search_min = st.number_input(
                "Umbral Mínimo",
                value=search[0],
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key="thresh_min"
            )
        with scol2:
            search_max = st.number_input(
                "Umbral Máximo",
                value=search[1],
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key="thresh_max"
            )
        with scol3:
            search_step = st.number_input(
                "Paso",
                value=search[2],
                min_value=0.01,
                max_value=0.1,
                step=0.01,
                format="%.2f",
                key="thresh_step"
            )

        if search_min >= search_max:
            st.error("El umbral mínimo debe ser menor que el máximo")
        else:
            thresh_config['search_thresholds'] = [search_min, search_max, search_step]

    with col2:
        st.subheader("Umbrales para Comparación en Test")
        test_thresh = thresh_config.get('test_thresholds', [0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
        test_thresh_text = st.text_input(
            "Umbrales a Comparar (separados por comas)",
            value=", ".join(map(str, test_thresh)),
            help="Umbrales que se evaluarán en el conjunto de prueba",
            key="test_thresholds"
        )
        try:
            test_thresholds = [float(t.strip()) for t in test_thresh_text.split(',')]
            all_valid = all(0 <= t <= 1 for t in test_thresholds)
            if all_valid:
                thresh_config['test_thresholds'] = test_thresholds
            else:
                st.error("Todos los umbrales deben estar entre 0 y 1")
        except ValueError:
            st.error("Formato inválido. Use números separados por comas")

    st.divider()

    permutation_n = st.number_input(
        "N° Permutaciones para Test de Significancia",
        value=thresh_config.get('permutation_n', 200),
        min_value=50,
        max_value=1000,
        step=50,
        help="Número de permutaciones para evaluar significancia estadística del ROC AUC",
        key="permutation_n"
    )
    thresh_config['permutation_n'] = int(permutation_n)


def show_feature_pruning_config(config: Dict[str, Any]):
    """Muestra y permite editar la configuración de poda de características."""
    st.header("Poda de Características")

    pruning_config = config.get('feature_pruning', {})

    col1, col2 = st.columns([1, 3])

    with col1:
        enabled = st.checkbox(
            "Habilitar Poda",
            value=pruning_config.get('enabled', True),
            help="Elimina features binarias con muy baja prevalencia",
            key="pruning_enabled"
        )
        pruning_config['enabled'] = enabled

    with col2:
        if enabled:
            min_ones = st.number_input(
                "Mínimo de Unos para Mantener Feature",
                value=pruning_config.get('min_total_ones', 10),
                min_value=1,
                max_value=100,
                step=5,
                help="Features binarias con menos de este número de 1s serán eliminadas",
                key="min_total_ones"
            )
            pruning_config['min_total_ones'] = int(min_ones)


def show_logging_config(config: Dict[str, Any]):
    """Muestra y permite editar la configuración de logging."""
    st.header("Configuración de Logs")

    logging_config = config.get('logging', {})

    col1, col2 = st.columns(2)

    with col1:
        log_level = st.selectbox(
            "Nivel de Log",
            options=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            index=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'].index(logging_config.get('level', 'INFO')),
            help="INFO es recomendado para ejecución normal",
            key="log_level"
        )
        logging_config['level'] = log_level

    with col2:
        log_dir = st.text_input(
            "Directorio de Logs",
            value=logging_config.get('log_dir', 'logs'),
            help="Carpeta donde se guardarán los logs",
            key="log_dir"
        )
        logging_config['log_dir'] = log_dir

    level_descriptions = {
        'DEBUG': 'Muy detallado - para debugging',
        'INFO': 'Información general - recomendado',
        'WARNING': 'Solo advertencias y errores',
        'ERROR': 'Solo errores',
        'CRITICAL': 'Solo errores críticos'
    }
    st.info(level_descriptions.get(log_level, ''))


def main():
    """Función principal de la aplicación."""

    st.title("Editor de Configuración del Pipeline")
    st.markdown("**Predicción de Shock Hemorrágico** - Interfaz de configuración visual y validada")

    with st.sidebar:
        st.header("Navegación")

        sections = {
            "General": "general",
            "Características": "features",
            "Limpieza": "cleaning",
            "Partición": "split",
            "Validación Cruzada": "cv",
            "Modelos": "models",
            "Búsqueda HP": "hp_search",
            "Optimización Umbral": "threshold",
            "Poda Features": "pruning",
            "Logging": "logging"
        }

        selected_section = st.radio(
            "Seleccione una sección:",
            list(sections.keys()),
            key="section_selector"
        )

        st.divider()

        st.info(f"**Archivo:** {CONFIG_PATH.name}")
        st.caption(f"Ubicación: {CONFIG_PATH.parent}")

        st.divider()

        if st.button("Guardar Configuración", type="primary", use_container_width=True):
            st.session_state.save_requested = True

        if st.button("Recargar desde Archivo", use_container_width=True):
            st.session_state.config = load_config()
            st.success("Configuración recargada")
            st.rerun()

        st.divider()
        st.caption("v1.0.0 - PDG Shock Hemorrágico")

    if 'config' not in st.session_state:
        st.session_state.config = load_config()

    config = st.session_state.config

    section_key = sections[selected_section]

    if section_key == "general":
        show_general_config(config)
    elif section_key == "features":
        show_features_config(config)
    elif section_key == "cleaning":
        show_cleaning_config(config)
    elif section_key == "split":
        show_data_split_config(config)
    elif section_key == "cv":
        show_cv_config(config)
    elif section_key == "models":
        show_models_config(config)
    elif section_key == "hp_search":
        show_hyperparameter_search_config(config)
    elif section_key == "threshold":
        show_threshold_optimization_config(config)
    elif section_key == "pruning":
        show_feature_pruning_config(config)
    elif section_key == "logging":
        show_logging_config(config)

    if st.session_state.get('save_requested', False):
        with st.spinner("Guardando configuración..."):
            if save_config(config):
                st.success("Configuración guardada exitosamente")
            else:
                st.error("Error al guardar la configuración")
        st.session_state.save_requested = False


if __name__ == "__main__":
    main()
