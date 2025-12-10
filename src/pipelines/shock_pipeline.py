"""
Pipeline principal para entrenamiento y evaluacion de modelo de shock hemorragico.
"""

import os
import random
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from src.preprocessing.data_cleaner import ShockDataCleaner
from src.preprocessing.feature_engineering import FeatureTransformer
from src.models.shock_classifier import ShockClassifier
from src.evaluation.metrics import ModelEvaluator
from src.visualization.plots import ShockVisualizer
from src.utils.helpers import save_json, bootstrap_ci
from sklearn.metrics import recall_score

# Imports
import mlflow
import mlflow.sklearn
mlflow.sklearn.autolog()  # Activar autologging para scikit-learn

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ShockPredictionPipeline:
    """Pipeline completo para prediccion de shock hemorragico."""

    def __init__(
        self,
        data_path: str,
        output_dir: str = "reports/shock_model",
        model_dir: str = "models",
        seed: int = 42,
    ):
        """
        Inicializa el pipeline.

        Args:
            data_path: Ruta al archivo CSV de datos
            output_dir: Directorio de salida para reportes y visualizaciones
            model_dir: Directorio para guardar modelos
            seed: Semilla aleatoria
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.seed = seed

        # Configurar semilla
        np.random.seed(seed)
        random.seed(seed)

        # Crear directorios de salida
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # Inicializar componentes
        self.cleaner = ShockDataCleaner()
        self.transformer = None
        self.classifier = None
        self.evaluator = ModelEvaluator()
        self.visualizer = ShockVisualizer(output_dir)

        # Datos
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Resultados
        self.best_model_name = None
        self.tuned_threshold = None

        # Parámetros del pipeline
        self.cv_splits = None

    def load_and_clean_data(self):
        """Carga y limpia los datos."""
        print("\n" + "=" * 60)
        print("CARGA Y LIMPIEZA DE DATOS")
        print("=" * 60)

        # Cargar datos
        self.df = pd.read_csv(self.data_path)
        print(f"Datos cargados: {self.df.shape[0]} filas x {self.df.shape[1]} columnas")

        # Detectar target
        target_col = self.cleaner.detect_target(self.df)
        print(f"Columna objetivo detectada: {target_col}")

        # Identificar variables de leakage
        postop_vars = self.cleaner.identify_leakage_vars(self.df)
        print(f"Variables postoperatorias excluidas ({len(postop_vars)}): {postop_vars}")

        # Preparar features
        X, y = self.cleaner.prepare_features(self.df)
        print(f"Features preparados: {X.shape[1]} columnas")
        print(f"Distribucion del target: {y.value_counts().to_dict()}")

        # Guardar resumen estadistico
        if self.cleaner.continuous_cols:
            desc_path = os.path.join(self.output_dir, "continuous_describe.csv")
            self.df[self.cleaner.continuous_cols].describe().to_csv(desc_path)
            print(f"Estadisticas descriptivas guardadas en: {desc_path}")

        return X, y

    def split_data(self, X, y, test_size: float = 0.20):
        """Divide datos en train y test."""
        print("\n" + "=" * 60)
        print("DIVISION TRAIN/TEST")
        print("=" * 60)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.seed
        )

        print(f"Train: {self.X_train.shape}")
        print(f"Test: {self.X_test.shape}")
        print(f"Distribucion train: {self.y_train.value_counts().to_dict()}")
        print(f"Distribucion test: {self.y_test.value_counts().to_dict()}")

    def build_preprocessor(self):
        """Construye el preprocesador."""
        print("\n" + "=" * 60)
        print("CONSTRUCCION DEL PREPROCESADOR")
        print("=" * 60)

        num_cols, cat_cols = self.cleaner.get_feature_split(self.X_train)
        print(f"Columnas numericas: {len(num_cols)}")
        print(f"Columnas categoricas/binarias: {len(cat_cols)}")

        # Visualizar distribuciones
        if num_cols:
            self.visualizer.plot_numeric_distributions(self.df, num_cols, self.cleaner.target_col)

        # Construir transformer
        self.transformer = FeatureTransformer(num_cols, cat_cols)
        preprocessor = self.transformer.build_preprocessor()

        return preprocessor

    def train_models(self, cv_splits: int = 5):
        """Entrena multiples modelos con GridSearchCV."""
        print("\n" + "=" * 60)
        print("ENTRENAMIENTO DE MODELOS")
        print("=" * 60)

        # Almacenar el número de splits para uso posterior
        self.cv_splits = cv_splits

        # Construir preprocesador si no existe
        if self.transformer is None:
            preprocessor = self.build_preprocessor()
        else:
            preprocessor = self.transformer.get_preprocessor()

        # Inicializar clasificador
        self.classifier = ShockClassifier(preprocessor, self.seed)

        # Construir grids
        self.classifier.build_grids(cv_splits=cv_splits)

        # Entrenar todos los modelos
        fitted_grids = self.classifier.fit_all_models(self.X_train, self.y_train)

        return fitted_grids

    def evaluate_models(self):
        """Evalua modelos en conjunto de test."""
        print("\n" + "=" * 60)
        print("EVALUACION EN TEST")
        print("=" * 60)

        # Evaluar todos los modelos
        evals = self.evaluator.evaluate_all_models(
            self.classifier.fitted_grids, self.X_test, self.y_test
        )

        # Guardar resumen
        summary = self.evaluator.get_summary()
        summary_path = os.path.join(self.output_dir, "model_summary.json")
        save_json(summary, summary_path)
        print(f"\nResumen guardado en: {summary_path}")

        return evals

    def select_and_calibrate_best_model(self):
        """Selecciona y calibra el mejor modelo."""
        print("\n" + "=" * 60)
        print("SELECCION Y CALIBRACION DEL MEJOR MODELO")
        print("=" * 60)

        # Seleccionar mejor modelo
        evals = self.evaluator.get_evaluations()
        self.best_model_name = self.classifier.select_best_model(evals)

        # Calibrar modelo
        self.classifier.calibrate_model(self.X_train, self.y_train, method="isotonic")

        # Evaluar modelo calibrado
        proba_calibrated = self.classifier.predict_calibrated(self.X_test)
        self.evaluator.evaluate_calibrated_model(
            self.best_model_name, proba_calibrated, self.y_test
        )

        return proba_calibrated

    def tune_threshold(self, proba_calibrated, min_spec: float = 0.6):
        """Optimiza umbral de decision."""
        print("\n" + "=" * 60)
        print("OPTIMIZACION DE UMBRAL")
        print("=" * 60)

        self.tuned_threshold = self.evaluator.tune_threshold_clinical(
            self.y_test.values, proba_calibrated, min_spec=min_spec
        )

        # Calcular CI bootstrap para recall
        yhat_tuned = (proba_calibrated >= self.tuned_threshold["th"]).astype(int)
        rec = recall_score(self.y_test, yhat_tuned)
        rec_ci = bootstrap_ci(
            lambda a, b: recall_score(a, b),
            self.y_test.values,
            yhat_tuned,
            n_boot=1000,
            seed=self.seed,
        )
        print(f"\nRecall en umbral {self.tuned_threshold['th']:.3f}: {rec:.3f}")
        print(f"95% CI: [{rec_ci[0]:.3f}, {rec_ci[1]:.3f}]")

        return self.tuned_threshold

    def generate_visualizations(self):
        """Genera todas las visualizaciones."""
        print("\n" + "=" * 60)
        print("GENERACION DE VISUALIZACIONES")
        print("=" * 60)

        evals = self.evaluator.get_evaluations()
        best_eval = evals[self.best_model_name]

        # Curva ROC
        roc_path = self.visualizer.plot_roc_curve(self.y_test, best_eval["proba_test"], self.best_model_name)

        # Curva Precision-Recall
        pr_path = self.visualizer.plot_precision_recall_curve(
            self.y_test, best_eval["proba_test"], self.best_model_name
        )

        # Matriz de confusion
        cm_path = self.visualizer.plot_confusion_matrix(best_eval["confusion"], self.best_model_name)

        # Curva de calibracion
        if "proba_calibrated" in best_eval:
            cal_path = self.visualizer.plot_calibration_curve(
                self.y_test,
                best_eval["proba_test"],
                best_eval["proba_calibrated"],
                self.best_model_name,
            )
        else:
            cal_path = None

        # Analisis de umbrales
        proba_for_analysis = best_eval.get("proba_calibrated", best_eval["proba_test"])
        th_path = self.visualizer.plot_threshold_analysis(self.y_test, proba_for_analysis)

        # Registrar los artefactos en MLflow
        if mlflow.active_run():
            for path in [roc_path, pr_path, cm_path, cal_path, th_path]:
                if path and os.path.exists(path):
                    mlflow.log_artifact(path)

    def compute_feature_importance(self):
        """Calcula importancia de features."""
        print("\n" + "=" * 60)
        print("IMPORTANCIA DE FEATURES")
        print("=" * 60)

        explain_dir = os.path.join(self.output_dir, "explain")
        os.makedirs(explain_dir, exist_ok=True)

        best_estimator = self.classifier.get_best_pipeline()

        # Intentar SHAP si esta disponible
        if SHAP_AVAILABLE and hasattr(best_estimator.named_steps["clf"], "feature_importances_"):
            try:
                print("Calculando SHAP values...")
                X_test_trans = best_estimator.named_steps["pre"].transform(self.X_test)
                explainer = shap.TreeExplainer(best_estimator.named_steps["clf"])
                shap_values = explainer.shap_values(X_test_trans)
                shap.summary_plot(shap_values, X_test_trans, show=False)
                import matplotlib.pyplot as plt

                shap_path = os.path.join(explain_dir, "shap_summary.png")
                plt.savefig(shap_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"SHAP summary guardado en: {shap_path}")

                # Registrar en MLflow si está activo
                if mlflow.active_run():
                    mlflow.log_artifact(shap_path)

                return
            except Exception as e:
                print(f"SHAP fallo: {e}. Usando permutation importance...")

        # Permutation importance como fallback
        print("Calculando permutation importance...")
        r = permutation_importance(
            best_estimator,
            self.X_test,
            self.y_test,
            n_repeats=30,
            random_state=self.seed,
            n_jobs=-1,
        )

        imp_df = pd.DataFrame(
            {
                "feature": self.X_test.columns,
                "importance_mean": r.importances_mean,
                "importance_std": r.importances_std,
            }
        )
        imp_df = imp_df.sort_values("importance_mean", ascending=False)

        # Guardar CSV
        imp_path = os.path.join(explain_dir, "permutation_importance.csv")
        imp_df.head(20).to_csv(imp_path, index=False)
        print(f"Permutation importance guardado en: {imp_path}")

        # Visualizar
        imp_plot_path = self.visualizer.plot_feature_importance(imp_df, top_n=20)

        # Registrar en MLflow si está activo
        if mlflow.active_run():
            mlflow.log_artifact(imp_path)
            if imp_plot_path and os.path.exists(imp_plot_path):
                mlflow.log_artifact(imp_plot_path)

    def save_artifacts(self):
        """Guarda artefactos del modelo."""
        print("\n" + "=" * 60)
        print("GUARDADO DE ARTEFACTOS")
        print("=" * 60)

        evals = self.evaluator.get_evaluations()
        best_eval = evals[self.best_model_name]

        # Metadata
        artifact_metadata = {
            "model_name": self.best_model_name,
            "timestamp": datetime.now().isoformat(),
            "tuned_threshold": self.tuned_threshold,
            "test_metrics": {
                "recall": best_eval["recall"],
                "precision": best_eval["precision"],
                "auc": best_eval["auc"],
                "specificity": best_eval["specificity"],
            },
            "calibrated_metrics": {
                "recall_cal": best_eval.get("recall_cal"),
                "precision_cal": best_eval.get("precision_cal"),
                "auc_cal": best_eval.get("auc_cal"),
                "spec_cal": best_eval.get("spec_cal"),
            },
            "best_params": best_eval["best_params"],
            "notes": "SMOTE aplicado dentro de CV para evitar leakage; preprocesador ajustado solo en train; umbral optimizado para especificidad >= 0.6.",
        }

        # Guardar pipeline y metadata
        artifact = {
            "pipeline": self.classifier.get_best_pipeline(),
            "calibrator": self.classifier.get_calibrator(),
            "metadata": artifact_metadata,
        }

        # Guardar modelo en directorio models/
        artifact_path = os.path.join(self.model_dir, f"shock_{self.best_model_name}.joblib")
        joblib.dump(artifact, artifact_path)
        print(f"Modelo guardado en: {artifact_path}")

        # Guardar metadata JSON en reports/
        metadata_path = os.path.join(self.output_dir, "model_metadata.json")
        save_json(artifact_metadata, metadata_path)
        print(f"Metadata guardada en: {metadata_path}")

    def log_to_mlflow(self):
        """Registra experimento en MLflow."""
        print("\n" + "=" * 60)
        print("LOGGING EN MLFLOW")
        print("=" * 60)

        # Configurar MLflow tracking
        # Permitir la configuración del tracking URI a través de variable de entorno
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)

        evals = self.evaluator.get_evaluations()
        best_eval = evals[self.best_model_name]

        # Configurar experimento
        experiment_name = "shock_prediction_experiments"
        experiment = mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"final_{self.best_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parámetros del modelo
            mlflow.log_params(best_eval["best_params"])

            # Log parámetros del pipeline
            mlflow.log_param("test_size", len(self.X_test) / (len(self.X_train) + len(self.X_test)))
            mlflow.log_param("train_samples", len(self.X_train))
            mlflow.log_param("test_samples", len(self.X_test))
            mlflow.log_param("seed", self.seed)

            # Log metadatos adicionales
            mlflow.log_param("total_features", self.X_train.shape[1])
            mlflow.log_param("total_samples", len(self.df))
            mlflow.log_param("numeric_features_count", len(self.cleaner.continuous_cols))
            mlflow.log_param("categorical_features_count", len(self.cleaner.categorical_cols))
            mlflow.log_param("target_distribution_pos", int(self.y_train.sum() + self.y_test.sum()))
            mlflow.log_param("target_distribution_neg", int(len(self.y_train) + len(self.y_test) - (self.y_train.sum() + self.y_test.sum())))
            mlflow.log_param("cv_folds", self.cv_splits or 5)  # Si cv_splits no está disponible, se asume 5

            # Incluir nombres de features como tag
            feature_names = ', '.join(list(self.X_train.columns))
            mlflow.set_tag("features", feature_names[:5000])  # Limitar longitud

            # Log métricas principales
            mlflow.log_metrics(
                {
                    "recall_test": best_eval["recall"],
                    "precision_test": best_eval["precision"],
                    "auc_test": best_eval["auc"],
                    "specificity_test": best_eval["specificity"],
                    "f1_score_test": best_eval["f1"],
                    "balanced_accuracy_test": best_eval["balanced_acc"],
                    "recall_cal": best_eval.get("recall_cal", 0.0),
                    "precision_cal": best_eval.get("precision_cal", 0.0),
                    "auc_cal": best_eval.get("auc_cal", 0.0),
                    "specificity_cal": best_eval.get("spec_cal", 0.0),
                }
            )

            # Log métricas del umbral optimizado
            if self.tuned_threshold:
                mlflow.log_metrics({
                    "optimal_threshold": self.tuned_threshold["th"],
                    "threshold_sensitivity": self.tuned_threshold["sens"],
                    "threshold_specificity": self.tuned_threshold["spec"]
                })

            # Log modelo con MLflow
            mlflow.sklearn.log_model(
                sk_model=self.classifier.get_best_pipeline(),
                artifact_path="model_pipeline",
                conda_env="./environment.yml"  # Asumiendo que existe un archivo environment.yml
            )

            # Log artifacts adicionales
            mlflow.log_artifact(os.path.join(self.output_dir, "model_metadata.json"))

            print(f"Experimento registrado en MLflow - Experiment ID: {experiment.experiment_id}")
            print(f"Run ID: {mlflow.active_run().info.run_id}")

    def run_full_pipeline(self, test_size: float = 0.20, cv_splits: int = 5, min_spec: float = 0.6):
        """
        Ejecuta el pipeline completo.

        Args:
            test_size: Proporcion de datos para test
            cv_splits: Numero de folds para CV
            min_spec: Especificidad minima para optimizacion de umbral
        """
        print("\n" + "=" * 60)
        print("INICIANDO PIPELINE DE PREDICCION DE SHOCK HEMORRAGICO")
        print("=" * 60)

        # 1. Cargar y limpiar datos
        X, y = self.load_and_clean_data()

        # 2. Dividir datos
        self.split_data(X, y, test_size=test_size)

        # 3. Construir preprocesador
        self.build_preprocessor()

        # 4. Entrenar modelos
        self.train_models(cv_splits=cv_splits)

        # 5. Evaluar modelos
        self.evaluate_models()

        # 6. Seleccionar y calibrar mejor modelo
        proba_calibrated = self.select_and_calibrate_best_model()

        # 7. Optimizar umbral
        self.tune_threshold(proba_calibrated, min_spec=min_spec)

        # Iniciar logging de MLflow
        self.log_to_mlflow()

        # 8. Generar visualizaciones
        self.generate_visualizations()

        # 9. Calcular importancia de features
        self.compute_feature_importance()

        # 10. Guardar artefactos
        self.save_artifacts()

        # En este punto, MLflow logging ya ha sido realizado

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETADO")
        print("=" * 60)
        print(f"Revise los resultados en: {self.output_dir}")


def main():
    """Funcion principal para ejecutar el pipeline."""
    # Configuracion
    DATA_PATH = "data/shock.csv"
    OUTPUT_DIR = "reports/shock_model"
    SEED = 42

    # Crear y ejecutar pipeline
    pipeline = ShockPredictionPipeline(data_path=DATA_PATH, output_dir=OUTPUT_DIR, seed=SEED)

    pipeline.run_full_pipeline(test_size=0.20, cv_splits=5, min_spec=0.6)


if __name__ == "__main__":
    main()
