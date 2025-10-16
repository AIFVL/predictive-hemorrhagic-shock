"""
Modulo de metricas y evaluacion de modelos.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
from typing import Dict, Tuple, Any


class ModelEvaluator:
    """Clase para evaluacion de modelos de clasificacion."""

    def __init__(self):
        """Inicializa el evaluador."""
        self.evaluations = {}

    def evaluate_model(
        self,
        name: str,
        estimator: Any,
        X_test,
        y_test,
        best_params: Dict = None,
        cv_score: float = None
    ) -> Dict:
        """
        Evalua un modelo en el conjunto de test.

        Args:
            name: Nombre del modelo
            estimator: Estimador entrenado
            X_test: Features de test
            y_test: Target de test
            best_params: Mejores parametros del GridSearch
            cv_score: Score promedio de cross-validation

        Returns:
            Dict: Diccionario con metricas de evaluacion
        """
        # Predicciones
        proba_test = estimator.predict_proba(X_test)[:, 1]
        yhat_test = (proba_test >= 0.5).astype(int)

        # Metricas basicas
        rec = recall_score(y_test, yhat_test)
        prec = precision_score(y_test, yhat_test, zero_division=0)
        auc = roc_auc_score(y_test, proba_test)

        # Matriz de confusion
        tn, fp, fn, tp = confusion_matrix(y_test, yhat_test).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Guardar evaluacion
        eval_dict = {
            'estimator': estimator,
            'proba_test': proba_test,
            'yhat_test': yhat_test,
            'recall': float(rec),
            'precision': float(prec),
            'auc': float(auc),
            'specificity': float(spec),
            'confusion': (int(tn), int(fp), int(fn), int(tp)),
            'best_params': best_params or {},
            'cv_recall_mean': float(cv_score) if cv_score else None
        }

        self.evaluations[name] = eval_dict

        print(f"\n{name} - Test Metrics:")
        print(f"  Recall: {rec:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Specificity: {spec:.4f}")
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        return eval_dict

    def evaluate_all_models(self, fitted_grids: Dict, X_test, y_test) -> Dict:
        """
        Evalua todos los modelos entrenados.

        Args:
            fitted_grids: Diccionario de GridSearchCV entrenados
            X_test: Features de test
            y_test: Target de test

        Returns:
            Dict: Diccionario con evaluaciones de todos los modelos
        """
        for name, grid in fitted_grids.items():
            self.evaluate_model(
                name=name,
                estimator=grid.best_estimator_,
                X_test=X_test,
                y_test=y_test,
                best_params=grid.best_params_,
                cv_score=grid.best_score_
            )

        return self.evaluations

    def evaluate_calibrated_model(
        self,
        model_name: str,
        proba_calibrated: np.ndarray,
        y_test
    ) -> Dict:
        """
        Evalua el modelo calibrado.

        Args:
            model_name: Nombre del modelo
            proba_calibrated: Probabilidades calibradas
            y_test: Target de test

        Returns:
            Dict: Metricas del modelo calibrado
        """
        yhat_cal = (proba_calibrated >= 0.5).astype(int)

        rec_cal = recall_score(y_test, yhat_cal)
        prec_cal = precision_score(y_test, yhat_cal, zero_division=0)
        auc_cal = roc_auc_score(y_test, proba_calibrated)

        tn, fp, fn, tp = confusion_matrix(y_test, yhat_cal).ravel()
        spec_cal = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        cal_metrics = {
            'proba_calibrated': proba_calibrated,
            'recall_cal': float(rec_cal),
            'precision_cal': float(prec_cal),
            'auc_cal': float(auc_cal),
            'spec_cal': float(spec_cal),
            'confusion_cal': (int(tn), int(fp), int(fn), int(tp))
        }

        # Actualizar evaluacion existente
        if model_name in self.evaluations:
            self.evaluations[model_name].update(cal_metrics)

        print(f"\n{model_name} - Calibrated Metrics:")
        print(f"  Recall: {rec_cal:.4f}")
        print(f"  Precision: {prec_cal:.4f}")
        print(f"  AUC: {auc_cal:.4f}")
        print(f"  Specificity: {spec_cal:.4f}")

        return cal_metrics

    def tune_threshold_clinical(
        self,
        y_true,
        probs: np.ndarray,
        min_spec: float = 0.55,
        prefer_recall: bool = True
    ) -> Dict:
        """
        Optimiza el umbral de decision con criterios clinicos.

        Args:
            y_true: Valores verdaderos
            probs: Probabilidades predichas
            min_spec: Especificidad minima requerida
            prefer_recall: Si priorizar recall sobre precision

        Returns:
            Dict: Diccionario con umbral optimo y metricas
        """
        fpr, tpr, roc_th = roc_curve(y_true, probs)
        prec, rec, pr_th = precision_recall_curve(y_true, probs)
        candidates = np.unique(np.concatenate([roc_th, pr_th]))

        best = {'th': 0.5, 'recall': 0.0, 'spec': 0.0, 'precision': 0.0}

        # Buscar umbral que cumpla especificidad minima
        for th in candidates:
            yhat = (probs >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            if spec >= min_spec:
                if recall > best['recall'] or (recall == best['recall'] and spec > best['spec']):
                    best.update({
                        'th': float(th),
                        'recall': float(recall),
                        'spec': float(spec),
                        'precision': float(prec_val)
                    })

        # Fallback: maximizar recall si no hay umbral que cumpla
        if best['recall'] == 0.0:
            for th in candidates:
                yhat = (probs >= th).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()

                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                prec_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0

                if recall > best['recall']:
                    best.update({
                        'th': float(th),
                        'recall': float(recall),
                        'spec': float(spec),
                        'precision': float(prec_val)
                    })

        print(f"\nUmbral optimizado: {best['th']:.4f}")
        print(f"  Recall: {best['recall']:.4f}")
        print(f"  Specificity: {best['spec']:.4f}")
        print(f"  Precision: {best['precision']:.4f}")

        return best

    def get_evaluations(self) -> Dict:
        """Retorna todas las evaluaciones."""
        return self.evaluations

    def get_summary(self) -> Dict:
        """
        Retorna un resumen de las metricas principales.

        Returns:
            Dict: Resumen de metricas
        """
        summary = {}
        for name, eval_dict in self.evaluations.items():
            summary[name] = {
                'recall': eval_dict['recall'],
                'precision': eval_dict['precision'],
                'auc': eval_dict['auc'],
                'specificity': eval_dict['specificity'],
                'best_params': eval_dict['best_params']
            }
        return summary
