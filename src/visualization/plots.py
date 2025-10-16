"""
Modulo de visualizaciones para analisis de shock hemorragico.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from typing import List, Optional


class ShockVisualizer:
    """Clase para crear visualizaciones del analisis."""

    def __init__(self, output_dir: str = "shock_output"):
        """
        Inicializa el visualizador.

        Args:
            output_dir: Directorio de salida para graficos
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_numeric_distributions(
        self,
        df: pd.DataFrame,
        num_cols: List[str],
        target_col: str
    ) -> str:
        """
        Grafica distribuciones de variables numericas por clase.

        Args:
            df: DataFrame con los datos
            num_cols: Lista de columnas numericas
            target_col: Nombre de la columna objetivo

        Returns:
            str: Ruta del archivo guardado
        """
        if not num_cols:
            print("No hay columnas numericas para graficar")
            return None

        n_cols = len(num_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

        if n_cols == 1:
            axes = [axes]

        for i, col in enumerate(num_cols):
            sns.histplot(
                data=df,
                x=col,
                hue=target_col,
                ax=axes[i],
                alpha=0.7,
                kde=True
            )
            axes[i].set_title(f'Distribucion de {col} por {target_col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frecuencia')

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'numeric_distributions.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Grafico de distribuciones guardado en: {filepath}")
        return filepath

    def plot_roc_curve(
        self,
        y_test,
        proba_test: np.ndarray,
        model_name: str = "Model"
    ) -> str:
        """
        Grafica curva ROC.

        Args:
            y_test: Valores verdaderos
            proba_test: Probabilidades predichas
            model_name: Nombre del modelo

        Returns:
            str: Ruta del archivo guardado
        """
        fpr, tpr, _ = roc_curve(y_test, proba_test)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr,
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {roc_auc:.2f})'
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title(f'Receiver Operating Characteristic - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        filepath = os.path.join(self.output_dir, 'roc_curve.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Curva ROC guardada en: {filepath}")
        return filepath

    def plot_precision_recall_curve(
        self,
        y_test,
        proba_test: np.ndarray,
        model_name: str = "Model"
    ) -> str:
        """
        Grafica curva Precision-Recall.

        Args:
            y_test: Valores verdaderos
            proba_test: Probabilidades predichas
            model_name: Nombre del modelo

        Returns:
            str: Ruta del archivo guardado
        """
        precision, recall, _ = precision_recall_curve(y_test, proba_test)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.grid(True, alpha=0.3)

        filepath = os.path.join(self.output_dir, 'precision_recall_curve.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Curva Precision-Recall guardada en: {filepath}")
        return filepath

    def plot_confusion_matrix(
        self,
        confusion: tuple,
        model_name: str = "Model"
    ) -> str:
        """
        Grafica matriz de confusion.

        Args:
            confusion: Tupla (TN, FP, FN, TP)
            model_name: Nombre del modelo

        Returns:
            str: Ruta del archivo guardado
        """
        tn, fp, fn, tp = confusion
        cm = np.array([[tn, fp], [fn, tp]])

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negativo', 'Positivo'],
            yticklabels=['Negativo', 'Positivo']
        )
        plt.ylabel('Valor Real')
        plt.xlabel('Prediccion')
        plt.title(f'Matriz de Confusion - {model_name}')

        filepath = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Matriz de confusion guardada en: {filepath}")
        return filepath

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20
    ) -> str:
        """
        Grafica importancia de features.

        Args:
            importance_df: DataFrame con columnas 'feature' e 'importance_mean'
            top_n: Numero de features top a mostrar

        Returns:
            str: Ruta del archivo guardado
        """
        top_features = importance_df.head(top_n).sort_values('importance_mean')

        plt.figure(figsize=(10, 8))
        plt.barh(top_features['feature'], top_features['importance_mean'])
        plt.xlabel('Importancia Promedio')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Features mas Importantes')
        plt.grid(True, alpha=0.3, axis='x')

        filepath = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Grafico de importancia guardado en: {filepath}")
        return filepath

    def plot_calibration_curve(
        self,
        y_test,
        proba_test: np.ndarray,
        proba_calibrated: Optional[np.ndarray] = None,
        model_name: str = "Model"
    ) -> str:
        """
        Grafica curva de calibracion.

        Args:
            y_test: Valores verdaderos
            proba_test: Probabilidades sin calibrar
            proba_calibrated: Probabilidades calibradas (opcional)
            model_name: Nombre del modelo

        Returns:
            str: Ruta del archivo guardado
        """
        from sklearn.calibration import calibration_curve

        plt.figure(figsize=(8, 6))

        # Curva para probabilidades sin calibrar
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, proba_test, n_bins=10
        )
        plt.plot(
            mean_predicted_value,
            fraction_of_positives,
            's-',
            label='Sin calibrar'
        )

        # Curva para probabilidades calibradas si existen
        if proba_calibrated is not None:
            fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
                y_test, proba_calibrated, n_bins=10
            )
            plt.plot(
                mean_predicted_value_cal,
                fraction_of_positives_cal,
                's-',
                label='Calibrado'
            )

        plt.plot([0, 1], [0, 1], 'k--', label='Perfectamente calibrado')
        plt.xlabel('Probabilidad Predicha Promedio')
        plt.ylabel('Fraccion de Positivos')
        plt.title(f'Curva de Calibracion - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        filepath = os.path.join(self.output_dir, 'calibration_curve.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Curva de calibracion guardada en: {filepath}")
        return filepath

    def plot_threshold_analysis(
        self,
        y_test,
        proba_test: np.ndarray
    ) -> str:
        """
        Analiza metricas a diferentes umbrales.

        Args:
            y_test: Valores verdaderos
            proba_test: Probabilidades predichas

        Returns:
            str: Ruta del archivo guardado
        """
        from sklearn.metrics import precision_score, recall_score

        thresholds = np.linspace(0, 1, 100)
        precisions = []
        recalls = []
        specificities = []

        for th in thresholds:
            y_pred = (proba_test >= th).astype(int)
            precisions.append(precision_score(y_test, y_pred, zero_division=0))
            recalls.append(recall_score(y_test, y_pred, zero_division=0))

            tn = ((y_test == 0) & (y_pred == 0)).sum()
            fp = ((y_test == 0) & (y_pred == 1)).sum()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(spec)

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, label='Recall', linewidth=2)
        plt.plot(thresholds, specificities, label='Specificity', linewidth=2)
        plt.xlabel('Umbral')
        plt.ylabel('Metrica')
        plt.title('Metricas vs Umbral de Decision')
        plt.legend()
        plt.grid(True, alpha=0.3)

        filepath = os.path.join(self.output_dir, 'threshold_analysis.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Analisis de umbrales guardado en: {filepath}")
        return filepath
