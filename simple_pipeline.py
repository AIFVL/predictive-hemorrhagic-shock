"""
Pipeline simplificado para predicción de shock hemorrágico.

Este pipeline realiza:
1. Carga de datos
2. Preprocesamiento
3. Entrenamiento de modelos
4. Evaluación
5. Registro en MLflow
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns


class SimpleShockPipeline:
    """Pipeline simplificado para predicción de shock hemorrágico."""
    
    def __init__(self, data_path: str, output_dir: str = "output"):
        """
        Inicializa el pipeline.
        
        Args:
            data_path: Ruta al archivo de datos
            output_dir: Directorio para guardar outputs
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Componentes del pipeline
        self.scaler = StandardScaler()
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42)
        }
        self.trained_models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self) -> pd.DataFrame:
        """Carga los datos desde el archivo especificado."""
        print(f"Cargando datos desde: {self.data_path}")
        
        if str(self.data_path).endswith('.csv'):
            df = pd.read_csv(self.data_path)
        elif str(self.data_path).endswith('.parquet'):
            df = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Formato de archivo no soportado: {self.data_path}")
            
        print(f"Datos cargados: {df.shape[0]} filas x {df.shape[1]} columnas")
        return df
    
    def preprocess_data(self, df: pd.DataFrame, target_col: str = 'SHOCK'):
        """Preprocesa los datos y divide en entrenamiento y prueba."""
        print("Preprocesando datos...")
        
        # Separar features y target
        if target_col not in df.columns:
            raise ValueError(f"Columna '{target_col}' no encontrada en los datos")
            
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Convertir columnas no numéricas a dummy variables
        X = pd.get_dummies(X, drop_first=True)
        
        # Dividir datos
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Escalar features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Datos divididos: {self.X_train.shape[0]} entrenamiento, {self.X_test.shape[0]} prueba")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Entrena los modelos definidos."""
        print("Entrenando modelos...")
        
        for name, model in self.models.items():
            print(f"  - Entrenando {name}...")
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model
    
    def evaluate_models(self):
        """Evalúa los modelos entrenados."""
        print("Evaluando modelos...")
        
        results = {}
        
        for name, model in self.trained_models.items():
            # Predicciones
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Métricas
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba) if len(np.unique(self.y_test)) > 1 else 0.5
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            }
            
            print(f"  - {name}: Accuracy={accuracy:.3f}, F1={f1:.3f}, ROC-AUC={roc_auc:.3f}")
        
        return results
    
    def plot_results(self, results: Dict[str, Dict[str, float]]):
        """Genera gráficas de resultados."""
        print("Generando gráficas de resultados...")
        
        # Gráfica de métricas por modelo
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in results.keys()]
            models = list(results.keys())
            
            axes[i].bar(models, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()} por modelo')
            axes[i].set_ylabel(metric)
            
            # Añadir valores encima de las barras
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Eliminar el sexto subplot no usado
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plot_path = self.output_dir / "model_comparison.png"
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Gráfica de comparación guardada en: {plot_path}")
        
        # Gráfica de importancia de features para Random Forest si existe
        if 'random_forest' in self.trained_models:
            rf_model = self.trained_models['random_forest']
            feature_names = [f'feature_{i}' for i in range(rf_model.n_features_in_)]
            
            plt.figure(figsize=(10, 6))
            indices = np.argsort(rf_model.feature_importances_)[::-1][:10]
            plt.bar(range(len(indices)), rf_model.feature_importances_[indices])
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
            plt.title('Top 10 Importancia de Features - Random Forest')
            plt.ylabel('Importancia')
            
            plot_path = self.output_dir / "feature_importance_rf.png"
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            
            print(f"Gráfica de importancia de features guardada en: {plot_path}")
    
    def log_to_mlflow(self, results: Dict[str, Dict[str, float]]):
        """Registra resultados en MLflow."""
        print("Registrando experimentos en MLflow...")
        
        # Crear experimento
        experiment_name = "simple_shock_prediction"
        mlflow.set_experiment(experiment_name)
        
        for model_name, metrics in results.items():
            with mlflow.start_run(run_name=f"{model_name}_run"):
                # Registrar métricas
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value)
                
                # Registrar parámetros
                model_instance = self.models[model_name]
                mlflow.log_param('model_type', model_name)
                if model_name == 'random_forest':
                    mlflow.log_param('n_estimators', model_instance.n_estimators)
                    mlflow.log_param('max_depth', getattr(model_instance, 'max_depth', 'None'))
                elif model_name == 'logistic_regression':
                    mlflow.log_param('C', getattr(model_instance, 'C', 'None'))
                    
                print(f"  - {model_name} registrado en MLflow")
    
    def run_pipeline(self, target_col: str = 'SHOCK'):
        """Ejecuta el pipeline completo."""
        print("="*60)
        print("INICIANDO PIPELINE SIMPLIFICADO DE SHOCK HEMORRÁGICO")
        print("="*60)

        # 1. Cargar datos
        df = self.load_data()

        # 2. Preprocesar datos
        self.preprocess_data(df, target_col)

        # 3. Entrenar modelos
        self.train_models()

        # 4. Evaluar modelos
        results = self.evaluate_models()

        # 5. Generar gráficas
        self.plot_results(results)

        # 6. Registrar en MLflow
        self.log_to_mlflow(results)

        print("="*60)
        print("PIPELINE COMPLETADO")
        print(f"Resultados guardados en: {self.output_dir}")
        print("="*60)

        return results


def main():
    """Función principal."""
    import sys
    import argparse
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    parser = argparse.ArgumentParser(description='Pipeline simplificado de shock hemorrágico')
    parser.add_argument('--data', type=str, required=True, help='Ruta al archivo de datos')
    parser.add_argument('--output', type=str, default='output', help='Directorio de salida')
    parser.add_argument('--target', type=str, default='SHOCK', help='Nombre de la columna objetivo')

    args = parser.parse_args()

    # Crear pipeline y ejecutar
    pipeline = SimpleShockPipeline(data_path=args.data, output_dir=args.output)
    results = pipeline.run_pipeline(target_col=args.target)

    print("\nResumen de resultados:")
    for model, metrics in results.items():
        print(f"{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")


if __name__ == "__main__":
    main()