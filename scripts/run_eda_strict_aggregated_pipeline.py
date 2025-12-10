"""
Pipeline de EDA con variables agregadas sobre dataset con eliminación estricta.
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.analysis.eda import ShockEDAAnalyzer
from src.analysis.eda_visualizations import EDAVisualizer
from src.preprocessing.feature_aggregator import FeatureAggregator


class EDA_Strict_Aggregated_Pipeline:
    """Pipeline de EDA con agregación sobre dataset strict."""

    def __init__(
        self,
        data_path: str,
        output_dir: str = "reports/eda_strict_aggregated",
        target_col: str = "SHOCK"
    ):
        """Inicializa el pipeline."""
        self.data_path = data_path
        self.output_dir = output_dir
        self.target_col = target_col
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.aggregator = FeatureAggregator(verbose=True)
        self.analyzer = ShockEDAAnalyzer(output_dir)
        self.visualizer = EDAVisualizer(os.path.join(output_dir, "figures"))
        
        self.df = None
        self.df_aggregated = None

    def run_full_pipeline(self):
        """Ejecuta el pipeline completo."""
        print("\n" + "="*70)
        print("PIPELINE EDA AGREGADO - DATASET STRICT")
        print("="*70)
        
        # 1. Cargar datos
        print("\nPASO 1: CARGA DE DATOS STRICT")
        print("-"*70)
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Datos cargados: {self.df.shape[0]} filas × {self.df.shape[1]} columnas")
        
        # 2. Agregar features
        print("\nPASO 2: AGREGACIÓN DE FEATURES")
        print("-"*70)
        self.df_aggregated = self.aggregator.fit_transform(self.df)
        
        aggregated_path = os.path.join("data", "processed", "shock_strict_aggregated.csv")
        self.df_aggregated.to_csv(aggregated_path, index=False)
        print(f"\n✓ Dataset guardado: {aggregated_path}")
        
        # 3. EDA completo
        print("\nPASO 3: ANÁLISIS EXPLORATORIO")
        print("-"*70)
        
        self.analyzer.df = self.df_aggregated
        self.analyzer.target_col = self.target_col
        
        basic_info = self.analyzer.basic_analysis()
        num_cols, cat_cols, bin_cols = self.analyzer.identify_variable_types()
        
        if self.analyzer.target_col:
            self.visualizer.plot_target_distribution(self.df_aggregated, self.analyzer.target_col)
            self.visualizer.plot_class_imbalance(self.df_aggregated, self.analyzer.target_col)
        
        cat_freq = self.analyzer.univariate_analysis_categorical()
        num_stats = self.analyzer.univariate_analysis_numerical()
        
        self.visualizer.plot_missing_values(self.df_aggregated)
        if num_cols:
            self.visualizer.plot_numerical_distributions(
                self.df_aggregated, num_cols, self.analyzer.target_col
            )
        
        cat_vs_target = self.analyzer.bivariate_analysis_categorical_vs_target()
        num_vs_target = self.analyzer.bivariate_analysis_numerical_vs_target()
        
        all_categorical = cat_cols + bin_cols
        if all_categorical and self.analyzer.target_col:
            self.visualizer.plot_categorical_by_target(
                self.df_aggregated, all_categorical, self.analyzer.target_col, top_n=15
            )
        
        if num_cols and self.analyzer.target_col:
            self.visualizer.plot_numerical_boxplots(
                self.df_aggregated, num_cols, self.analyzer.target_col
            )
        
        if not cat_vs_target.empty:
            self.visualizer.plot_top_associations(
                cat_vs_target, 'Cramers_V', top_n=20,
                title="Top Asociaciones - Dataset Strict Agregado"
            )
        
        if not num_vs_target.empty:
            num_copy = num_vs_target.copy()
            num_copy['abs_Cohens_d'] = num_copy['Cohens_d'].abs()
            self.visualizer.plot_top_associations(
                num_copy, 'abs_Cohens_d', top_n=20,
                title="Top Diferencias - Dataset Strict Agregado"
            )
        
        corr_num, cramers_mat = self.analyzer.correlation_analysis()
        
        if not corr_num.empty and len(corr_num) > 1:
            self.visualizer.plot_correlation_heatmap(
                corr_num,
                title="Correlación Spearman - Strict Agregado",
                aggregated_vars=self.aggregator.aggregated_features
            )
        
        if not cramers_mat.empty and len(cramers_mat) > 1:
            self.visualizer.plot_cramers_v_heatmap(
                cramers_mat, top_n=25, aggregated_vars=self.aggregator.aggregated_features
            )
        
        try:
            pca_components, pca_obj = self.analyzer.pca_analysis(n_components=2)
            self.visualizer.plot_pca_2d(
                pca_components,
                self.df_aggregated[self.analyzer.target_col],
                pca_obj.explained_variance_ratio_
            )
        except Exception as e:
            print(f"Error en PCA: {e}")
        
        report_path = self.analyzer.generate_summary_report()
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETADO")
        print("="*70)
        print(f"\nArchivos en: {self.output_dir}")
        print("="*70)


def main():
    """Función principal."""
    DATA_PATH = "data/processed/shock_strict_cleaned.csv"
    OUTPUT_DIR = "reports/eda_strict_aggregated"
    
    pipeline = EDA_Strict_Aggregated_Pipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR
    )
    
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
