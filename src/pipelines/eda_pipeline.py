"""
Pipeline completo de Análisis Exploratorio de Datos (EDA) para shock hemorrágico.
"""

import os
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.eda import ShockEDAAnalyzer
from src.analysis.eda_visualizations import EDAVisualizer


class EDA_Pipeline:
    """Pipeline completo de EDA."""

    def __init__(
        self,
        data_path: str,
        output_dir: str = "reports/eda",
        target_col: str = None
    ):
        """
        Inicializa el pipeline de EDA.

        Args:
            data_path: Ruta al archivo CSV de datos
            output_dir: Directorio de salida para reportes
            target_col: Nombre de la columna objetivo
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.target_col = target_col
        
        # Crear directorios
        os.makedirs(output_dir, exist_ok=True)
        
        # Inicializar componentes
        self.analyzer = ShockEDAAnalyzer(output_dir)
        self.visualizer = EDAVisualizer(os.path.join(output_dir, "figures"))
        
        self.df = None

    def run_full_eda(self):
        """Ejecuta el pipeline completo de EDA."""
        print("\n" + "="*70)
        print("PIPELINE DE ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
        print("="*70)
        print(f"\nDataset: {self.data_path}")
        print(f"Output: {self.output_dir}")
        print()

        # 1. Cargar datos
        self.df = self.analyzer.load_data(self.data_path, self.target_col)
        
        # 2. Análisis básico
        basic_info = self.analyzer.basic_analysis()
        
        # 3. Identificar tipos de variables
        num_cols, cat_cols, bin_cols = self.analyzer.identify_variable_types()
        
        # 4. Visualizar distribución del target
        if self.analyzer.target_col:
            print("\n" + "="*70)
            print("VISUALIZACIONES - DISTRIBUCIÓN DEL TARGET")
            print("="*70)
            self.visualizer.plot_target_distribution(self.df, self.analyzer.target_col)
            self.visualizer.plot_class_imbalance(self.df, self.analyzer.target_col)
        
        # 5. Análisis univariado
        print("\n" + "="*70)
        print("ANÁLISIS UNIVARIADO")
        print("="*70)
        
        # Categóricas
        cat_freq = self.analyzer.univariate_analysis_categorical()
        
        # Numéricas
        num_stats = self.analyzer.univariate_analysis_numerical()
        
        # 6. Visualizaciones univariadas
        print("\n" + "="*70)
        print("VISUALIZACIONES UNIVARIADAS")
        print("="*70)
        
        # Valores faltantes
        self.visualizer.plot_missing_values(self.df)
        
        # Distribuciones numéricas
        if num_cols:
            self.visualizer.plot_numerical_distributions(
                self.df, num_cols, self.analyzer.target_col
            )
        
        # 7. Análisis bivariado
        print("\n" + "="*70)
        print("ANÁLISIS BIVARIADO")
        print("="*70)
        
        # Categóricas vs target
        cat_vs_target = self.analyzer.bivariate_analysis_categorical_vs_target()
        
        # Numéricas vs target
        num_vs_target = self.analyzer.bivariate_analysis_numerical_vs_target()
        
        # 8. Visualizaciones bivariadas
        print("\n" + "="*70)
        print("VISUALIZACIONES BIVARIADAS")
        print("="*70)
        
        # Categóricas por target
        all_categorical = cat_cols + bin_cols
        if all_categorical and self.analyzer.target_col:
            self.visualizer.plot_categorical_by_target(
                self.df, all_categorical, self.analyzer.target_col, top_n=10
            )
        
        # Boxplots numéricas
        if num_cols and self.analyzer.target_col:
            self.visualizer.plot_numerical_boxplots(
                self.df, num_cols, self.analyzer.target_col
            )
        
        # Top asociaciones categóricas
        if not cat_vs_target.empty:
            self.visualizer.plot_top_associations(
                cat_vs_target, 'Cramers_V', top_n=15,
                title="Top Asociaciones Categóricas (Cramer's V)"
            )
        
        # Top diferencias numéricas
        if not num_vs_target.empty:
            # Usar valor absoluto de Cohen's d para ranking
            num_copy = num_vs_target.copy()
            num_copy['abs_Cohens_d'] = num_copy['Cohens_d'].abs()
            self.visualizer.plot_top_associations(
                num_copy, 'abs_Cohens_d', top_n=15,
                title="Top Diferencias Numéricas (|Cohen's d|)"
            )
        
        # 9. Análisis multivariado
        print("\n" + "="*70)
        print("ANÁLISIS MULTIVARIADO")
        print("="*70)
        
        # Correlaciones
        corr_num, cramers_mat = self.analyzer.correlation_analysis()
        
        # Visualizar correlaciones
        if not corr_num.empty and len(corr_num) > 1:
            self.visualizer.plot_correlation_heatmap(
                corr_num, title="Correlación Spearman (Variables Numéricas)"
            )
        
        if not cramers_mat.empty and len(cramers_mat) > 1:
            self.visualizer.plot_cramers_v_heatmap(cramers_mat, top_n=20)
        
        # PCA
        try:
            pca_components, pca_obj = self.analyzer.pca_analysis(n_components=2)
            
            # Visualizar PCA
            self.visualizer.plot_pca_2d(
                pca_components, 
                self.df[self.analyzer.target_col],
                pca_obj.explained_variance_ratio_
            )
        except Exception as e:
            print(f"Error en PCA: {e}")
        
        # 10. Feature engineering exploratorio
        print("\n" + "="*70)
        print("FEATURE ENGINEERING EXPLORATORIO")
        print("="*70)
        
        fe_results = self.analyzer.feature_engineering_analysis()
        
        # 11. Generar reporte resumen
        print("\n" + "="*70)
        print("GENERACIÓN DE REPORTE")
        print("="*70)
        
        report_path = self.analyzer.generate_summary_report()
        
        # 12. Resumen final
        print("\n" + "="*70)
        print("PIPELINE DE EDA COMPLETADO")
        print("="*70)
        print(f"\nAnalisis completado exitosamente")
        print(f"\nArchivos generados:")
        print(f"  - Reporte resumen: {report_path}")
        print(f"  - Tablas CSV: {os.path.join(self.output_dir, 'tables/')}")
        print(f"  - Visualizaciones: {os.path.join(self.output_dir, 'figures/')}")
        
        # Listar archivos generados
        tables_dir = os.path.join(self.output_dir, 'tables')
        figures_dir = os.path.join(self.output_dir, 'figures')
        
        if os.path.exists(tables_dir):
            tables = [f for f in os.listdir(tables_dir) if f.endswith('.csv')]
            print(f"\nTablas generadas ({len(tables)}):")
            for table in sorted(tables):
                print(f"  - {table}")
        
        if os.path.exists(figures_dir):
            figures = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
            print(f"\nVisualizaciones generadas ({len(figures)}):")
            for figure in sorted(figures):
                print(f"  - {figure}")
        
        print("\n" + "="*70)

    def get_results(self):
        """
        Obtiene los resultados del análisis.

        Returns:
            dict: Diccionario con todos los resultados
        """
        return self.analyzer.results


def main():
    """Función principal para ejecutar el pipeline."""
    # Configuración
    DATA_PATH = "data/processed/shock.csv"
    OUTPUT_DIR = "reports/eda"
    
    # Crear y ejecutar pipeline
    pipeline = EDA_Pipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR
    )
    
    pipeline.run_full_eda()


if __name__ == "__main__":
    main()
