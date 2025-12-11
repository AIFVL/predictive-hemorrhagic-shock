"""
Pipeline de EDA eliminando filas con valores inválidos en variables binarias.

Este pipeline:
1. Elimina filas con FALLA_CARDIACA > 1 o TABAQUISMO > 1 (17 filas = 1.28%)
2. Ejecuta EDA completo con el dataset limpio
3. Compara estadísticamente con los resultados del EDA con recodificación
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.analysis.eda import ShockEDAAnalyzer
from src.analysis.eda_visualizations import EDAVisualizer
from src.preprocessing.feature_aggregator import FeatureAggregator


class EDA_StrictCleaning_Pipeline:
    """Pipeline de EDA con eliminación estricta de valores inválidos."""

    def __init__(
        self,
        data_path: str,
        output_dir: str = "reports/eda_strict",
        target_col: str = "SHOCK"
    ):
        """
        Inicializa el pipeline.

        Args:
            data_path: Ruta al archivo CSV original
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
        
        self.df_original = None
        self.df_strict = None
        self.removed_rows = None

    def remove_invalid_rows(self, df: pd.DataFrame) -> tuple:
        """
        Elimina filas con valores inválidos en variables binarias.
        
        Args:
            df: DataFrame original
            
        Returns:
            tuple: (df_limpio, df_eliminadas, reporte)
        """
        print("\n" + "="*70)
        print("ELIMINACIÓN ESTRICTA DE VALORES INVÁLIDOS")
        print("="*70)
        
        # Variables binarias que deben tener solo 0 o 1
        binary_vars = ['FALLA_CARDIACA', 'TABAQUISMO']
        
        # Identificar filas inválidas
        mask_invalidos = pd.Series(False, index=df.index)
        
        reporte = []
        
        for var in binary_vars:
            if var in df.columns:
                mask_var = (df[var] != 0) & (df[var] != 1)
                n_invalidos = mask_var.sum()
                
                if n_invalidos > 0:
                    valores = df[mask_var][var].value_counts().to_dict()
                    print(f"\n  Variable '{var}':")
                    print(f"    Valores inválidos: {valores}")
                    print(f"    Filas afectadas: {n_invalidos}")
                    
                    reporte.append({
                        'Variable': var,
                        'N_Invalidos': n_invalidos,
                        'Valores': str(valores)
                    })
                    
                    mask_invalidos |= mask_var
        
        # Filas a eliminar
        df_eliminadas = df[mask_invalidos].copy()
        df_limpio = df[~mask_invalidos].copy()
        
        print("\n" + "-"*70)
        print(f"Total filas originales: {len(df)}")
        print(f"Filas eliminadas: {len(df_eliminadas)} ({len(df_eliminadas)/len(df)*100:.2f}%)")
        print(f"Filas restantes: {len(df_limpio)} ({len(df_limpio)/len(df)*100:.2f}%)")
        
        # Análisis de SHOCK en filas eliminadas
        if len(df_eliminadas) > 0:
            shock_eliminadas = df_eliminadas[self.target_col].value_counts()
            print(f"\nDistribución de SHOCK en filas eliminadas:")
            print(f"  SHOCK=0: {shock_eliminadas.get(0, 0)} ({shock_eliminadas.get(0, 0)/len(df_eliminadas)*100:.1f}%)")
            print(f"  SHOCK=1: {shock_eliminadas.get(1, 0)} ({shock_eliminadas.get(1, 0)/len(df_eliminadas)*100:.1f}%)")
            
            # Comparar con distribución general
            shock_original = df[self.target_col].value_counts(normalize=True)
            shock_limpio = df_limpio[self.target_col].value_counts(normalize=True)
            
            print(f"\nComparación de distribución de SHOCK:")
            print(f"  Original: SHOCK=1 {shock_original.get(1, 0)*100:.2f}%")
            print(f"  Limpio: SHOCK=1 {shock_limpio.get(1, 0)*100:.2f}%")
            print(f"  Diferencia: {abs(shock_original.get(1, 0) - shock_limpio.get(1, 0))*100:.2f} puntos porcentuales")
        
        print("="*70)
        
        return df_limpio, df_eliminadas, reporte

    def statistical_comparison(self, df_recodificado: pd.DataFrame, df_strict: pd.DataFrame):
        """
        Compara estadísticamente los dos datasets.
        
        Args:
            df_recodificado: Dataset con valores recodificados
            df_strict: Dataset con filas eliminadas
        """
        print("\n" + "="*70)
        print("COMPARACIÓN ESTADÍSTICA: RECODIFICACIÓN vs ELIMINACIÓN")
        print("="*70)
        
        comparisons = []
        
        # Comparar medias de variables numéricas
        num_cols = ['EDAD', 'HB_PREQX']
        
        for col in num_cols:
            if col in df_recodificado.columns and col in df_strict.columns:
                mean_recod = df_recodificado[col].mean()
                mean_strict = df_strict[col].mean()
                std_recod = df_recodificado[col].std()
                std_strict = df_strict[col].std()
                
                # Test t para diferencia de medias
                t_stat, p_value = stats.ttest_ind(
                    df_recodificado[col].dropna(),
                    df_strict[col].dropna(),
                    equal_var=False
                )
                
                diff_pct = abs(mean_recod - mean_strict) / mean_recod * 100
                
                comparisons.append({
                    'Variable': col,
                    'Tipo': 'Numérica',
                    'Media_Recodificado': mean_recod,
                    'Media_Strict': mean_strict,
                    'Diferencia_%': diff_pct,
                    'p_value': p_value,
                    'Significativa': 'Sí' if p_value < 0.05 else 'No'
                })
                
                print(f"\n{col}:")
                print(f"  Recodificado: {mean_recod:.2f} ± {std_recod:.2f}")
                print(f"  Strict:       {mean_strict:.2f} ± {std_strict:.2f}")
                print(f"  Diferencia:   {diff_pct:.4f}% (p={p_value:.4f})")
        
        # Comparar proporciones de SHOCK
        prop_recod = df_recodificado[self.target_col].mean()
        prop_strict = df_strict[self.target_col].mean()
        
        # Test chi-cuadrado para diferencia de proporciones
        # Crear dataframe combinado con reset_index
        df_combined = pd.DataFrame({
            'Dataset': ['Recodificado']*len(df_recodificado) + ['Strict']*len(df_strict),
            'SHOCK': pd.concat([df_recodificado[self.target_col], df_strict[self.target_col]], ignore_index=True)
        })
        contingency = pd.crosstab(df_combined['Dataset'], df_combined['SHOCK'])
        
        chi2, p_value_shock, dof, expected = stats.chi2_contingency(contingency)
        diff_pct_shock = abs(prop_recod - prop_strict) / prop_recod * 100
        
        print(f"\n{self.target_col} (proporción de SHOCK=1):")
        print(f"  Recodificado: {prop_recod*100:.2f}%")
        print(f"  Strict:       {prop_strict*100:.2f}%")
        print(f"  Diferencia:   {diff_pct_shock:.4f}% (p={p_value_shock:.4f})")
        
        comparisons.append({
            'Variable': self.target_col,
            'Tipo': 'Target',
            'Proporcion_Recodificado': prop_recod,
            'Proporcion_Strict': prop_strict,
            'Diferencia_%': diff_pct_shock,
            'p_value': p_value_shock,
            'Significativa': 'Sí' if p_value_shock < 0.05 else 'No'
        })
        
        # Comparar FALLA_CARDIACA y TABAQUISMO (ahora sin valores inválidos)
        for var in ['FALLA_CARDIACA', 'TABAQUISMO']:
            if var in df_recodificado.columns and var in df_strict.columns:
                prop_recod_var = df_recodificado[var].mean()
                prop_strict_var = df_strict[var].mean()
                
                # Crear dataframe combinado
                df_combined_var = pd.DataFrame({
                    'Dataset': ['Recodificado']*len(df_recodificado) + ['Strict']*len(df_strict),
                    var: pd.concat([df_recodificado[var], df_strict[var]], ignore_index=True)
                })
                contingency_var = pd.crosstab(df_combined_var['Dataset'], df_combined_var[var])
                
                chi2_var, p_value_var, dof_var, expected_var = stats.chi2_contingency(contingency_var)
                diff_pct_var = abs(prop_recod_var - prop_strict_var) / prop_recod_var * 100 if prop_recod_var > 0 else 0
                
                print(f"\n{var} (proporción de valor=1):")
                print(f"  Recodificado: {prop_recod_var*100:.2f}%")
                print(f"  Strict:       {prop_strict_var*100:.2f}%")
                print(f"  Diferencia:   {diff_pct_var:.4f}% (p={p_value_var:.4f})")
                
                comparisons.append({
                    'Variable': var,
                    'Tipo': 'Binaria',
                    'Proporcion_Recodificado': prop_recod_var,
                    'Proporcion_Strict': prop_strict_var,
                    'Diferencia_%': diff_pct_var,
                    'p_value': p_value_var,
                    'Significativa': 'Sí' if p_value_var < 0.05 else 'No'
                })
        
        # Guardar comparaciones
        comp_df = pd.DataFrame(comparisons)
        comp_path = os.path.join(self.output_dir, 'tables', 'comparison_recodificado_vs_strict.csv')
        os.makedirs(os.path.dirname(comp_path), exist_ok=True)
        comp_df.to_csv(comp_path, index=False)
        
        print(f"\n✓ Tabla comparativa guardada en: {comp_path}")
        print("="*70)
        
        return comp_df

    def run_full_pipeline(self):
        """Ejecuta el pipeline completo."""
        print("\n" + "="*70)
        print("PIPELINE DE EDA CON ELIMINACIÓN ESTRICTA")
        print("="*70)
        print(f"\nDataset: {self.data_path}")
        print(f"Output: {self.output_dir}")
        
        # 1. Cargar datos originales
        print("\n" + "="*70)
        print("PASO 1: CARGA DE DATOS")
        print("="*70)
        
        self.df_original = pd.read_csv(self.data_path)
        print(f"✓ Datos cargados: {self.df_original.shape[0]} filas × {self.df_original.shape[1]} columnas")
        
        # 2. Eliminar filas inválidas
        print("\n" + "="*70)
        print("PASO 2: ELIMINACIÓN DE FILAS INVÁLIDAS")
        print("="*70)
        
        self.df_strict, self.removed_rows, reporte = self.remove_invalid_rows(self.df_original)
        
        # Guardar dataset limpio
        strict_path = os.path.join("data", "processed", "shock_strict_cleaned.csv")
        self.df_strict.to_csv(strict_path, index=False)
        print(f"\n✓ Dataset estrictamente limpio guardado en: {strict_path}")
        
        # 3. Comparación estadística con dataset recodificado
        print("\n" + "="*70)
        print("PASO 3: COMPARACIÓN ESTADÍSTICA")
        print("="*70)
        
        df_recodificado = pd.read_csv("data/processed/shock_cleaned.csv")
        comp_df = self.statistical_comparison(df_recodificado, self.df_strict)
        
        # 4. EDA completo con dataset strict
        print("\n" + "="*70)
        print("PASO 4: ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
        print("="*70)
        
        self.analyzer.df = self.df_strict
        self.analyzer.target_col = self.target_col
        
        # Análisis básico
        basic_info = self.analyzer.basic_analysis()
        
        # Identificar tipos
        num_cols, cat_cols, bin_cols = self.analyzer.identify_variable_types()
        
        # Visualizaciones
        if self.analyzer.target_col:
            self.visualizer.plot_target_distribution(self.df_strict, self.analyzer.target_col)
            self.visualizer.plot_class_imbalance(self.df_strict, self.analyzer.target_col)
        
        # Análisis univariado
        cat_freq = self.analyzer.univariate_analysis_categorical()
        num_stats = self.analyzer.univariate_analysis_numerical()
        
        self.visualizer.plot_missing_values(self.df_strict)
        if num_cols:
            self.visualizer.plot_numerical_distributions(
                self.df_strict, num_cols, self.analyzer.target_col
            )
        
        # Análisis bivariado
        cat_vs_target = self.analyzer.bivariate_analysis_categorical_vs_target()
        num_vs_target = self.analyzer.bivariate_analysis_numerical_vs_target()
        
        # Visualizaciones bivariadas
        all_categorical = cat_cols + bin_cols
        if all_categorical and self.analyzer.target_col:
            self.visualizer.plot_categorical_by_target(
                self.df_strict, all_categorical, self.analyzer.target_col, top_n=15
            )
        
        if num_cols and self.analyzer.target_col:
            self.visualizer.plot_numerical_boxplots(
                self.df_strict, num_cols, self.analyzer.target_col
            )
        
        if not cat_vs_target.empty:
            self.visualizer.plot_top_associations(
                cat_vs_target, 'Cramers_V', top_n=20,
                title="Top Asociaciones Categóricas (Cramer's V) - Dataset Strict"
            )
        
        if not num_vs_target.empty:
            num_copy = num_vs_target.copy()
            num_copy['abs_Cohens_d'] = num_copy['Cohens_d'].abs()
            self.visualizer.plot_top_associations(
                num_copy, 'abs_Cohens_d', top_n=20,
                title="Top Diferencias Numéricas (Cohen's d) - Dataset Strict"
            )
        
        # Análisis multivariado
        corr_num, cramers_mat = self.analyzer.correlation_analysis()
        
        if not corr_num.empty and len(corr_num) > 1:
            self.visualizer.plot_correlation_heatmap(
                corr_num, title="Correlación Spearman - Dataset Strict"
            )
        
        if not cramers_mat.empty and len(cramers_mat) > 1:
            self.visualizer.plot_cramers_v_heatmap(cramers_mat, top_n=25)
        
        # PCA
        try:
            pca_components, pca_obj = self.analyzer.pca_analysis(n_components=2)
            self.visualizer.plot_pca_2d(
                pca_components,
                self.df_strict[self.analyzer.target_col],
                pca_obj.explained_variance_ratio_
            )
        except Exception as e:
            print(f"Error en PCA: {e}")
        
        # 5. Generar reporte
        report_path = self._generate_report(comp_df, cat_vs_target, num_vs_target)
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETADO")
        print("="*70)
        print(f"\n✓ Análisis completado exitosamente")
        print(f"\nArchivos generados:")
        print(f"  - Dataset strict: {strict_path}")
        print(f"  - Reporte: {report_path}")
        print(f"  - Tablas: {os.path.join(self.output_dir, 'tables/')}")
        print(f"  - Figuras: {os.path.join(self.output_dir, 'figures/')}")
        print("="*70)

    def _generate_report(self, comp_df, cat_vs_target, num_vs_target):
        """Genera reporte resumen."""
        report_path = os.path.join(self.output_dir, 'eda_strict_summary_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Reporte EDA - Dataset con Eliminación Estricta de Valores Inválidos\n\n")
            f.write("## Predicción de Shock Hemorrágico\n\n")
            f.write("---\n\n")
            
            f.write("## 1. Estrategia de Limpieza\n\n")
            f.write("### Eliminación Estricta\n\n")
            f.write(f"- **Filas eliminadas**: {len(self.removed_rows)} ({len(self.removed_rows)/len(self.df_original)*100:.2f}%)\n")
            f.write(f"- **Filas conservadas**: {len(self.df_strict)} ({len(self.df_strict)/len(self.df_original)*100:.2f}%)\n\n")
            
            f.write("### Razón de Eliminación\n\n")
            f.write("Variables binarias con valores inválidos (>1):\n")
            f.write("- `FALLA_CARDIACA`: valores 2 encontrados\n")
            f.write("- `TABAQUISMO`: valores 2 encontrados\n\n")
            
            f.write("## 2. Comparación Estadística\n\n")
            f.write("### Dataset Recodificado vs Dataset Strict\n\n")
            
            if not comp_df.empty:
                f.write("| Variable | Diferencia % | p-value | Significativa |\n")
                f.write("|----------|-------------|---------|---------------|\n")
                for _, row in comp_df.iterrows():
                    f.write(f"| {row['Variable']} | {row['Diferencia_%']:.4f}% | {row['p_value']:.4f} | {row['Significativa']} |\n")
            
            f.write("\n### Interpretación\n\n")
            f.write("Ver tabla `comparison_recodificado_vs_strict.csv` para análisis detallado.\n\n")
            
            # Variables significativas
            info = self.analyzer.results['basic_info']
            f.write("## 3. Información del Dataset Strict\n\n")
            f.write(f"- **Registros**: {info['n_rows']}\n")
            f.write(f"- **Variables**: {info['n_columns']}\n\n")
            
            if info['target_distribution']:
                f.write("### Distribución del Target\n\n")
                for k, v in info['target_distribution'].items():
                    pct = info['target_percentage'][k] * 100
                    f.write(f"- Clase {k}: {v} registros ({pct:.2f}%)\n")
            
            f.write("\n---\n\n")
            f.write("*Reporte generado por EDA_StrictCleaning_Pipeline*\n")
        
        return report_path


def main():
    """Función principal."""
    DATA_PATH = "data/processed/shock.csv"
    OUTPUT_DIR = "reports/eda_strict"
    
    pipeline = EDA_StrictCleaning_Pipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR
    )
    
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
