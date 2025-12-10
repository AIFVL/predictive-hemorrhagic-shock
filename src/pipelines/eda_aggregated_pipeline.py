"""
Pipeline de EDA para dataset con variables agregadas.

Este pipeline ejecuta el análisis exploratorio completo sobre el dataset
que incluye tanto las variables originales como las variables agregadas
documentadas en doc/3.5_feature_aggregation.md
"""

import os
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.eda import ShockEDAAnalyzer
from src.analysis.eda_visualizations import EDAVisualizer
from src.preprocessing.feature_aggregator import FeatureAggregator
import pandas as pd


class EDA_Aggregated_Pipeline:
    """Pipeline de EDA con agregación de features."""

    def __init__(
        self,
        data_path: str,
        output_dir: str = "reports/eda_aggregated",
        target_col: str = "SHOCK"
    ):
        """
        Inicializa el pipeline de EDA con agregación.

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
        self.aggregator = FeatureAggregator(verbose=True)
        self.analyzer = ShockEDAAnalyzer(output_dir)
        self.visualizer = EDAVisualizer(os.path.join(output_dir, "figures"))
        
        self.df = None
        self.df_aggregated = None

    def run_full_pipeline(self):
        """Ejecuta el pipeline completo: agregación + EDA."""
        print("\n" + "="*70)
        print("PIPELINE DE EDA CON AGREGACIÓN DE FEATURES")
        print("="*70)
        print(f"\nDataset: {self.data_path}")
        print(f"Output: {self.output_dir}")
        print()

        # PASO 1: Cargar datos originales
        print("\n" + "="*70)
        print("PASO 1: CARGA DE DATOS ORIGINALES")
        print("="*70)
        
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Datos cargados: {self.df.shape[0]} filas × {self.df.shape[1]} columnas")
        
        # PASO 2: Aplicar agregación de features
        print("\n" + "="*70)
        print("PASO 2: AGREGACIÓN DE FEATURES")
        print("="*70)
        
        self.df_aggregated = self.aggregator.fit_transform(self.df)
        
        # Guardar dataset con features agregadas
        aggregated_path = os.path.join("data", "processed", "shock_aggregated.csv")
        os.makedirs(os.path.dirname(aggregated_path), exist_ok=True)
        self.df_aggregated.to_csv(aggregated_path, index=False)
        print(f"\n✓ Dataset con features agregadas guardado en: {aggregated_path}")
        
        # Guardar resumen de features
        summary = self.aggregator.get_feature_summary()
        summary_path = os.path.join(self.output_dir, "tables", "feature_aggregation_summary.csv")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"✓ Resumen de features guardado en: {summary_path}")
        
        # PASO 3: Análisis exploratorio completo
        print("\n" + "="*70)
        print("PASO 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
        print("="*70)
        
        # Cargar datos en el analizador
        self.analyzer.df = self.df_aggregated
        self.analyzer.target_col = self.target_col
        
        # 3.1 Análisis básico
        basic_info = self.analyzer.basic_analysis()
        
        # 3.2 Identificar tipos de variables
        num_cols, cat_cols, bin_cols = self.analyzer.identify_variable_types()
        
        # 3.3 Visualizar distribución del target
        if self.analyzer.target_col:
            print("\n" + "="*70)
            print("VISUALIZACIONES - DISTRIBUCIÓN DEL TARGET")
            print("="*70)
            self.visualizer.plot_target_distribution(
                self.df_aggregated, self.analyzer.target_col
            )
            self.visualizer.plot_class_imbalance(
                self.df_aggregated, self.analyzer.target_col
            )
        
        # 3.4 Análisis univariado
        print("\n" + "="*70)
        print("ANÁLISIS UNIVARIADO")
        print("="*70)
        
        cat_freq = self.analyzer.univariate_analysis_categorical()
        num_stats = self.analyzer.univariate_analysis_numerical()
        
        # 3.5 Visualizaciones univariadas
        print("\n" + "="*70)
        print("VISUALIZACIONES UNIVARIADAS")
        print("="*70)
        
        self.visualizer.plot_missing_values(self.df_aggregated)
        
        if num_cols:
            self.visualizer.plot_numerical_distributions(
                self.df_aggregated, num_cols, self.analyzer.target_col
            )
        
        # 3.6 Análisis bivariado (CRÍTICO: evaluar poder predictivo)
        print("\n" + "="*70)
        print("ANÁLISIS BIVARIADO - EVALUACIÓN DE PODER PREDICTIVO")
        print("="*70)
        
        cat_vs_target = self.analyzer.bivariate_analysis_categorical_vs_target()
        num_vs_target = self.analyzer.bivariate_analysis_numerical_vs_target()
        
        # 3.7 Análisis específico de variables agregadas
        print("\n" + "="*70)
        print("ANÁLISIS COMPARATIVO: VARIABLES AGREGADAS vs ORIGINALES")
        print("="*70)
        
        self._compare_aggregated_vs_original(cat_vs_target, num_vs_target)
        
        # 3.8 Visualizaciones bivariadas
        print("\n" + "="*70)
        print("VISUALIZACIONES BIVARIADAS")
        print("="*70)
        
        all_categorical = cat_cols + bin_cols
        # Asegurar que las variables agregadas (si existen) estén incluidas en las categóricas/binaries
        for v in self.aggregator.aggregated_features:
            if v in self.df_aggregated.columns and v not in all_categorical:
                all_categorical.append(v)

        if all_categorical and self.analyzer.target_col:
            self.visualizer.plot_categorical_by_target(
                self.df_aggregated, all_categorical, self.analyzer.target_col, top_n=15
            )
        
        if num_cols and self.analyzer.target_col:
            self.visualizer.plot_numerical_boxplots(
                self.df_aggregated, num_cols, self.analyzer.target_col
            )
        
        # Top asociaciones
        if not cat_vs_target.empty:
            self.visualizer.plot_top_associations(
                cat_vs_target, 'Cramers_V', top_n=20,
                title="Top Asociaciones Categóricas (Cramer's V) - Dataset Agregado"
            )
        
        if not num_vs_target.empty:
            num_copy = num_vs_target.copy()
            num_copy['abs_Cohens_d'] = num_copy['Cohens_d'].abs()
            self.visualizer.plot_top_associations(
                num_copy, 'abs_Cohens_d', top_n=20,
                title="Top Diferencias Numéricas (|Cohen's d|) - Dataset Agregado"
            )
        
        # 3.9 Análisis multivariado
        print("\n" + "="*70)
        print("ANÁLISIS MULTIVARIADO")
        print("="*70)
        
        corr_num, cramers_mat = self.analyzer.correlation_analysis()
        
        if not corr_num.empty and len(corr_num) > 1:
            self.visualizer.plot_correlation_heatmap(
                corr_num,
                title="Correlación Spearman - Variables Numéricas (Dataset Agregado)",
                aggregated_vars=self.aggregator.aggregated_features
            )
        
        if not cramers_mat.empty and len(cramers_mat) > 1:
            self.visualizer.plot_cramers_v_heatmap(
                cramers_mat, top_n=25, aggregated_vars=self.aggregator.aggregated_features
            )
        
        # 3.10 PCA
        try:
            pca_components, pca_obj = self.analyzer.pca_analysis(n_components=2)
            self.visualizer.plot_pca_2d(
                pca_components,
                self.df_aggregated[self.analyzer.target_col],
                pca_obj.explained_variance_ratio_
            )
        except Exception as e:
            print(f"Error en PCA: {e}")
        
        # 3.11 Generar reporte resumen
        print("\n" + "="*70)
        print("GENERACIÓN DE REPORTE")
        print("="*70)
        
        report_path = self._generate_custom_report()
        
        # Resumen final
        print("\n" + "="*70)
        print("PIPELINE COMPLETADO")
        print("="*70)
        print(f"\n✓ Pipeline de EDA con agregación completado exitosamente")
        print(f"\nArchivos generados:")
        print(f"  - Dataset agregado: data/processed/shock_aggregated.csv")
        print(f"  - Reporte resumen: {report_path}")
        print(f"  - Tablas CSV: {os.path.join(self.output_dir, 'tables/')}")
        print(f"  - Visualizaciones: {os.path.join(self.output_dir, 'figures/')}")
        print("\n" + "="*70)

    def _compare_aggregated_vs_original(
        self, 
        cat_vs_target: pd.DataFrame, 
        num_vs_target: pd.DataFrame
    ):
        """
        Compara poder predictivo de variables agregadas vs originales.
        
        Args:
            cat_vs_target: Resultados de análisis categórico vs target
            num_vs_target: Resultados de análisis numérico vs target
        """
        print("\nComparación de Poder Predictivo:")
        print("-" * 70)
        
        # Identificar variables agregadas
        aggregated_vars = self.aggregator.aggregated_features
        
        # Variables agregadas categóricas/numéricas
        if not cat_vs_target.empty:
            agg_categorical = cat_vs_target[
                cat_vs_target['Variable'].isin(aggregated_vars)
            ]
            
            if len(agg_categorical) > 0:
                print(f"\nVariables Agregadas Categóricas ({len(agg_categorical)}):")
                print(agg_categorical[['Variable', 'Cramers_V', 'p_value', 'Significant']].to_string(index=False))
                
                # Estadísticas
                sig_agg = agg_categorical[agg_categorical['p_value'] < 0.05]
                print(f"\n  ✓ Significativas: {len(sig_agg)}/{len(agg_categorical)} ({len(sig_agg)/len(agg_categorical)*100:.1f}%)")
                if len(agg_categorical) > 0:
                    print(f"  ✓ Cramer's V promedio: {agg_categorical['Cramers_V'].mean():.4f}")
                    print(f"  ✓ Cramer's V máximo: {agg_categorical['Cramers_V'].max():.4f}")
        
        if not num_vs_target.empty:
            agg_numerical = num_vs_target[
                num_vs_target['Variable'].isin(aggregated_vars)
            ]
            
            if len(agg_numerical) > 0:
                print(f"\nVariables Agregadas Numéricas ({len(agg_numerical)}):")
                print(agg_numerical[['Variable', 'Cohens_d', 'p_value_MW', 'Significant']].to_string(index=False))
                
                # Estadísticas
                sig_agg = agg_numerical[agg_numerical['p_value_MW'] < 0.05]
                print(f"\n  ✓ Significativas: {len(sig_agg)}/{len(agg_numerical)} ({len(sig_agg)/len(agg_numerical)*100:.1f}%)")
                if len(agg_numerical) > 0:
                    print(f"  ✓ |Cohen's d| promedio: {agg_numerical['Cohens_d'].abs().mean():.4f}")
                    print(f"  ✓ |Cohen's d| máximo: {agg_numerical['Cohens_d'].abs().max():.4f}")
        
        # Comparación con variables originales
        print("\n" + "-" * 70)
        print("COMPARACIÓN CON DATASET ORIGINAL:")
        print("-" * 70)
        print("\nVariables Originales (según EDA previo):")
        print("  - Variables categóricas significativas: 1/17 (5.9%)")
        print("  - Cramer's V máximo: 0.18 (SANGRADO_MAYOR)")
        print("  - Variables numéricas significativas: 1/2 (50%)")
        print("  - Cohen's d máximo: 0.29 (EDAD)")
        
        # Guardar comparación
        comparison_path = os.path.join(self.output_dir, "tables", "comparison_original_vs_aggregated.csv")
        os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
        
        comparison_data = []
        
        # Dataset original
        comparison_data.append({
            'Dataset': 'Original',
            'N_Variables_Categoricas': 17,
            'N_Cat_Significativas': 1,
            'Pct_Cat_Significativas': 5.9,
            'Max_Cramers_V': 0.18,
            'N_Variables_Numericas': 2,
            'N_Num_Significativas': 1,
            'Pct_Num_Significativas': 50.0,
            'Max_Cohens_d': 0.29
        })
        
        # Dataset agregado
        if not cat_vs_target.empty:
            agg_cat = cat_vs_target[cat_vs_target['Variable'].isin(aggregated_vars)]
            n_cat_sig = len(agg_cat[agg_cat['p_value'] < 0.05])
            pct_cat = (n_cat_sig / len(agg_cat) * 100) if len(agg_cat) > 0 else 0
            max_cramers = agg_cat['Cramers_V'].max() if len(agg_cat) > 0 else 0
        else:
            n_cat_sig, pct_cat, max_cramers = 0, 0, 0
        
        if not num_vs_target.empty:
            agg_num = num_vs_target[num_vs_target['Variable'].isin(aggregated_vars)]
            n_num_sig = len(agg_num[agg_num['p_value_MW'] < 0.05])
            pct_num = (n_num_sig / len(agg_num) * 100) if len(agg_num) > 0 else 0
            max_cohens = agg_num['Cohens_d'].abs().max() if len(agg_num) > 0 else 0
        else:
            n_num_sig, pct_num, max_cohens = 0, 0, 0
        
        comparison_data.append({
            'Dataset': 'Agregado',
            'N_Variables_Categoricas': len(agg_cat) if not cat_vs_target.empty else 0,
            'N_Cat_Significativas': n_cat_sig,
            'Pct_Cat_Significativas': pct_cat,
            'Max_Cramers_V': max_cramers,
            'N_Variables_Numericas': len(agg_num) if not num_vs_target.empty else 0,
            'N_Num_Significativas': n_num_sig,
            'Pct_Num_Significativas': pct_num,
            'Max_Cohens_d': max_cohens
        })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\n✓ Tabla comparativa guardada en: {comparison_path}")

    def _generate_custom_report(self) -> str:
        """
        Genera reporte resumen personalizado para dataset agregado.
        
        Returns:
            str: Ruta del archivo de reporte
        """
        report_path = os.path.join(self.output_dir, 'eda_aggregated_summary_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Reporte de EDA - Dataset con Variables Agregadas\n\n")
            f.write("## Predicción de Shock Hemorrágico\n\n")
            f.write("---\n\n")
            
            f.write("## Resumen Ejecutivo\n\n")
            f.write("Este análisis exploratorio se realizó sobre el dataset que incluye **variables agregadas**\n")
            f.write("creadas mediante la estrategia de feature engineering documentada en `doc/3.5_feature_aggregation.md`.\n\n")
            
            f.write("### Objetivo de la Agregación\n\n")
            f.write("Mejorar el poder predictivo del modelo mediante la creación de índices compuestos que capturen:\n\n")
            f.write("- Carga acumulativa de comorbilidades\n")
            f.write("- Riesgo por sistemas fisiológicos (cardiovascular, respiratorio, metabólico)\n")
            f.write("- Interacciones entre factores de riesgo\n")
            f.write("- Categorización de variables continuas en grupos de riesgo clínico\n\n")
            
            # Información básica
            info = self.analyzer.results['basic_info']
            f.write("## 1. Información del Dataset Agregado\n\n")
            f.write(f"- **Número de registros**: {info['n_rows']}\n")
            f.write(f"- **Número de variables**: {info['n_columns']}\n")
            f.write(f"- **Variables agregadas creadas**: {len(self.aggregator.aggregated_features)}\n\n")
            
            if info['target_distribution']:
                f.write("### Distribución del Target (SHOCK)\n\n")
                for k, v in info['target_distribution'].items():
                    pct = info['target_percentage'][k] * 100
                    f.write(f"- Clase {k}: {v} registros ({pct:.2f}%)\n")
                if info['imbalance_ratio']:
                    f.write(f"- **Ratio de desbalance**: {info['imbalance_ratio']:.4f}\n\n")
            
            # Variables agregadas creadas
            f.write("\n## 2. Variables Agregadas Creadas\n\n")
            for feat in self.aggregator.aggregated_features:
                f.write(f"- `{feat}`\n")
            
            # Análisis bivariado
            if 'categorical_vs_target' in self.analyzer.results['bivariate']:
                f.write("\n## 3. Poder Predictivo de Variables Agregadas\n\n")
                
                cat_results = self.analyzer.results['bivariate']['categorical_vs_target']
                agg_categorical = cat_results[
                    cat_results['Variable'].isin(self.aggregator.aggregated_features)
                ]
                
                if len(agg_categorical) > 0:
                    sig_cat = agg_categorical[agg_categorical['p_value'] < 0.05]
                    f.write(f"### Variables Categóricas Agregadas\n\n")
                    f.write(f"- **Total**: {len(agg_categorical)}\n")
                    f.write(f"- **Significativas (p < 0.05)**: {len(sig_cat)} ({len(sig_cat)/len(agg_categorical)*100:.1f}%)\n")
                    f.write(f"- **Cramer's V promedio**: {agg_categorical['Cramers_V'].mean():.4f}\n")
                    f.write(f"- **Cramer's V máximo**: {agg_categorical['Cramers_V'].max():.4f}\n\n")
                    
                    if len(sig_cat) > 0:
                        f.write("**Top variables agregadas significativas:**\n\n")
                        for _, row in sig_cat.head(10).iterrows():
                            f.write(f"- **{row['Variable']}**: Cramer's V = {row['Cramers_V']:.4f}, p = {row['p_value']:.4e}\n")
            
            if 'numerical_vs_target' in self.analyzer.results['bivariate']:
                num_results = self.analyzer.results['bivariate']['numerical_vs_target']
                agg_numerical = num_results[
                    num_results['Variable'].isin(self.aggregator.aggregated_features)
                ]
                
                if len(agg_numerical) > 0:
                    sig_num = agg_numerical[agg_numerical['p_value_MW'] < 0.05]
                    f.write(f"\n### Variables Numéricas Agregadas\n\n")
                    f.write(f"- **Total**: {len(agg_numerical)}\n")
                    f.write(f"- **Significativas (p < 0.05)**: {len(sig_num)} ({len(sig_num)/len(agg_numerical)*100:.1f}%)\n")
                    f.write(f"- **|Cohen's d| promedio**: {agg_numerical['Cohens_d'].abs().mean():.4f}\n")
                    f.write(f"- **|Cohen's d| máximo**: {agg_numerical['Cohens_d'].abs().max():.4f}\n\n")
                    
                    if len(sig_num) > 0:
                        f.write("**Top variables agregadas significativas:**\n\n")
                        for _, row in sig_num.head(10).iterrows():
                            f.write(f"- **{row['Variable']}**: Cohen's d = {row['Cohens_d']:.4f}, p = {row['p_value_MW']:.4e}\n")
            
            # Comparación con dataset original
            f.write("\n## 4. Comparación con Dataset Original\n\n")
            f.write("### Dataset Original (sin agregación)\n")
            f.write("- Variables categóricas significativas: **1/17 (5.9%)**\n")
            f.write("- Cramer's V máximo: **0.18** (SANGRADO_MAYOR)\n")
            f.write("- Variables numéricas significativas: **1/2 (50%)**\n")
            f.write("- Cohen's d máximo: **0.29** (EDAD)\n\n")
            
            f.write("### Mejora Obtenida\n")
            f.write("Ver tabla `comparison_original_vs_aggregated.csv` para análisis detallado.\n\n")
            
            # Conclusiones
            f.write("\n## 5. Conclusiones\n\n")
            f.write("### Hallazgos Clave:\n\n")
            f.write("1. La agregación de variables permite capturar la **carga acumulativa** de comorbilidades\n")
            f.write("2. Índices compuestos por sistemas fisiológicos reflejan mejor el riesgo multifactorial\n")
            f.write("3. Variables de interacción (ej: EDAD × CARGA_COMORBILIDADES) capturan efectos sinérgicos\n")
            f.write("4. Categorización de variables continuas puede revelar efectos de umbral clínico\n\n")
            
            f.write("### Próximos Pasos:\n\n")
            f.write("- Modelado con variables agregadas y comparación de métricas de rendimiento\n")
            f.write("- Análisis de importancia de features (SHAP values)\n")
            f.write("- Validación de coherencia clínica con expertos médicos\n")
            f.write("- Ajuste de pesos en score de riesgo compuesto\n\n")
            
            f.write("---\n\n")
            f.write("*Reporte generado automáticamente por EDA_Aggregated_Pipeline*\n")
            f.write(f"*Fecha: Noviembre 4, 2025*\n")
        
        print(f"\n✓ Reporte personalizado guardado en: {report_path}")
        
        return report_path


def main():
    """Función principal para ejecutar el pipeline."""
    # Configuración
    DATA_PATH = "data/processed/shock_cleaned.csv"
    OUTPUT_DIR = "reports/eda_aggregated"
    
    # Crear y ejecutar pipeline
    pipeline = EDA_Aggregated_Pipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR
    )
    
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
