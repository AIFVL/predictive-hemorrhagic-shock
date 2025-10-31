"""
Módulo de Análisis Exploratorio de Datos (EDA) para predicción de shock hemorrágico.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ShockEDAAnalyzer:
    """Clase para análisis exploratorio de datos de shock hemorrágico."""

    def __init__(self, output_dir: str = "reports/eda"):
        """
        Inicializa el analizador de EDA.

        Args:
            output_dir: Directorio de salida para reportes y visualizaciones
        """
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, "figures")
        self.tables_dir = os.path.join(output_dir, "tables")
        
        # Crear directorios
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)
        
        # Configuración de visualización
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        self.df = None
        self.target_col = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.binary_cols = []
        
        # Almacenar resultados
        self.results = {
            'basic_info': {},
            'univariate': {},
            'bivariate': {},
            'multivariate': {},
            'feature_engineering': {}
        }

    def load_data(self, data_path: str, target_col: str = None) -> pd.DataFrame:
        """
        Carga datos desde CSV.

        Args:
            data_path: Ruta al archivo CSV
            target_col: Nombre de la columna objetivo

        Returns:
            pd.DataFrame: Datos cargados
        """
        print("\n" + "="*70)
        print("CARGA DE DATOS")
        print("="*70)
        
        self.df = pd.read_csv(data_path)
        
        # Detectar target (búsqueda case-insensitive)
        if target_col is None:
            # Búsqueda automática de columna con 'shock'
            self.target_col = next((c for c in self.df.columns if 'shock' in c.lower()), None)
        else:
            # Búsqueda case-insensitive del target proporcionado
            target_lower = target_col.lower()
            self.target_col = next(
                (c for c in self.df.columns if c.lower() == target_lower),
                target_col  # Fallback al valor original si no se encuentra
            )
            
        # Validar que el target existe
        if self.target_col not in self.df.columns:
            raise ValueError(
                f"Columna objetivo '{target_col}' no encontrada. "
                f"Columnas disponibles: {self.df.columns.tolist()}"
            )
            
        print(f"Datos cargados: {self.df.shape[0]} filas x {self.df.shape[1]} columnas")
        print(f"Columna objetivo: {self.target_col}")
        
        return self.df

    def basic_analysis(self) -> Dict:
        """
        Realiza análisis básico del dataset.

        Returns:
            Dict: Resultados del análisis básico
        """
        print("\n" + "="*70)
        print("ANÁLISIS BÁSICO DEL DATASET")
        print("="*70)
        
        # Dimensiones
        n_rows, n_cols = self.df.shape
        
        # Tipos de datos
        dtypes_count = self.df.dtypes.value_counts().to_dict()
        
        # Valores únicos
        unique_counts = self.df.nunique().to_dict()
        
        # Duplicados
        n_duplicates = self.df.duplicated().sum()
        
        # Valores nulos
        null_counts = self.df.isnull().sum()
        total_nulls = null_counts.sum()
        
        # Distribución del target
        if self.target_col:
            target_dist = self.df[self.target_col].value_counts().to_dict()
            target_pct = self.df[self.target_col].value_counts(normalize=True).to_dict()
            
            # Calcular desbalance
            if len(target_pct) == 2:
                imbalance_ratio = min(target_pct.values()) / max(target_pct.values())
            else:
                imbalance_ratio = None
        else:
            target_dist = None
            target_pct = None
            imbalance_ratio = None
        
        results = {
            'n_rows': n_rows,
            'n_columns': n_cols,
            'dtypes': dtypes_count,
            'n_duplicates': n_duplicates,
            'total_nulls': total_nulls,
            'target_distribution': target_dist,
            'target_percentage': target_pct,
            'imbalance_ratio': imbalance_ratio
        }
        
        self.results['basic_info'] = results
        
        # Imprimir resultados
        print(f"\nDimensiones: {n_rows} filas x {n_cols} columnas")
        print(f"Duplicados: {n_duplicates}")
        print(f"Valores nulos totales: {total_nulls}")
        
        if target_dist:
            print(f"\nDistribución del target '{self.target_col}':")
            for k, v in target_dist.items():
                pct = target_pct[k] * 100
                print(f"  Clase {k}: {v} ({pct:.2f}%)")
            if imbalance_ratio:
                print(f"\nRatio de desbalance: {imbalance_ratio:.4f}")
        
        # Guardar tabla
        self._save_basic_info_table(results)
        
        return results

    def identify_variable_types(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Identifica tipos de variables.

        Returns:
            Tuple[List[str], List[str], List[str]]: (numéricas, categóricas, binarias)
        """
        print("\n" + "="*70)
        print("TIPIFICACIÓN DE VARIABLES")
        print("="*70)
        
        exclude_cols = ['CODIGO', 'codigo', self.target_col]
        
        # Variables numéricas (continuas)
        self.numerical_cols = [
            col for col in self.df.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols and self.df[col].nunique() > 10
        ]
        
        # Variables binarias
        self.binary_cols = [
            col for col in self.df.columns
            if col not in exclude_cols and self.df[col].nunique() == 2
        ]
        
        # Variables categóricas
        self.categorical_cols = [
            col for col in self.df.columns
            if col not in exclude_cols + self.numerical_cols + self.binary_cols
        ]
        
        print(f"\nVariables numéricas (continuas): {len(self.numerical_cols)}")
        print(f"  {self.numerical_cols[:5]}..." if len(self.numerical_cols) > 5 else f"  {self.numerical_cols}")
        
        print(f"\nVariables binarias: {len(self.binary_cols)}")
        print(f"  {self.binary_cols[:10]}..." if len(self.binary_cols) > 10 else f"  {self.binary_cols}")
        
        print(f"\nVariables categóricas: {len(self.categorical_cols)}")
        print(f"  {self.categorical_cols}")
        
        # Guardar tipificación
        type_df = pd.DataFrame({
            'Variable': self.numerical_cols + self.binary_cols + self.categorical_cols,
            'Tipo': ['Numérica']*len(self.numerical_cols) + ['Binaria']*len(self.binary_cols) + ['Categórica']*len(self.categorical_cols),
            'N_Unicos': [self.df[col].nunique() for col in self.numerical_cols + self.binary_cols + self.categorical_cols]
        })
        type_df.to_csv(os.path.join(self.tables_dir, 'variable_types.csv'), index=False)
        
        return self.numerical_cols, self.categorical_cols, self.binary_cols

    def univariate_analysis_categorical(self) -> pd.DataFrame:
        """
        Análisis univariado de variables categóricas.

        Returns:
            pd.DataFrame: Resumen de frecuencias
        """
        print("\n" + "="*70)
        print("ANÁLISIS UNIVARIADO - VARIABLES CATEGÓRICAS/BINARIAS")
        print("="*70)
        
        all_categorical = self.categorical_cols + self.binary_cols
        
        freq_summary = []
        
        for col in all_categorical:
            freq = self.df[col].value_counts()
            freq_pct = self.df[col].value_counts(normalize=True) * 100
            
            for val in freq.index:
                freq_summary.append({
                    'Variable': col,
                    'Valor': val,
                    'Frecuencia': freq[val],
                    'Porcentaje': freq_pct[val]
                })
        
        freq_df = pd.DataFrame(freq_summary)
        
        # Guardar tabla
        freq_df.to_csv(os.path.join(self.tables_dir, 'categorical_frequencies.csv'), index=False)
        
        self.results['univariate']['categorical'] = freq_df
        
        print(f"\nAnálisis completado para {len(all_categorical)} variables categóricas/binarias")
        
        return freq_df

    def univariate_analysis_numerical(self) -> pd.DataFrame:
        """
        Análisis univariado de variables numéricas.

        Returns:
            pd.DataFrame: Estadísticas descriptivas
        """
        print("\n" + "="*70)
        print("ANÁLISIS UNIVARIADO - VARIABLES NUMÉRICAS")
        print("="*70)
        
        if not self.numerical_cols:
            print("No hay variables numéricas para analizar")
            return pd.DataFrame()
        
        # Estadísticas descriptivas
        desc_stats = self.df[self.numerical_cols].describe().T
        
        # Agregar asimetría y curtosis
        desc_stats['skewness'] = self.df[self.numerical_cols].skew()
        desc_stats['kurtosis'] = self.df[self.numerical_cols].kurtosis()
        
        # Valores nulos
        desc_stats['nulls'] = self.df[self.numerical_cols].isnull().sum()
        
        # Guardar tabla
        desc_stats.to_csv(os.path.join(self.tables_dir, 'numerical_statistics.csv'))
        
        self.results['univariate']['numerical'] = desc_stats
        
        print(f"\nEstadísticas calculadas para {len(self.numerical_cols)} variables numéricas")
        print(f"\n{desc_stats[['mean', 'std', 'min', 'max', 'skewness']]}")
        
        return desc_stats

    def cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """
        Calcula Cramer's V para medir asociación entre variables categóricas.

        Args:
            x: Primera variable categórica
            y: Segunda variable categórica

        Returns:
            float: Valor de Cramer's V (0-1)
        """
        confusion_matrix = pd.crosstab(x, y)
        
        if confusion_matrix.size == 0 or confusion_matrix.shape[0] == 1 or confusion_matrix.shape[1] == 1:
            return 0.0
        
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        denom = min(kcorr-1, rcorr-1)
        
        if denom <= 0:
            return 0.0
        
        return np.sqrt(phi2corr / denom)

    def bivariate_analysis_categorical_vs_target(self) -> pd.DataFrame:
        """
        Análisis bivariado: variables categóricas vs target.

        Returns:
            pd.DataFrame: Resultados de tests estadísticos
        """
        print("\n" + "="*70)
        print("ANÁLISIS BIVARIADO - CATEGÓRICAS vs TARGET")
        print("="*70)
        
        if not self.target_col:
            print("No se ha definido columna objetivo")
            return pd.DataFrame()
        
        all_categorical = self.categorical_cols + self.binary_cols
        
        results = []
        
        for col in all_categorical:
            try:
                # Chi-cuadrado
                contingency = pd.crosstab(self.df[col], self.df[self.target_col])
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                
                # Cramer's V
                cramers = self.cramers_v(self.df[col], self.df[self.target_col])
                
                results.append({
                    'Variable': col,
                    'Chi2': chi2,
                    'p_value': p_value,
                    'Cramers_V': cramers,
                    'Significant': 'Sí' if p_value < 0.05 else 'No'
                })
            except Exception as e:
                print(f"Error procesando {col}: {e}")
        
        results_df = pd.DataFrame(results).sort_values('Cramers_V', ascending=False)
        
        # Guardar tabla
        results_df.to_csv(os.path.join(self.tables_dir, 'categorical_vs_target.csv'), index=False)
        
        self.results['bivariate']['categorical_vs_target'] = results_df
        
        print(f"\nVariables con asociación significativa (p < 0.05):")
        sig_vars = results_df[results_df['p_value'] < 0.05]
        print(f"  Total: {len(sig_vars)}/{len(results_df)}")
        
        if len(sig_vars) > 0:
            print(f"\nTop 5 asociaciones más fuertes (Cramer's V):")
            print(sig_vars[['Variable', 'Cramers_V', 'p_value']].head())
        
        return results_df

    def bivariate_analysis_numerical_vs_target(self) -> pd.DataFrame:
        """
        Análisis bivariado: variables numéricas vs target.

        Returns:
            pd.DataFrame: Resultados de tests estadísticos
        """
        print("\n" + "="*70)
        print("ANÁLISIS BIVARIADO - NUMÉRICAS vs TARGET")
        print("="*70)
        
        if not self.target_col or not self.numerical_cols:
            print("No hay variables numéricas o target no definido")
            return pd.DataFrame()
        
        results = []
        
        for col in self.numerical_cols:
            try:
                # Separar por grupos
                group_0 = self.df[self.df[self.target_col] == 0][col].dropna()
                group_1 = self.df[self.df[self.target_col] == 1][col].dropna()
                
                if len(group_0) < 2 or len(group_1) < 2:
                    continue
                
                # Mann-Whitney U test (no paramétrico)
                u_stat, p_value_mw = mannwhitneyu(group_0, group_1, alternative='two-sided')
                
                # T-test
                t_stat, p_value_t = ttest_ind(group_0, group_1, equal_var=False)
                
                # Cohen's d (tamaño del efecto)
                mean_0, mean_1 = group_0.mean(), group_1.mean()
                std_0, std_1 = group_0.std(), group_1.std()
                pooled_std = np.sqrt(((len(group_0)-1)*std_0**2 + (len(group_1)-1)*std_1**2) / (len(group_0)+len(group_1)-2))
                cohens_d = (mean_1 - mean_0) / pooled_std if pooled_std > 0 else 0
                
                results.append({
                    'Variable': col,
                    'Mean_NoShock': mean_0,
                    'Mean_Shock': mean_1,
                    'Std_NoShock': std_0,
                    'Std_Shock': std_1,
                    'MannWhitney_U': u_stat,
                    'p_value_MW': p_value_mw,
                    't_statistic': t_stat,
                    'p_value_t': p_value_t,
                    'Cohens_d': cohens_d,
                    'Significant': 'Sí' if p_value_mw < 0.05 else 'No'
                })
            except Exception as e:
                print(f"Error procesando {col}: {e}")
        
        results_df = pd.DataFrame(results).sort_values('p_value_MW')
        
        # Guardar tabla
        results_df.to_csv(os.path.join(self.tables_dir, 'numerical_vs_target.csv'), index=False)
        
        self.results['bivariate']['numerical_vs_target'] = results_df
        
        print(f"\nVariables con diferencias significativas (p < 0.05):")
        sig_vars = results_df[results_df['p_value_MW'] < 0.05]
        print(f"  Total: {len(sig_vars)}/{len(results_df)}")
        
        if len(sig_vars) > 0:
            print(f"\nTop 5 diferencias más significativas:")
            print(sig_vars[['Variable', 'Cohens_d', 'p_value_MW']].head())
        
        return results_df

    def correlation_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Análisis de correlaciones (numéricas y categóricas).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (correlaciones numéricas, matriz Cramer's V)
        """
        print("\n" + "="*70)
        print("ANÁLISIS DE CORRELACIONES")
        print("="*70)
        
        # Correlaciones numéricas (Spearman)
        if len(self.numerical_cols) > 1:
            corr_matrix = self.df[self.numerical_cols].corr(method='spearman')
            corr_matrix.to_csv(os.path.join(self.tables_dir, 'correlation_matrix_numerical.csv'))
            print(f"\nMatriz de correlación Spearman guardada ({len(self.numerical_cols)} variables)")
        else:
            corr_matrix = pd.DataFrame()
            print("\nNo hay suficientes variables numéricas para calcular correlaciones")
        
        # Matriz Cramer's V (categóricas)
        all_categorical = self.categorical_cols + self.binary_cols
        
        if len(all_categorical) > 1:
            cramers_matrix = pd.DataFrame(
                np.zeros((len(all_categorical), len(all_categorical))),
                index=all_categorical,
                columns=all_categorical
            )
            
            for i, col1 in enumerate(all_categorical):
                for j, col2 in enumerate(all_categorical):
                    if i <= j:
                        v = self.cramers_v(self.df[col1], self.df[col2])
                        cramers_matrix.iloc[i, j] = v
                        cramers_matrix.iloc[j, i] = v
            
            cramers_matrix.to_csv(os.path.join(self.tables_dir, 'cramers_v_matrix.csv'))
            print(f"Matriz de Cramer's V guardada ({len(all_categorical)} variables)")
        else:
            cramers_matrix = pd.DataFrame()
            print("No hay suficientes variables categóricas para Cramer's V")
        
        self.results['multivariate']['correlations'] = {
            'numerical': corr_matrix,
            'categorical': cramers_matrix
        }
        
        return corr_matrix, cramers_matrix

    def pca_analysis(self, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
        """
        Análisis de componentes principales (PCA).

        Args:
            n_components: Número de componentes

        Returns:
            Tuple[np.ndarray, PCA]: (componentes transformados, objeto PCA)
        """
        print("\n" + "="*70)
        print("ANÁLISIS PCA (REDUCCIÓN DE DIMENSIONALIDAD)")
        print("="*70)
        
        # Preparar datos: one-hot encoding para categóricas
        df_encoded = pd.get_dummies(self.df.drop(columns=[self.target_col, 'CODIGO'] if 'CODIGO' in self.df.columns else [self.target_col]), 
                                     drop_first=True)
        
        # Manejar valores nulos
        df_encoded = df_encoded.fillna(df_encoded.median())
        
        # Escalar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_encoded)
        
        # PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Varianza explicada
        var_explained = pca.explained_variance_ratio_
        
        print(f"\nVarianza explicada por componente:")
        for i, var in enumerate(var_explained):
            print(f"  PC{i+1}: {var*100:.2f}%")
        print(f"  Total: {sum(var_explained)*100:.2f}%")
        
        # Guardar resultados
        pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        pca_df['target'] = self.df[self.target_col].values
        pca_df.to_csv(os.path.join(self.tables_dir, 'pca_components.csv'), index=False)
        
        # Guardar varianza explicada
        var_df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(len(var_explained))],
            'Variance_Explained': var_explained,
            'Cumulative_Variance': np.cumsum(var_explained)
        })
        var_df.to_csv(os.path.join(self.tables_dir, 'pca_variance_explained.csv'), index=False)
        
        self.results['multivariate']['pca'] = {
            'components': X_pca,
            'variance_explained': var_explained,
            'pca_object': pca
        }
        
        return X_pca, pca

    def feature_engineering_analysis(self) -> pd.DataFrame:
        """
        Análisis de feature engineering exploratorio.

        Returns:
            pd.DataFrame: Nuevas variables y su asociación con el target
        """
        print("\n" + "="*70)
        print("FEATURE ENGINEERING EXPLORATORIO")
        print("="*70)
        
        # Crear variables compuestas (adaptar según el dataset real)
        new_features = {}
        
        # Ejemplo: Conteo de comorbilidades
        # Identificar columnas de comorbilidades (booleanas)
        comorbidity_cols = [col for col in self.binary_cols if any(
            keyword in col.lower() for keyword in ['enfermedad', 'diabetes', 'hipertension', 'falla', 'insuficiencia']
        )]
        
        if comorbidity_cols:
            new_features['n_comorbilidades'] = self.df[comorbidity_cols].sum(axis=1)
            print(f"\nVariable creada: 'n_comorbilidades' (suma de {len(comorbidity_cols)} condiciones)")
        
        # Crear DataFrame con nuevas features
        if new_features:
            new_df = pd.DataFrame(new_features)
            new_df[self.target_col] = self.df[self.target_col]
            
            # Analizar asociación con target
            results = []
            
            for feat in new_features.keys():
                if new_df[feat].nunique() > 2:
                    # Variable numérica
                    group_0 = new_df[new_df[self.target_col] == 0][feat].dropna()
                    group_1 = new_df[new_df[self.target_col] == 1][feat].dropna()
                    
                    if len(group_0) > 0 and len(group_1) > 0:
                        u_stat, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')
                        
                        results.append({
                            'Feature': feat,
                            'Type': 'Numérica',
                            'Mean_NoShock': group_0.mean(),
                            'Mean_Shock': group_1.mean(),
                            'p_value': p_value,
                            'Significant': 'Sí' if p_value < 0.05 else 'No'
                        })
                else:
                    # Variable categórica/binaria
                    contingency = pd.crosstab(new_df[feat], new_df[self.target_col])
                    chi2, p_value, _, _ = chi2_contingency(contingency)
                    cramers = self.cramers_v(new_df[feat], new_df[self.target_col])
                    
                    results.append({
                        'Feature': feat,
                        'Type': 'Categórica',
                        'Cramers_V': cramers,
                        'p_value': p_value,
                        'Significant': 'Sí' if p_value < 0.05 else 'No'
                    })
            
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(self.tables_dir, 'feature_engineering_results.csv'), index=False)
            
            print(f"\nResultados de {len(results)} nuevas features guardados")
            print(results_df)
            
            self.results['feature_engineering'] = results_df
            
            return results_df
        else:
            print("\nNo se crearon nuevas features")
            return pd.DataFrame()

    def _save_basic_info_table(self, info: Dict):
        """Guarda información básica en tabla."""
        basic_df = pd.DataFrame([
            {'Métrica': 'Número de filas', 'Valor': info['n_rows']},
            {'Métrica': 'Número de columnas', 'Valor': info['n_columns']},
            {'Métrica': 'Duplicados', 'Valor': info['n_duplicates']},
            {'Métrica': 'Valores nulos totales', 'Valor': info['total_nulls']},
            {'Métrica': 'Ratio de desbalance', 'Valor': f"{info['imbalance_ratio']:.4f}" if info['imbalance_ratio'] else 'N/A'}
        ])
        basic_df.to_csv(os.path.join(self.tables_dir, 'basic_info.csv'), index=False)

    def generate_summary_report(self) -> str:
        """
        Genera reporte resumen en formato markdown.

        Returns:
            str: Ruta del archivo de reporte
        """
        print("\n" + "="*70)
        print("GENERANDO REPORTE RESUMEN")
        print("="*70)
        
        report_path = os.path.join(self.output_dir, 'eda_summary_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Reporte de Análisis Exploratorio de Datos (EDA)\n\n")
            f.write("## Predicción de Shock Hemorrágico\n\n")
            f.write("---\n\n")
            
            # Información básica
            f.write("## 1. Información General del Dataset\n\n")
            info = self.results['basic_info']
            f.write(f"- **Número de registros**: {info['n_rows']}\n")
            f.write(f"- **Número de variables**: {info['n_columns']}\n")
            f.write(f"- **Duplicados**: {info['n_duplicates']}\n")
            f.write(f"- **Valores nulos totales**: {info['total_nulls']}\n\n")
            
            if info['target_distribution']:
                f.write("### Distribución de la Variable Objetivo\n\n")
                for k, v in info['target_distribution'].items():
                    pct = info['target_percentage'][k] * 100
                    f.write(f"- Clase {k}: {v} registros ({pct:.2f}%)\n")
                if info['imbalance_ratio']:
                    f.write(f"- **Ratio de desbalance**: {info['imbalance_ratio']:.4f}\n\n")
            
            # Tipos de variables
            f.write("\n## 2. Tipificación de Variables\n\n")
            f.write(f"- **Variables numéricas**: {len(self.numerical_cols)}\n")
            f.write(f"- **Variables binarias**: {len(self.binary_cols)}\n")
            f.write(f"- **Variables categóricas**: {len(self.categorical_cols)}\n\n")
            
            # Análisis bivariado
            if 'categorical_vs_target' in self.results['bivariate']:
                f.write("\n## 3. Variables Significativas vs Target\n\n")
                cat_results = self.results['bivariate']['categorical_vs_target']
                sig_cat = cat_results[cat_results['p_value'] < 0.05]
                f.write(f"### Variables Categóricas ({len(sig_cat)} significativas)\n\n")
                if len(sig_cat) > 0:
                    f.write("Top 5 asociaciones más fuertes (Cramer's V):\n\n")
                    for _, row in sig_cat.head().iterrows():
                        f.write(f"- **{row['Variable']}**: Cramer's V = {row['Cramers_V']:.4f}, p = {row['p_value']:.4e}\n")
            
            if 'numerical_vs_target' in self.results['bivariate']:
                num_results = self.results['bivariate']['numerical_vs_target']
                sig_num = num_results[num_results['p_value_MW'] < 0.05]
                f.write(f"\n### Variables Numéricas ({len(sig_num)} significativas)\n\n")
                if len(sig_num) > 0:
                    f.write("Top 5 diferencias más significativas:\n\n")
                    for _, row in sig_num.head().iterrows():
                        f.write(f"- **{row['Variable']}**: Cohen's d = {row['Cohens_d']:.4f}, p = {row['p_value_MW']:.4e}\n")
            
            # Conclusiones
            f.write("\n## 4. Conclusiones Principales\n\n")
            f.write("### Hallazgos Clave:\n\n")
            f.write("- Dataset clínico con predominancia de variables categóricas/binarias\n")
            f.write(f"- Desbalance de clases: {info['imbalance_ratio']:.2%} (requiere manejo especial)\n" if info['imbalance_ratio'] else "")
            f.write("- Ver archivos CSV en `tables/` para análisis detallados\n")
            f.write("- Ver visualizaciones en `figures/` para interpretación gráfica\n\n")
            
            f.write("### Archivos Generados:\n\n")
            f.write("**Tablas:**\n")
            f.write("- `basic_info.csv`: Información general\n")
            f.write("- `variable_types.csv`: Tipificación de variables\n")
            f.write("- `numerical_statistics.csv`: Estadísticas descriptivas\n")
            f.write("- `categorical_vs_target.csv`: Tests Chi-cuadrado y Cramer's V\n")
            f.write("- `numerical_vs_target.csv`: Tests Mann-Whitney y Cohen's d\n")
            f.write("- `correlation_matrix_numerical.csv`: Correlaciones Spearman\n")
            f.write("- `cramers_v_matrix.csv`: Asociaciones categóricas\n")
            f.write("- `pca_components.csv`: Componentes principales\n")
            f.write("- `feature_engineering_results.csv`: Nuevas features\n\n")
            
            f.write("**Visualizaciones:**\n")
            f.write("- Ver directorio `figures/` para gráficos generados\n\n")
            
            f.write("---\n\n")
            f.write(f"*Reporte generado automáticamente por ShockEDAAnalyzer*\n")
        
        print(f"\nReporte guardado en: {report_path}")
        
        return report_path
