"""
Módulo de visualizaciones para EDA de shock hemorrágico.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


class EDAVisualizer:
    """Clase para generar visualizaciones del EDA."""

    def __init__(self, output_dir: str = "reports/eda/figures"):
        """
        Inicializa el visualizador.

        Args:
            output_dir: Directorio de salida para figuras
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configuración de estilo
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def plot_target_distribution(self, df: pd.DataFrame, target_col: str) -> str:
        """
        Grafica distribución de la variable objetivo.

        Args:
            df: DataFrame con los datos
            target_col: Nombre de la columna objetivo

        Returns:
            str: Ruta del archivo guardado
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Countplot
        counts = df[target_col].value_counts()
        axes[0].bar(counts.index.astype(str), counts.values, color=['#3498db', '#e74c3c'])
        axes[0].set_xlabel('Clase')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title(f'Distribución de {target_col}')
        axes[0].grid(True, alpha=0.3)
        
        # Añadir porcentajes
        total = len(df)
        for i, (idx, val) in enumerate(counts.items()):
            pct = (val / total) * 100
            axes[0].text(i, val, f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        axes[1].pie(counts.values, labels=[f'Clase {i}' for i in counts.index], 
                   autopct='%1.1f%%', startangle=90, colors=['#3498db', '#e74c3c'])
        axes[1].set_title(f'Proporción de {target_col}')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'target_distribution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico guardado: {filepath}")
        return filepath

    def plot_numerical_distributions(self, df: pd.DataFrame, numerical_cols: List[str], 
                                     target_col: Optional[str] = None) -> str:
        """
        Grafica distribuciones de variables numéricas.

        Args:
            df: DataFrame con los datos
            numerical_cols: Lista de columnas numéricas
            target_col: Columna objetivo (opcional)

        Returns:
            str: Ruta del archivo guardado
        """
        if not numerical_cols:
            return None
        
        n_cols = min(len(numerical_cols), 4)
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for idx, col in enumerate(numerical_cols):
            ax = axes[idx]
            
            if target_col:
                # Histograma por clase
                for target_val in df[target_col].unique():
                    data = df[df[target_col] == target_val][col].dropna()
                    ax.hist(data, alpha=0.6, label=f'{target_col}={target_val}', bins=20)
                ax.legend()
            else:
                # Histograma simple
                ax.hist(df[col].dropna(), bins=20, color='steelblue', edgecolor='black')
            
            ax.set_xlabel(col)
            ax.set_ylabel('Frecuencia')
            ax.set_title(f'Distribución de {col}')
            ax.grid(True, alpha=0.3)
        
        # Ocultar ejes sobrantes
        for idx in range(len(numerical_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'numerical_distributions.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico guardado: {filepath}")
        return filepath

    def plot_numerical_boxplots(self, df: pd.DataFrame, numerical_cols: List[str], 
                                target_col: str) -> str:
        """
        Grafica boxplots de variables numéricas por clase.

        Args:
            df: DataFrame con los datos
            numerical_cols: Lista de columnas numéricas
            target_col: Columna objetivo

        Returns:
            str: Ruta del archivo guardado
        """
        if not numerical_cols:
            return None
        
        n_cols = min(len(numerical_cols), 3)
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for idx, col in enumerate(numerical_cols):
            ax = axes[idx]
            
            df_plot = df[[col, target_col]].dropna()
            sns.boxplot(data=df_plot, x=target_col, y=col, ax=ax, palette='Set2')
            ax.set_title(f'{col} por {target_col}')
            ax.set_xlabel(target_col)
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Ocultar ejes sobrantes
        for idx in range(len(numerical_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'numerical_boxplots_by_target.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico guardado: {filepath}")
        return filepath

    def plot_categorical_by_target(self, df: pd.DataFrame, categorical_cols: List[str], 
                                   target_col: str, top_n: int = 10) -> str:
        """
        Grafica barras apiladas de variables categóricas por target.

        Args:
            df: DataFrame con los datos
            categorical_cols: Lista de columnas categóricas
            target_col: Columna objetivo
            top_n: Número máximo de variables a graficar

        Returns:
            str: Ruta del archivo guardado
        """
        if not categorical_cols:
            return None
        
        cols_to_plot = categorical_cols[:top_n]
        n_cols = min(len(cols_to_plot), 2)
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 4*n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]
            
            # Crear tabla de contingencia normalizada
            ct = pd.crosstab(df[col], df[target_col], normalize='index') * 100
            
            ct.plot(kind='bar', stacked=False, ax=ax, color=['#3498db', '#e74c3c'])
            ax.set_title(f'{col} vs {target_col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Porcentaje (%)')
            ax.legend(title=target_col)
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Ocultar ejes sobrantes
        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'categorical_by_target.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico guardado: {filepath}")
        return filepath

    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame, title: str = "Correlación",
                                  aggregated_vars: Optional[List[str]] = None) -> str:
        """
        Grafica heatmap de correlaciones.

        Args:
            corr_matrix: Matriz de correlaciones
            title: Título del gráfico

        Returns:
            str: Ruta del archivo guardado
        """
        if corr_matrix.empty:
            return None
        
    fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, ax=ax,
                   cbar_kws={"shrink": 0.8}, linewidths=0.5)

        # Si se especificaron variables agregadas, colorear sus etiquetas en el heatmap
        if aggregated_vars:
            agg_set = set(aggregated_vars)
            # Eje X
            for lbl in ax.get_xticklabels():
                name = lbl.get_text()
                if name in agg_set:
                    lbl.set_color('#c0392b')  # rojo oscuro
                    lbl.set_fontweight('bold')
                else:
                    lbl.set_color('k')
            # Eje Y
            for lbl in ax.get_yticklabels():
                name = lbl.get_text()
                if name in agg_set:
                    lbl.set_color('#c0392b')
                    lbl.set_fontweight('bold')
                else:
                    lbl.set_color('k')
            # Nota en el gráfico indicando el significado del color
            fig.text(0.99, 0.01, 'Variables agregadas (rojo)', ha='right', va='bottom', fontsize=9, color='#c0392b')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filename = title.lower().replace(' ', '_').replace('á', 'a').replace('é', 'e')
        filepath = os.path.join(self.output_dir, f'{filename}_heatmap.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico guardado: {filepath}")
        return filepath

    def plot_cramers_v_heatmap(self, cramers_matrix: pd.DataFrame, top_n: int = 20,
                               aggregated_vars: Optional[List[str]] = None) -> str:
        """
        Grafica heatmap de Cramer's V (top variables).

        Args:
            cramers_matrix: Matriz de Cramer's V
            top_n: Número de variables a mostrar

        Returns:
            str: Ruta del archivo guardado
        """
        if cramers_matrix.empty or len(cramers_matrix) == 0:
            return None
        
        # Seleccionar top_n variables con mayor asociación promedio
        avg_cramers = cramers_matrix.mean(axis=1).sort_values(ascending=False)
        top_vars = avg_cramers.head(top_n).index.tolist()
        
        cramers_subset = cramers_matrix.loc[top_vars, top_vars]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(cramers_subset, dtype=bool))
        
        sns.heatmap(cramers_subset, mask=mask, annot=True, fmt='.2f', 
                   cmap='YlOrRd', vmin=0, vmax=1, square=True, ax=ax,
                   cbar_kws={"shrink": 0.8}, linewidths=0.5)

        # Colorear etiquetas si se indicó conjunto de variables agregadas
        if aggregated_vars:
            agg_set = set(aggregated_vars)
            for lbl in ax.get_xticklabels():
                name = lbl.get_text()
                if name in agg_set:
                    lbl.set_color('#c0392b')
                    lbl.set_fontweight('bold')
                else:
                    lbl.set_color('k')
            for lbl in ax.get_yticklabels():
                name = lbl.get_text()
                if name in agg_set:
                    lbl.set_color('#c0392b')
                    lbl.set_fontweight('bold')
                else:
                    lbl.set_color('k')
            fig.text(0.99, 0.01, 'Variables agregadas (rojo)', ha='right', va='bottom', fontsize=9, color='#c0392b')
        
        ax.set_title("Matriz de Cramer's V (Top Variables Categóricas)", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'cramers_v_heatmap.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico guardado: {filepath}")
        return filepath

    def plot_pca_2d(self, pca_components: np.ndarray, target: pd.Series, 
                    variance_explained: np.ndarray) -> str:
        """
        Grafica PCA en 2D.

        Args:
            pca_components: Componentes principales
            target: Variable objetivo
            variance_explained: Varianza explicada por cada componente

        Returns:
            str: Ruta del archivo guardado
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(pca_components[:, 0], pca_components[:, 1], 
                           c=target, cmap='coolwarm', alpha=0.6, edgecolors='k', linewidth=0.5)
        
        ax.set_xlabel(f'PC1 ({variance_explained[0]*100:.2f}% varianza)', fontsize=12)
        ax.set_ylabel(f'PC2 ({variance_explained[1]*100:.2f}% varianza)', fontsize=12)
        ax.set_title('Análisis de Componentes Principales (PCA)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Clase (Shock)', fontsize=12)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'pca_2d_plot.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico guardado: {filepath}")
        return filepath

    def plot_missing_values(self, df: pd.DataFrame, top_n: int = 20) -> str:
        """
        Grafica valores faltantes.

        Args:
            df: DataFrame con los datos
            top_n: Número de variables a mostrar

        Returns:
            str: Ruta del archivo guardado
        """
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            print("No hay valores faltantes para graficar")
            return None
        
        missing_top = missing.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(missing_top) * 0.4)))
        
        missing_top.plot(kind='barh', ax=ax, color='coral')
        ax.set_xlabel('Número de Valores Faltantes', fontsize=12)
        ax.set_ylabel('Variable', fontsize=12)
        ax.set_title(f'Top {len(missing_top)} Variables con Valores Faltantes', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'missing_values.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico guardado: {filepath}")
        return filepath

    def plot_top_associations(self, results_df: pd.DataFrame, metric_col: str, 
                             top_n: int = 15, title: str = "Top Asociaciones") -> str:
        """
        Grafica top asociaciones con el target.

        Args:
            results_df: DataFrame con resultados de tests
            metric_col: Nombre de la columna con la métrica
            top_n: Número de variables a mostrar
            title: Título del gráfico

        Returns:
            str: Ruta del archivo guardado
        """
        if results_df.empty:
            return None
        
        top_results = results_df.nlargest(top_n, metric_col)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(top_results) * 0.4)))
        
        # Buscar columna de p-value (puede ser 'p_value' o 'p_value_MW')
        p_col = None
        for col in ['p_value', 'p_value_MW', 'p_value_t']:
            if col in top_results.columns:
                p_col = col
                break
        
        # Asignar colores según significancia (si hay p-value)
        if p_col:
            colors = ['#2ecc71' if p < 0.05 else '#95a5a6' for p in top_results[p_col]]
        else:
            colors = '#3498db'  # Color único si no hay p-value
        
        ax.barh(top_results['Variable'], top_results[metric_col], color=colors)
        ax.set_xlabel(metric_col, fontsize=12)
        ax.set_ylabel('Variable', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Leyenda (solo si hay p-value)
        if p_col:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ecc71', label='p < 0.05 (Significativa)'),
                Patch(facecolor='#95a5a6', label='p ≥ 0.05 (No significativa)')
            ]
            ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        
        # Limpiar nombre de archivo (remover caracteres inválidos en Windows)
        filename = title.lower().replace(' ', '_').replace('á', 'a').replace('í', 'i')
        filename = filename.replace('|', '').replace(':', '').replace('<', '').replace('>', '')
        filename = filename.replace('"', '').replace('/', '').replace('\\', '').replace('?', '')
        
        filepath = os.path.join(self.output_dir, f'{filename}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico guardado: {filepath}")
        return filepath

    def plot_class_imbalance(self, df: pd.DataFrame, target_col: str) -> str:
        """
        Grafica análisis detallado de desbalance de clases.

        Args:
            df: DataFrame con los datos
            target_col: Columna objetivo

        Returns:
            str: Ruta del archivo guardado
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        counts = df[target_col].value_counts()
        
        # Gráfico 1: Barras
        axes[0].bar(counts.index.astype(str), counts.values, color=['#3498db', '#e74c3c'])
        axes[0].set_xlabel('Clase', fontsize=12)
        axes[0].set_ylabel('Frecuencia', fontsize=12)
        axes[0].set_title('Distribución de Clases', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        for i, (idx, val) in enumerate(counts.items()):
            axes[0].text(i, val, f'{val}', ha='center', va='bottom', fontweight='bold')
        
        # Gráfico 2: Porcentajes
        percentages = counts / counts.sum() * 100
        axes[1].bar(percentages.index.astype(str), percentages.values, color=['#3498db', '#e74c3c'])
        axes[1].set_xlabel('Clase', fontsize=12)
        axes[1].set_ylabel('Porcentaje (%)', fontsize=12)
        axes[1].set_title('Porcentaje por Clase', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Balance perfecto (50%)')
        axes[1].legend()
        
        for i, (idx, val) in enumerate(percentages.items()):
            axes[1].text(i, val, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Gráfico 3: Ratio
        if len(counts) == 2:
            minority_class = counts.min()
            majority_class = counts.max()
            ratio = minority_class / majority_class
            
            axes[2].text(0.5, 0.6, f'Ratio de Desbalance', ha='center', va='center', 
                        fontsize=16, fontweight='bold')
            axes[2].text(0.5, 0.4, f'{ratio:.4f}', ha='center', va='center', 
                        fontsize=40, color='#e74c3c', fontweight='bold')
            axes[2].text(0.5, 0.25, f'Minoritaria: {minority_class}', ha='center', va='center', fontsize=12)
            axes[2].text(0.5, 0.15, f'Mayoritaria: {majority_class}', ha='center', va='center', fontsize=12)
            axes[2].axis('off')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'class_imbalance_analysis.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico guardado: {filepath}")
        return filepath
