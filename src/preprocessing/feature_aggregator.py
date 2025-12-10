"""
Módulo de agregación de features para mejorar poder predictivo.

Este módulo implementa la estrategia de agregación de variables documentada
en doc/3.5_feature_aggregation.md, creando variables compuestas basadas en
agrupaciones fisiopatológicas y evidencia clínica.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureAggregator:
    """
    Clase para agregar variables clínicas en índices compuestos.
    
    Crea variables agregadas basadas en:
    - Carga de comorbilidades por sistemas fisiológicos
    - Categorización de variables continuas en grupos de riesgo
    - Interacciones entre factores de riesgo
    """
    
    def __init__(self, verbose: bool = True):
        """
        Inicializa el agregador de features.
        
        Args:
            verbose: Si True, imprime mensajes de progreso
        """
        self.verbose = verbose
        self.feature_mapping = {}  # Mapeo de features originales a agregadas
        self.aggregated_features = []  # Lista de features creadas
        
    def aggregate_comorbidities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea variable de carga total de comorbilidades.
        
        Args:
            df: DataFrame con variables clínicas
            
        Returns:
            pd.DataFrame: DataFrame con nueva columna CARGA_COMORBILIDADES
        """
        comorbidity_cols = [
            'HIPERTENSION', 'DIABETES', 'ENFERMEDAD_CORONARIA',
            'FALLA_CARDIACA', 'HIPOTIROIDISMO', 'ERC',
            'INMUNOSUPRESION', 'OBESIDAD', 'HIPERTENSION_PULMONAR',
            'EPOC', 'ASMA', 'ENF_CEREBROVASCULAR', 'CANCER_ACTIVO'
        ]
        
        # Filtrar columnas que existen
        available_cols = [col for col in comorbidity_cols if col in df.columns]
        
        # Crear variable agregada (suma de comorbilidades)
        df['CARGA_COMORBILIDADES'] = df[available_cols].sum(axis=1)
        
        self.feature_mapping['CARGA_COMORBILIDADES'] = available_cols
        self.aggregated_features.append('CARGA_COMORBILIDADES')
        
        if self.verbose:
            print(f"✓ CARGA_COMORBILIDADES creada (suma de {len(available_cols)} condiciones)")
            print(f"  Rango: {df['CARGA_COMORBILIDADES'].min():.0f} - {df['CARGA_COMORBILIDADES'].max():.0f}")
            print(f"  Media: {df['CARGA_COMORBILIDADES'].mean():.2f} (SD: {df['CARGA_COMORBILIDADES'].std():.2f})")
        
        return df
    
    def aggregate_cardiovascular_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea índice de riesgo cardiovascular.
        
        Args:
            df: DataFrame con variables clínicas
            
        Returns:
            pd.DataFrame: DataFrame con nueva columna RIESGO_CARDIOVASCULAR
        """
        cv_cols = [
            'HIPERTENSION', 'ENFERMEDAD_CORONARIA',
            'FALLA_CARDIACA', 'ENF_CEREBROVASCULAR'
        ]
        
        available_cols = [col for col in cv_cols if col in df.columns]
        
        df['RIESGO_CARDIOVASCULAR'] = df[available_cols].sum(axis=1)
        
        self.feature_mapping['RIESGO_CARDIOVASCULAR'] = available_cols
        self.aggregated_features.append('RIESGO_CARDIOVASCULAR')
        
        if self.verbose:
            print(f"✓ RIESGO_CARDIOVASCULAR creado (suma de {len(available_cols)} condiciones)")
            print(f"  Rango: {df['RIESGO_CARDIOVASCULAR'].min():.0f} - {df['RIESGO_CARDIOVASCULAR'].max():.0f}")
            
        return df
    
    def aggregate_respiratory_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea índice de riesgo respiratorio.
        
        Args:
            df: DataFrame con variables clínicas
            
        Returns:
            pd.DataFrame: DataFrame con nueva columna RIESGO_RESPIRATORIO
        """
        resp_cols = ['EPOC', 'ASMA', 'HIPERTENSION_PULMONAR']
        
        available_cols = [col for col in resp_cols if col in df.columns]
        
        df['RIESGO_RESPIRATORIO'] = df[available_cols].sum(axis=1)
        
        self.feature_mapping['RIESGO_RESPIRATORIO'] = available_cols
        self.aggregated_features.append('RIESGO_RESPIRATORIO')
        
        if self.verbose:
            print(f"✓ RIESGO_RESPIRATORIO creado (suma de {len(available_cols)} condiciones)")
            print(f"  Rango: {df['RIESGO_RESPIRATORIO'].min():.0f} - {df['RIESGO_RESPIRATORIO'].max():.0f}")
            
        return df
    
    def aggregate_metabolic_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea índice de riesgo metabólico.
        
        Args:
            df: DataFrame con variables clínicas
            
        Returns:
            pd.DataFrame: DataFrame con nueva columna RIESGO_METABOLICO
        """
        met_cols = ['DIABETES', 'OBESIDAD', 'HIPOTIROIDISMO']
        
        available_cols = [col for col in met_cols if col in df.columns]
        
        df['RIESGO_METABOLICO'] = df[available_cols].sum(axis=1)
        
        self.feature_mapping['RIESGO_METABOLICO'] = available_cols
        self.aggregated_features.append('RIESGO_METABOLICO')
        
        if self.verbose:
            print(f"✓ RIESGO_METABOLICO creado (suma de {len(available_cols)} condiciones)")
            print(f"  Rango: {df['RIESGO_METABOLICO'].min():.0f} - {df['RIESGO_METABOLICO'].max():.0f}")
            
        return df
    
    def aggregate_immunocompromised(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea índice de inmunocompromiso/cáncer.
        
        Args:
            df: DataFrame con variables clínicas
            
        Returns:
            pd.DataFrame: DataFrame con nueva columna INMUNOCOMPROMISO_CANCER
        """
        immune_cols = ['INMUNOSUPRESION', 'CANCER_ACTIVO']
        
        available_cols = [col for col in immune_cols if col in df.columns]
        
        df['INMUNOCOMPROMISO_CANCER'] = df[available_cols].sum(axis=1)
        
        self.feature_mapping['INMUNOCOMPROMISO_CANCER'] = available_cols
        self.aggregated_features.append('INMUNOCOMPROMISO_CANCER')
        
        if self.verbose:
            print(f"✓ INMUNOCOMPROMISO_CANCER creado (suma de {len(available_cols)} condiciones)")
            print(f"  Rango: {df['INMUNOCOMPROMISO_CANCER'].min():.0f} - {df['INMUNOCOMPROMISO_CANCER'].max():.0f}")
            
        return df
    
    def aggregate_bleeding_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea índice de factores de riesgo de sangrado.
        
        Args:
            df: DataFrame con variables clínicas
            
        Returns:
            pd.DataFrame: DataFrame con nueva columna FACTORES_RIESGO_SANGRADO
        """
        bleeding_cols = ['SANGRADO_MAYOR', 'TABAQUISMO']
        
        available_cols = [col for col in bleeding_cols if col in df.columns]
        
        df['FACTORES_RIESGO_SANGRADO'] = df[available_cols].sum(axis=1)
        
        self.feature_mapping['FACTORES_RIESGO_SANGRADO'] = available_cols
        self.aggregated_features.append('FACTORES_RIESGO_SANGRADO')
        
        if self.verbose:
            print(f"✓ FACTORES_RIESGO_SANGRADO creado (suma de {len(available_cols)} factores)")
            print(f"  Rango: {df['FACTORES_RIESGO_SANGRADO'].min():.0f} - {df['FACTORES_RIESGO_SANGRADO'].max():.0f}")
            
        return df
    
    def categorize_age(self, df: pd.DataFrame, age_col: str = 'EDAD') -> pd.DataFrame:
        """
        Categoriza edad en grupos de riesgo quirúrgico según estándares médicos.
        
        Categorías basadas en ASA (American Society of Anesthesiologists):
        0: Pediátrico (<18 años) - Mayor vulnerabilidad fisiológica
        1: Adulto joven (18-44 años) - Bajo riesgo quirúrgico
        2: Adulto medio (45-64 años) - Riesgo moderado
        3: Adulto mayor (65-74 años) - Riesgo aumentado
        4: Anciano (≥75 años) - Alto riesgo quirúrgico
        
        Referencia: ASA Physical Status Classification System & 
                   WHO Age Categories for Surgical Risk
        
        Args:
            df: DataFrame con variable de edad
            age_col: Nombre de la columna de edad
            
        Returns:
            pd.DataFrame: DataFrame con nueva columna CATEGORIA_EDAD
        """
        if age_col not in df.columns:
            if self.verbose:
                print(f"⚠ Advertencia: columna '{age_col}' no encontrada")
            return df
        
        def categorize(age):
            if pd.isna(age):
                return np.nan
            if age < 18:
                return 0  # Pediátrico
            elif age < 45:
                return 1  # Adulto joven
            elif age < 65:
                return 2  # Adulto medio
            elif age < 75:
                return 3  # Adulto mayor
            else:
                return 4  # Anciano
        
        df['CATEGORIA_EDAD'] = df[age_col].apply(categorize)
        
        self.feature_mapping['CATEGORIA_EDAD'] = [age_col]
        self.aggregated_features.append('CATEGORIA_EDAD')
        
        if self.verbose:
            print(f"✓ CATEGORIA_EDAD creada desde '{age_col}'")
            print(f"  Distribución:")
            counts = df['CATEGORIA_EDAD'].value_counts().sort_index()
            labels = {
                0: 'Pediátrico (<18)', 
                1: 'Adulto joven (18-44)', 
                2: 'Adulto medio (45-64)', 
                3: 'Adulto mayor (65-74)', 
                4: 'Anciano (≥75)'
            }
            for cat, count in counts.items():
                if not pd.isna(cat):
                    pct = count / len(df) * 100
                    print(f"    {labels[int(cat)]}: {count} ({pct:.1f}%)")
        
        return df
    
    def categorize_hemoglobin(
        self, 
        df: pd.DataFrame, 
        hb_col: str = 'HB_PREQX',
        gender_col: str = 'GENERO'
    ) -> pd.DataFrame:
        """
        Categoriza hemoglobina prequirúrgica según niveles de anemia OMS.
        
        Categorías:
        0: Normal (≥12 g/dL)
        1: Anemia leve (10-11.9 g/dL)
        2: Anemia moderada (8-9.9 g/dL)
        3: Anemia severa (<8 g/dL)
        
        Args:
            df: DataFrame con hemoglobina
            hb_col: Nombre de columna de hemoglobina
            gender_col: Nombre de columna de género (0=F, 1=M)
            
        Returns:
            pd.DataFrame: DataFrame con nueva columna CATEGORIA_HB_PREQX
        """
        if hb_col not in df.columns:
            if self.verbose:
                print(f"⚠ Advertencia: columna '{hb_col}' no encontrada")
            return df

        def categorize(row):
            hb = row[hb_col]
            if pd.isna(hb):
                return np.nan

            # Umbrales simplificados (idealmente ajustar por sexo)
            if hb >= 12:
                return 0  # Normal
            elif hb >= 10:
                return 1  # Anemia leve
            elif hb >= 8:
                return 2  # Anemia moderada
            else:
                return 3  # Anemia severa

        # Categoría ordinal de hemoglobina
        df['CATEGORIA_HB_PREQX'] = df.apply(categorize, axis=1)

        # Variable binaria clínicamente útil: ANEMIA_PREOP (HB < 10 g/dL)
        # 1 = anemia preoperatoria (HB < 10), 0 = no
        df['ANEMIA_PREOP'] = df[hb_col].apply(lambda x: 1 if pd.notna(x) and x < 10 else 0)

        # Registrar en el mapeo de features
        self.feature_mapping['CATEGORIA_HB_PREQX'] = [hb_col]
        self.aggregated_features.append('CATEGORIA_HB_PREQX')
        self.feature_mapping['ANEMIA_PREOP'] = [hb_col]
        self.aggregated_features.append('ANEMIA_PREOP')

        if self.verbose:
            print(f"✓ CATEGORIA_HB_PREQX creada desde '{hb_col}'")
            print(f"  Distribución:")
            counts = df['CATEGORIA_HB_PREQX'].value_counts().sort_index()
            labels = {0: 'Normal (≥12)', 1: 'Anemia leve (10-11.9)', 
                     2: 'Anemia moderada (8-9.9)', 3: 'Anemia severa (<8)'}
            for cat, count in counts.items():
                if not pd.isna(cat):
                    pct = count / len(df) * 100
                    print(f"    {labels[cat]}: {count} ({pct:.1f}%)")

            # Resumen de ANEMIA_PREOP
            an_count = int(df['ANEMIA_PREOP'].sum())
            print(f"\n✓ ANEMIA_PREOP creada (HB < 10 g/dL): {an_count} pacientes ({an_count/len(df)*100:.1f}%)")

        return df
    
    def create_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea variables de interacción entre factores de riesgo.
        
        Args:
            df: DataFrame con variables clínicas
            
        Returns:
            pd.DataFrame: DataFrame con variables de interacción
        """
        interactions_created = []
        
        # Interacción: EDAD × CARGA_COMORBILIDADES
        if 'EDAD' in df.columns and 'CARGA_COMORBILIDADES' in df.columns:
            df['EDAD_X_CARGA_COMORBILIDADES'] = df['EDAD'] * df['CARGA_COMORBILIDADES']
            interactions_created.append('EDAD_X_CARGA_COMORBILIDADES')
            self.aggregated_features.append('EDAD_X_CARGA_COMORBILIDADES')
            
            if self.verbose:
                print(f"✓ EDAD_X_CARGA_COMORBILIDADES creada")
                print(f"  Rango: {df['EDAD_X_CARGA_COMORBILIDADES'].min():.0f} - {df['EDAD_X_CARGA_COMORBILIDADES'].max():.0f}")
        
        # Interacción: HB_PREQX × CARGA_COMORBILIDADES (hemoglobina baja + comorbilidades)
        if 'HB_PREQX' in df.columns and 'CARGA_COMORBILIDADES' in df.columns:
            # Invertir HB para que valores bajos den score alto
            df['HB_INVERSA_X_COMORBILIDADES'] = (20 - df['HB_PREQX']) * df['CARGA_COMORBILIDADES']
            interactions_created.append('HB_INVERSA_X_COMORBILIDADES')
            self.aggregated_features.append('HB_INVERSA_X_COMORBILIDADES')
            
            if self.verbose:
                print(f"✓ HB_INVERSA_X_COMORBILIDADES creada")
        
        # Interacción: SANGRADO_MAYOR × RIESGO_CARDIOVASCULAR
        if 'SANGRADO_MAYOR' in df.columns and 'RIESGO_CARDIOVASCULAR' in df.columns:
            df['SANGRADO_X_CV'] = df['SANGRADO_MAYOR'] * (df['RIESGO_CARDIOVASCULAR'] + 1)
            interactions_created.append('SANGRADO_X_CV')
            self.aggregated_features.append('SANGRADO_X_CV')
            
            if self.verbose:
                print(f"✓ SANGRADO_X_CV creada")
        
        if self.verbose and len(interactions_created) > 0:
            print(f"\nTotal de interacciones creadas: {len(interactions_created)}")
        
        return df
    
    def create_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea score de riesgo ponderado basado en variables agregadas.
        
        Fórmula empírica ponderada por relevancia clínica:
        RISK_SCORE = (CARGA_COMORBILIDADES × 1.0) + 
                     (RIESGO_CARDIOVASCULAR × 2.0) + 
                     (FACTORES_RIESGO_SANGRADO × 3.0) +
                     (CATEGORIA_EDAD × 0.5)
        
        Nota: CATEGORIA_EDAD ahora tiene rango 0-4 (antes 0-3) tras 
              incluir categoría pediátrica.
        
        Args:
            df: DataFrame con variables agregadas
            
        Returns:
            pd.DataFrame: DataFrame con RISK_SCORE
        """
        # Requerimos las columnas utilizadas en la nueva fórmula
        required_cols = [
            'CARGA_COMORBILIDADES', 'RIESGO_CARDIOVASCULAR',
            'TABAQUISMO', 'SANGRADO_MAYOR', 'CATEGORIA_EDAD', 'ANEMIA_PREOP'
        ]

        if all(col in df.columns for col in required_cols):
            # Nueva fórmula solicitada:
            # 1.0 * CARGA_COMORBILIDADES +
            # 1.5 * RIESGO_CARDIOVASCULAR +
            # 1.0 * TABAQUISMO +
            # 3.0 * SANGRADO_MAYOR +
            # 1.0 * CATEGORIA_EDAD +
            # 3.0 * ANEMIA_PREOP
            df['RISK_SCORE'] = (
                1.0 * df['CARGA_COMORBILIDADES'] +
                1.5 * df['RIESGO_CARDIOVASCULAR'] +
                1.0 * df['TABAQUISMO'] +
                3.0 * df['SANGRADO_MAYOR'] +
                1.0 * df['CATEGORIA_EDAD'] +
                3.0 * df['ANEMIA_PREOP']
            )
            
            self.aggregated_features.append('RISK_SCORE')
            
            if self.verbose:
                print(f"\n✓ RISK_SCORE creado (score ponderado)")
                print(f"  Media: {df['RISK_SCORE'].mean():.2f} (SD: {df['RISK_SCORE'].std():.2f})")
                print(f"  Rango: {df['RISK_SCORE'].min():.2f} - {df['RISK_SCORE'].max():.2f}")
        else:
            if self.verbose:
                missing = [col for col in required_cols if col not in df.columns]
                print(f"⚠ No se pudo crear RISK_SCORE. Faltan: {missing}")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, create_all: bool = True) -> pd.DataFrame:
        """
        Aplica todas las agregaciones de features al DataFrame.
        
        Args:
            df: DataFrame original con variables clínicas
            create_all: Si True, crea todas las variables agregadas
            
        Returns:
            pd.DataFrame: DataFrame con variables originales + agregadas
        """
        if self.verbose:
            print("\n" + "="*70)
            print("AGREGACIÓN DE FEATURES - INICIANDO")
            print("="*70)
            print(f"Dataset original: {df.shape[0]} filas × {df.shape[1]} columnas")
            print()
        
        df_agg = df.copy()
        
        # 1. Agregaciones de comorbilidades
        if self.verbose:
            print("\n1. AGREGACIONES DE COMORBILIDADES")
            print("-" * 70)
        
        df_agg = self.aggregate_comorbidities(df_agg)
        df_agg = self.aggregate_cardiovascular_risk(df_agg)
        df_agg = self.aggregate_respiratory_risk(df_agg)
        df_agg = self.aggregate_metabolic_risk(df_agg)
        df_agg = self.aggregate_immunocompromised(df_agg)
        df_agg = self.aggregate_bleeding_risk(df_agg)
        
        # 2. Categorizaciones
        if self.verbose:
            print("\n2. CATEGORIZACIONES DE VARIABLES CONTINUAS")
            print("-" * 70)
        
        df_agg = self.categorize_age(df_agg)
        df_agg = self.categorize_hemoglobin(df_agg)
        
        # 3. Interacciones
        if self.verbose:
            print("\n3. VARIABLES DE INTERACCIÓN")
            print("-" * 70)
        
        df_agg = self.create_interactions(df_agg)
        
        # 4. Score de riesgo compuesto
        if create_all:
            if self.verbose:
                print("\n4. SCORE DE RIESGO COMPUESTO")
                print("-" * 70)
            
            df_agg = self.create_risk_score(df_agg)
        
        # Resumen final
        if self.verbose:
            print("\n" + "="*70)
            print("AGREGACIÓN DE FEATURES - COMPLETADA")
            print("="*70)
            print(f"Dataset resultante: {df_agg.shape[0]} filas × {df_agg.shape[1]} columnas")
            print(f"Nuevas features creadas: {len(self.aggregated_features)}")
            print(f"\nFeatures agregadas:")
            for feat in self.aggregated_features:
                print(f"  - {feat}")
            print("="*70)
        
        return df_agg
    
    def get_feature_summary(self) -> pd.DataFrame:
        """
        Obtiene resumen de features agregadas y sus componentes.
        
        Returns:
            pd.DataFrame: Resumen de mapeo de features
        """
        summary_data = []
        
        for agg_feat, original_feats in self.feature_mapping.items():
            summary_data.append({
                'Aggregated_Feature': agg_feat,
                'N_Components': len(original_feats),
                'Original_Features': ', '.join(original_feats)
            })
        
        return pd.DataFrame(summary_data)


def main():
    """Función de ejemplo para uso del agregador."""
    # Cargar datos
    df = pd.read_csv("data/processed/shock_cleaned.csv")
    
    # Crear agregador
    aggregator = FeatureAggregator(verbose=True)
    
    # Aplicar agregaciones
    df_aggregated = aggregator.fit_transform(df)
    
    # Guardar dataset con features agregadas
    output_path = "data/processed/shock_aggregated.csv"
    df_aggregated.to_csv(output_path, index=False)
    print(f"\n✓ Dataset guardado en: {output_path}")
    
    # Guardar resumen de features
    summary = aggregator.get_feature_summary()
    summary_path = "data/processed/feature_aggregation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"✓ Resumen de features guardado en: {summary_path}")


if __name__ == "__main__":
    main()
