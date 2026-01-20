"""
Feature aggregations module.

This module implements the clinically-validated aggregated features
as documented in docs/project_specification.md Section 5.1.

Each aggregation function includes a docstring with clinical justification.
"""

import pandas as pd
from typing import List, Dict, Optional
import yaml
from pathlib import Path


def load_aggregation_config(config_path: Optional[str] = None) -> Dict:
    """
    Load aggregation configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Dict with aggregation definitions
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Try both 'aggregated_features' (new format) and 'aggregations' (old format)
            return config.get('aggregated_features', config.get('aggregations', {}))
    
    # Default configuration based on project specification
    return get_default_aggregation_config()


def get_default_aggregation_config() -> Dict:
    """
    Return default aggregation configuration.
    
    These are the clinically-validated aggregations from the project specification.
    """
    return {
        'CARGA_COMORBILIDADES': {
            'description': 'Total burden of chronic comorbidities (Charlson Index concept)',
            'components': [
                'HIPERTENSION', 'DIABETES', 'ENFERMEDAD_CORONARIA',
                'FALLA_CARDIACA', 'HIPOTIROIDISMO', 'ERC',
                'INMUNOSUPRESION', 'OBESIDAD', 'HIPERTENSION_PULMONAR',
                'EPOC', 'ASMA', 'ENF_CEREBROVASCULAR', 'CANCER_ACTIVO'
            ],
            'operation': 'sum',
            'range': [0, 13]
        },
        'RIESGO_CARDIOVASCULAR': {
            'description': 'Cardiovascular risk index - compromised hemodynamic reserve',
            'components': [
                'HIPERTENSION', 'ENFERMEDAD_CORONARIA', 
                'FALLA_CARDIACA', 'ENF_CEREBROVASCULAR'
            ],
            'operation': 'sum',
            'range': [0, 4]
        },
        'RIESGO_RESPIRATORIO': {
            'description': 'Respiratory risk index - limited oxygenation capacity',
            'components': [
                'EPOC', 'ASMA', 'HIPERTENSION_PULMONAR'
            ],
            'operation': 'sum',
            'range': [0, 3]
        },
        'RIESGO_METABOLICO': {
            'description': 'Metabolic risk index - endothelial dysfunction',
            'components': [
                'DIABETES', 'OBESIDAD', 'HIPOTIROIDISMO'
            ],
            'operation': 'sum',
            'range': [0, 3]
        }
    }


def create_carga_comorbilidades(df: pd.DataFrame, config: Dict = None) -> pd.Series:
    """
    Create CARGA_COMORBILIDADES feature.
    
    CLINICAL JUSTIFICATION:
    The Charlson Comorbidity Index (Charlson et al., 1987) and ASA Physical 
    Status demonstrate that accumulated chronic conditions predict perioperative 
    complications better than isolated comorbidities. This variable captures
    systemic frailty by counting the total number of comorbidities present.
    
    Van Walraven et al. (2009) validated that simple comorbidity counts have
    predictive power comparable to weighted indices.
    
    COMPONENTS (13 conditions):
    - HIPERTENSION: Arterial hypertension
    - DIABETES: Diabetes mellitus
    - ENFERMEDAD_CORONARIA: Coronary artery disease
    - FALLA_CARDIACA: Heart failure
    - HIPOTIROIDISMO: Hypothyroidism
    - ERC: Chronic kidney disease
    - INMUNOSUPRESION: Immunosuppression
    - OBESIDAD: Obesity (BMI >= 30)
    - HIPERTENSION_PULMONAR: Pulmonary hypertension
    - EPOC: Chronic obstructive pulmonary disease
    - ASMA: Bronchial asthma
    - ENF_CEREBROVASCULAR: Cerebrovascular disease
    - CANCER_ACTIVO: Active cancer
    
    Args:
        df: DataFrame with binary comorbidity columns
        config: Optional configuration dict
    
    Returns:
        Series with comorbidity count (range 0-13)
    """
    if config is None:
        config = get_default_aggregation_config()
    
    components = config['CARGA_COMORBILIDADES']['components']
    available = [c for c in components if c in df.columns]
    
    return df[available].sum(axis=1).astype(int)


def create_riesgo_cardiovascular(df: pd.DataFrame, config: Dict = None) -> pd.Series:
    """
    Create RIESGO_CARDIOVASCULAR feature.
    
    CLINICAL JUSTIFICATION:
    The cardiovascular system is directly responsible for hemodynamic compensation
    during blood loss (Gutierrez et al., 2004). Pre-existing cardiovascular 
    dysfunction eliminates the compensatory reserve needed to maintain cardiac
    output during hypovolemia.
    
    - Chronic hypertension causes vascular stiffness and diastolic dysfunction
    - Coronary artery disease reduces myocardial contractility
    - Heart failure eliminates cardiac reserve
    - Cerebrovascular disease indicates systemic vascular dysfunction
    
    European Society of Intensive Care Medicine (2014) guidelines identify 
    prior cardiovascular dysfunction as an independent predictor of 
    hemorrhagic decompensation.
    
    COMPONENTS (4 conditions):
    - HIPERTENSION
    - ENFERMEDAD_CORONARIA
    - FALLA_CARDIACA
    - ENF_CEREBROVASCULAR
    
    Args:
        df: DataFrame with cardiovascular comorbidity columns
        config: Optional configuration dict
    
    Returns:
        Series with cardiovascular risk count (range 0-4)
    """
    if config is None:
        config = get_default_aggregation_config()
    
    components = config['RIESGO_CARDIOVASCULAR']['components']
    available = [c for c in components if c in df.columns]
    
    return df[available].sum(axis=1).astype(int)


def create_riesgo_respiratorio(df: pd.DataFrame, config: Dict = None) -> pd.Series:
    """
    Create RIESGO_RESPIRATORIO feature.
    
    CLINICAL JUSTIFICATION:
    During hemorrhagic shock, tissue oxygenation critically depends on 
    respiratory capacity to compensate for metabolic acidosis. COPD, asthma,
    and pulmonary hypertension limit the ability to increase minute ventilation
    during hypoxemia.
    
    National Trauma Data Bank (NTDB, 2020) data show that patients with 
    chronic respiratory disease have higher mortality in hemorrhagic shock
    due to limited hypoxic compensation.
    
    COMPONENTS (3 conditions):
    - EPOC: Chronic obstructive pulmonary disease
    - ASMA: Bronchial asthma
    - HIPERTENSION_PULMONAR: Pulmonary hypertension
    
    Args:
        df: DataFrame with respiratory comorbidity columns
        config: Optional configuration dict
    
    Returns:
        Series with respiratory risk count (range 0-3)
    """
    if config is None:
        config = get_default_aggregation_config()
    
    components = config['RIESGO_RESPIRATORIO']['components']
    available = [c for c in components if c in df.columns]
    
    return df[available].sum(axis=1).astype(int)


def create_riesgo_metabolico(df: pd.DataFrame, config: Dict = None) -> pd.Series:
    """
    Create RIESGO_METABOLICO feature.
    
    CLINICAL JUSTIFICATION:
    These three metabolic disorders share mechanisms of endothelial dysfunction
    and altered vascular homeostasis:
    
    - DIABETES: Associated with endothelial dysfunction, coagulation cascade
      alterations, and autonomic neuropathy affecting sympathetic response
      (Preiser et al., 2009)
    
    - OBESITY: Increases volume of distribution, complicates vascular access,
      associated with chronic proinflammatory state and fibrinolysis 
      alterations (Allman-Farinelli, 2011)
    
    - HYPOTHYROIDISM: Reduces cardiac contractility, diminishes response
      to catecholamines, alters coagulation factor metabolism
      (Squizzato et al., 2007)
    
    COMPONENTS (3 conditions):
    - DIABETES
    - OBESIDAD
    - HIPOTIROIDISMO
    
    Args:
        df: DataFrame with metabolic comorbidity columns
        config: Optional configuration dict
    
    Returns:
        Series with metabolic risk count (range 0-3)
    """
    if config is None:
        config = get_default_aggregation_config()
    
    components = config['RIESGO_METABOLICO']['components']
    available = [c for c in components if c in df.columns]
    
    return df[available].sum(axis=1).astype(int)


def create_all_aggregations(
    df: pd.DataFrame,
    config_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create all aggregated features at once.
    
    Args:
        df: DataFrame with original features
        config_path: Optional path to YAML configuration
    
    Returns:
        DataFrame with only the new aggregated columns
    """
    config = load_aggregation_config(config_path)
    
    aggregations = pd.DataFrame(index=df.index)
    
    aggregations['CARGA_COMORBILIDADES'] = create_carga_comorbilidades(df, config)
    aggregations['RIESGO_CARDIOVASCULAR'] = create_riesgo_cardiovascular(df, config)
    aggregations['RIESGO_RESPIRATORIO'] = create_riesgo_respiratorio(df, config)
    aggregations['RIESGO_METABOLICO'] = create_riesgo_metabolico(df, config)
    
    print(f"Created {len(aggregations.columns)} aggregated features:")
    for col in aggregations.columns:
        print(f"  - {col}: range [{aggregations[col].min()}, {aggregations[col].max()}]")
    
    return aggregations


if __name__ == "__main__":
    """Test aggregation creation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create aggregated features")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input data path")
    parser.add_argument("--config", "-c", type=str, default=None, help="Config YAML path")
    
    args = parser.parse_args()
    
    # Load data
    if args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    
    # Create aggregations
    agg_df = create_all_aggregations(df, args.config)
    
    print(f"\nAggregations shape: {agg_df.shape}")
    print(f"\nStatistics:\n{agg_df.describe()}")
