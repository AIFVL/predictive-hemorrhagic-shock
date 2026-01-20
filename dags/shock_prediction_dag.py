"""
DAG de Airflow para el pipeline de prediccion de shock hemorragico.
Llama directamente a los metodos de los modulos de Python.
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys
import yaml

from airflow import DAG
from airflow.operators.python import PythonOperator

# Asegurar que src esta en el path
BASE_DIR = Path('/opt/airflow')
sys.path.insert(0, str(BASE_DIR))

# Cargar configuracion
CONFIG_FILE = BASE_DIR / 'src' / 'config' / 'pipeline_config.yaml'
with open(CONFIG_FILE, 'r') as f:
    CONFIG = yaml.safe_load(f)

VERSION = CONFIG['version']
SEED = CONFIG['random_seed']

def build_path(template: str) -> Path:
    return BASE_DIR / template.format(version=VERSION)

PATHS = {k: build_path(v) for k, v in CONFIG['paths'].items()}

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 18),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# ==============================================================================
# TASK FUNCTIONS
# ==============================================================================

def step1_validate_raw_data(**kwargs):
    """Step 1/7: Validate Raw Data"""
    import json
    from src.data.load import load_raw_data
    from src.data.validate import validate_data
    
    print("=" * 70)
    print("Step 1/7: Validating raw data")
    print(f"Input: {PATHS['raw_data']}")
    print("=" * 70)
    
    output_dir = PATHS['evaluation_output'].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_raw_data(PATHS['raw_data'])
    df_validated, report = validate_data(df, str(PATHS['feature_config']))
    
    report_path = output_dir / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Validation report saved to: {report_path}")
    print("Step 1/7: COMPLETED")
    return report


def step2_clean_data(**kwargs):
    """Step 2/7: Clean Data"""
    import json
    from src.data.load import load_raw_data
    from src.data.validate import validate_data
    from src.data.clean import clean_data, save_clean_data
    
    print("=" * 70)
    print("Step 2/7: Cleaning data")
    print(f"Input: {PATHS['raw_data']}")
    print(f"Output: {PATHS['cleaned_data']}")
    print("=" * 70)
    
    PATHS['cleaned_data'].parent.mkdir(parents=True, exist_ok=True)
    
    df = load_raw_data(PATHS['raw_data'])
    df_validated, _ = validate_data(df, str(PATHS['feature_config']))
    df_cleaned, report = clean_data(df_validated)
    
    save_clean_data(
        df_cleaned,
        PATHS['cleaned_data'],
        format=CONFIG['cleaning']['output_format']
    )
    
    report_path = PATHS['evaluation_output'].parent / 'cleaning_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Cleaned data saved to: {PATHS['cleaned_data']}")
    print(f"Cleaning report saved to: {report_path}")
    print("Step 2/7: COMPLETED")
    return str(PATHS['cleaned_data'])


def step3_create_training_dataset(**kwargs):
    """Step 3/7: Create Training Dataset with Feature Engineering"""
    import pandas as pd
    from src.datasets.make_dataset import (
        make_training_dataset,
        save_training_dataset,
        create_train_test_split,
        save_splits
    )
    
    print("=" * 70)
    print("Step 3/7: Creating training dataset with feature engineering")
    print(f"Input: {PATHS['cleaned_data']}")
    print(f"Output: {PATHS['training_data']}")
    print("=" * 70)
    
    PATHS['training_data'].parent.mkdir(parents=True, exist_ok=True)
    PATHS['splits_dir'].mkdir(parents=True, exist_ok=True)
    
    df = pd.read_parquet(PATHS['cleaned_data'])
    
    X, y = make_training_dataset(df, config_path=str(PATHS['feature_config']))
    
    save_training_dataset(
        X, y,
        PATHS['training_data'],
        format=CONFIG['cleaning']['output_format']
    )
    
    X_train, X_test, y_train, y_test = create_train_test_split(
        X, y,
        test_size=CONFIG['data_split']['test_size'],
        random_state=SEED
    )
    
    save_splits(
        X_train, X_test, y_train, y_test,
        PATHS['splits_dir'],
        version=VERSION,
        format=CONFIG['cleaning']['output_format']
    )
    
    print(f"Training dataset saved to: {PATHS['training_data']}")
    print(f"Splits saved to: {PATHS['splits_dir']}/{VERSION}/")
    print("Step 3/7: COMPLETED")
    return str(PATHS['training_data'])


def step4_generate_eda_plots(**kwargs):
    """Step 4/7: Generate EDA Plots"""
    import pandas as pd
    from src.visualization.generate_eda_plots import generate_all_eda_plots
    
    print("=" * 70)
    print("Step 4/7: Generating EDA plots")
    print(f"Input: {PATHS['cleaned_data']}")
    print(f"Output: {PATHS['eda_plots']}")
    print("=" * 70)
    
    PATHS['eda_plots'].mkdir(parents=True, exist_ok=True)
    
    df = pd.read_parquet(PATHS['cleaned_data'])
    
    NUMERICAL_FEATURES = ['EDAD', 'HB_PREQX']
    BINARY_FEATURES = [
        'GENERO', 'ACT_FISICA_METS', 'HIPERTENSION', 'DIABETES',
        'ENFERMEDAD_CORONARIA', 'FALLA_CARDIACA', 'HIPOTIROIDISMO', 'ERC',
        'INMUNOSUPRESION', 'OBESIDAD', 'HIPERTENSION_PULMONAR', 'EPOC',
        'ASMA', 'ENF_CEREBROVASCULAR', 'CANCER_ACTIVO', 'TABAQUISMO', 'SANGRADO_MAYOR'
    ]
    
    numerical_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    binary_cols = [c for c in BINARY_FEATURES if c in df.columns]
    
    plots = generate_all_eda_plots(
        df, numerical_cols, binary_cols, 'SHOCK', str(PATHS['eda_plots'])
    )
    
    print(f"EDA plots saved to: {PATHS['eda_plots']}")
    print(f"Generated {len(plots)} plots")
    print("Step 4/7: COMPLETED")


def step5_train_model(**kwargs):
    """Step 5/7: Train Model with Cross-Validation"""
    import pandas as pd
    from src.models.train import cross_validate_model, train_model, save_model
    
    print("=" * 70)
    print("Step 5/7: Training model with cross-validation")
    print(f"Input: {PATHS['splits_dir']}/{VERSION}/train.parquet")
    print(f"Output: {PATHS['model_output']}")
    print("=" * 70)
    
    PATHS['model_output'].parent.mkdir(parents=True, exist_ok=True)
    
    train_split = PATHS['splits_dir'] / VERSION / 'train.parquet'
    df = pd.read_parquet(train_split)
    
    y = df['SHOCK']
    X = df.drop(columns=['SHOCK'])
    
    print(f"Loaded {len(df)} samples with {X.shape[1]} features")
    
    cv_results, _ = cross_validate_model(
        X, y,
        model_name=CONFIG['model']['algorithm'],
        n_folds=CONFIG['model']['cross_validation']['n_folds'],
        scale_features=True,
        random_state=SEED
    )
    
    pipeline, metadata = train_model(
        X, y,
        model_name=CONFIG['model']['algorithm'],
        scale_features=True
    )
    
    metadata['cv_results'] = cv_results
    save_model(pipeline, str(PATHS['model_output']), metadata)
    
    print(f"Model saved to: {PATHS['model_output']}")
    print("Step 5/7: COMPLETED")


def step6_evaluate_model(**kwargs):
    """Step 6/7: Evaluate Model on Test Set"""
    import pandas as pd
    import json
    from src.models.train import load_model
    from src.models.evaluate import evaluate_model
    
    print("=" * 70)
    print("Step 6/7: Evaluating model on test set")
    print(f"Model: {PATHS['model_output']}")
    print(f"Data: {PATHS['splits_dir']}/{VERSION}/test.parquet")
    print(f"Output: {PATHS['evaluation_output']}")
    print("=" * 70)
    
    test_split = PATHS['splits_dir'] / VERSION / 'test.parquet'
    df_test = pd.read_parquet(test_split)
    
    y_test = df_test['SHOCK']
    X_test = df_test.drop(columns=['SHOCK'])
    
    pipeline, metadata = load_model(str(PATHS['model_output']))
    
    results = evaluate_model(
        pipeline, X_test, y_test,
        dataset_name='test',
        use_optimal_threshold=True
    )
    
    # Add metadata to results
    results['model_metadata'] = metadata
    
    PATHS['evaluation_output'].parent.mkdir(parents=True, exist_ok=True)
    with open(PATHS['evaluation_output'], 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Evaluation results saved to: {PATHS['evaluation_output']}")
    print("Step 6/7: COMPLETED")


def step7_generate_evaluation_plots(**kwargs):
    """Step 7/7: Generate Evaluation Plots"""
    import pandas as pd
    from src.models.train import load_model
    from src.visualization.generate_plots import generate_all_plots
    
    print("=" * 70)
    print("Step 7/7: Generating evaluation plots")
    print(f"Model: {PATHS['model_output']}")
    print(f"Data: {PATHS['splits_dir']}/{VERSION}/test.parquet")
    print(f"Output: {PATHS['eval_plots']}")
    print("=" * 70)
    
    PATHS['eval_plots'].mkdir(parents=True, exist_ok=True)
    
    test_split = PATHS['splits_dir'] / VERSION / 'test.parquet'
    df_test = pd.read_parquet(test_split)
    
    y_test = df_test['SHOCK']
    X_test = df_test.drop(columns=['SHOCK'])
    
    pipeline, _ = load_model(str(PATHS['model_output']))
    
    plots = generate_all_plots(
        pipeline, X_test, y_test,
        str(PATHS['eval_plots']),
        model_name='Random Forest'
    )
    
    print(f"Evaluation plots saved to: {PATHS['eval_plots']}")
    print(f"Generated {len(plots)} plots")
    print("Step 7/7: COMPLETED")


# ==============================================================================
# DAG DEFINITION
# ==============================================================================
with DAG(
    dag_id='shock_prediction_pipeline',
    default_args=default_args,
    description='Pipeline de prediccion de shock hemorragico',
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'shock', 'prediction']
) as dag:
    
    t1_validate = PythonOperator(
        task_id='validate_raw_data',
        python_callable=step1_validate_raw_data
    )
    
    t2_clean = PythonOperator(
        task_id='clean_data',
        python_callable=step2_clean_data
    )
    
    t3_create_dataset = PythonOperator(
        task_id='create_training_dataset',
        python_callable=step3_create_training_dataset
    )
    
    t4_eda = PythonOperator(
        task_id='generate_eda_plots',
        python_callable=step4_generate_eda_plots
    )
    
    t5_train = PythonOperator(
        task_id='train_model',
        python_callable=step5_train_model
    )
    
    t6_evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=step6_evaluate_model
    )
    
    t7_plots = PythonOperator(
        task_id='generate_evaluation_plots',
        python_callable=step7_generate_evaluation_plots
    )
    
    # Dependencies
    t1_validate >> t2_clean >> t3_create_dataset
    t3_create_dataset >> t4_eda
    t3_create_dataset >> t5_train >> t6_evaluate >> t7_plots