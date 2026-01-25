"""
DAG de Airflow para el pipeline de prediccion de shock hemorragico.
Llama directamente a los metodos de los modulos de Python.
Entrena y evalua multiples modelos configurados en pipeline_config.yaml.
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import TaskInstance

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
    from src.utils import get_config, logger, log_section, DataWriter
    from src.data.load import load_raw_data
    from src.data.validate import validate_data
    
    config = get_config()
    log_section("STEP 1/7: VALIDATING RAW DATA")
    
    df = load_raw_data()
    df_validated, report = validate_data(df)
    
    # Save validation report
    report_path = Path(config.get_path('output_base')) / 'validation_report.json'
    DataWriter.write_json_file(report, str(report_path))
    
    logger.success("Step 1/7: COMPLETED")
    return report


def step2_clean_data(**kwargs):
    """Step 2/7: Clean Data"""
    from src.utils import get_config, logger, log_section, DataWriter
    from src.data.load import load_raw_data
    from src.data.validate import validate_data
    from src.data.clean import clean_data, save_clean_data
    
    config = get_config()
    log_section("STEP 2/7: CLEANING DATA")
    
    df = load_raw_data()
    df_validated, _ = validate_data(df)
    df_cleaned, report = clean_data(df_validated)
    
    save_clean_data(df_cleaned, format='parquet')
    
    # Save cleaning report
    report_path = Path(config.get_path('output_base')) / 'cleaning_report.json'
    DataWriter.write_json_file(report, str(report_path))
    
    logger.success("Step 2/7: COMPLETED")
    return True


def step3_create_training_dataset(**kwargs):
    """Step 3/7: Create Training Dataset with Feature Engineering"""
    from src.utils import get_config, logger, log_section
    from src.data.load import load_processed_data
    from src.datasets.make_dataset import (
        make_training_dataset,
        save_training_dataset,
        create_train_test_split,
        save_splits
    )
    
    config = get_config()
    log_section("STEP 3/7: CREATING TRAINING DATASET WITH FEATURE ENGINEERING")
    
    df = load_processed_data()
    
    X, y = make_training_dataset(df)
    
    save_training_dataset(X, y, format='parquet')
    
    X_train, X_test, y_train, y_test = create_train_test_split(
        X, y,
        test_size=config.get_data_split_config()['test_size'],
        random_state=config.get_random_seed()
    )
    
    save_splits(X_train, X_test, y_train, y_test, format='parquet')
    
    logger.success("Step 3/7: COMPLETED")
    return True


def step4_generate_eda_plots(**kwargs):
    """Step 4/7: Generate EDA Plots"""
    from src.utils import get_config, logger, log_section
    from src.data.load import load_processed_data
    from src.visualization.generate_eda_plots import generate_all_eda_plots
    from src.features.base_features import (
        get_numerical_features, 
        get_binary_features, 
        get_target_variable
    )
    
    config = get_config()
    log_section("STEP 4/7: GENERATING EDA PLOTS")
    
    df = load_processed_data()
    
    # Load feature lists from config
    numerical_features = get_numerical_features()
    binary_features = get_binary_features()
    target_variable = get_target_variable()
    
    # Filter to only columns present in the dataframe
    numerical_cols = [c for c in numerical_features if c in df.columns]
    binary_cols = [c for c in binary_features if c in df.columns]
    
    plots = generate_all_eda_plots(df, numerical_cols, binary_cols, target_variable)
    
    logger.info({"generated_plots": len(plots)})
    logger.success("Step 4/7: COMPLETED")


def step5_train_model(model_name: str, **kwargs):
    """Step 5/7: Train Model with Cross-Validation"""
    import pandas as pd
    from src.utils import get_config, logger, log_section
    from src.models.train import cross_validate_model, train_model, save_model
    from src.features.base_features import get_target_variable
    
    config = get_config()
    log_section(f"STEP 5/7: TRAINING MODEL - {model_name.upper()}")
    
    # Load training split
    train_split_path = Path(config.get_path('splits_dir')) / 'train.parquet'
    df = pd.read_parquet(train_split_path)
    
    # Get target variable from config
    target_variable = get_target_variable()
    
    y = df[target_variable]
    X = df.drop(columns=[target_variable])
    
    logger.info({"samples": len(df), "features": X.shape[1]})
    
    # Get model configuration
    model_config = config.get_model(model_name)
    cv_config = config.get_cv_config()
    random_state = config.get_random_seed()
    
    # Cross-validate
    cv_results = cross_validate_model(
        X, y,
        model_config=model_config,
        model_name=model_name,
        n_folds=cv_config.get('n_folds', 5),
        scale_features=cv_config.get('scale_features', True),
        random_state=random_state
    )
    
    # Train final model
    pipeline, metadata = train_model(
        X, y,
        model_config=model_config,
        model_name=model_name,
        scale_features=cv_config.get('scale_features', True)
    )
    
    metadata['cv_results'] = cv_results
    
    # Save model
    save_model(pipeline, model_name, metadata)
    
    logger.success(f"Step 5/7: COMPLETED for {model_name}")
    return model_name


def step6_evaluate_model(model_name: str, **kwargs):
    """Step 6/7: Evaluate Model on Test Set"""
    import pandas as pd
    from src.utils import get_config, logger, log_section
    from src.models.train import load_model
    from src.models.evaluate import evaluate_model, save_evaluation_results
    from src.features.base_features import get_target_variable
    
    config = get_config()
    log_section(f"STEP 6/7: EVALUATING MODEL - {model_name.upper()}")
    
    # Load test split
    test_split_path = Path(config.get_path('splits_dir')) / 'test.parquet'
    df_test = pd.read_parquet(test_split_path)
    
    # Get target variable from config
    target_variable = get_target_variable()
    
    y_test = df_test[target_variable]
    X_test = df_test.drop(columns=[target_variable])
    
    # Load model
    pipeline = load_model(model_name)
    
    # Evaluate
    results = evaluate_model(
        pipeline, X_test, y_test,
        dataset_name='test',
        use_optimal_threshold=True
    )
    
    # Add model name to results
    results['model_name'] = model_name
    
    # Save results
    save_evaluation_results(results, model_name)
    
    logger.success(f"Step 6/7: COMPLETED for {model_name}")


def step7_generate_evaluation_plots(model_name: str, **kwargs):
    """Step 7/7: Generate Evaluation Plots"""
    import pandas as pd
    from src.utils import get_config, logger, log_section
    from src.models.train import load_model
    from src.visualization.generate_plots import generate_all_plots
    from src.features.base_features import get_target_variable
    
    config = get_config()
    log_section(f"STEP 7/7: GENERATING EVALUATION PLOTS - {model_name.upper()}")
    
    # Load test split
    test_split_path = Path(config.get_path('splits_dir')) / 'test.parquet'
    df_test = pd.read_parquet(test_split_path)
    
    # Get target variable from config
    target_variable = get_target_variable()
    
    y_test = df_test[target_variable]
    X_test = df_test.drop(columns=[target_variable])
    
    # Load model
    pipeline = load_model(model_name)
    
    # Generate plots
    plots = generate_all_plots(pipeline, X_test, y_test, model_name=model_name)
    
    logger.info({"generated_plots": len(plots)})
    logger.success(f"Step 7/7: COMPLETED for {model_name}")


# ==============================================================================
# DAG DEFINITION
# ==============================================================================

with DAG(
    dag_id='shock_prediction_pipeline',
    default_args=default_args,
    description='Pipeline de prediccion de shock hemorragico - Entrena y evalua multiples modelos',
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'shock', 'prediction', 'multi-model']
) as dag:
    
    # Steps 1-4: Data preparation (common for all models)
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
    
    # Steps 5-7: Model-specific tasks (one set per model)
    from src.utils import get_config
    model_tasks = {}
    for model_name in list(get_config().get_model_names()):
        # Create tasks for this model
        train_task = PythonOperator(
            task_id=f'train_model_{model_name}',
            python_callable=step5_train_model,
            op_kwargs={'model_name': model_name}
        )
        
        evaluate_task = PythonOperator(
            task_id=f'evaluate_model_{model_name}',
            python_callable=step6_evaluate_model,
            op_kwargs={'model_name': model_name}
        )
        
        plots_task = PythonOperator(
            task_id=f'generate_plots_{model_name}',
            python_callable=step7_generate_evaluation_plots,
            op_kwargs={'model_name': model_name}
        )
        
        # Set dependencies for this model
        t3_create_dataset >> train_task >> evaluate_task >> plots_task
        
        model_tasks[model_name] = {
            'train': train_task,
            'evaluate': evaluate_task,
            'plots': plots_task
        }
    
    # Dependencies for data preparation
    t1_validate >> t2_clean >> t3_create_dataset
    t3_create_dataset >> t4_eda
