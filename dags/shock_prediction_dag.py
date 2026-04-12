"""
DAG de Airflow para el pipeline de prediccion de shock hemorragico.
Llama directamente a los metodos de los modulos de Python.
Entrena y evalua multiples modelos configurados en pipeline_config.yaml.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from src.utils import get_config
from src.pipeline.steps import (
    step1_validate_raw_data,
    step2_clean_data,
    step3_create_training_dataset,
    step4_generate_eda_plots,
    step5_train_model,
    step5b_optimize_threshold,
    step6_evaluate_model,
    step6b_compare_thresholds_on_test,
    step7_generate_evaluation_plots,
    step8_generate_comparison_plots,
)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 18),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# ==============================================================================
# DAG DEFINITION
# ==============================================================================

with DAG(
    dag_id='shock_prediction_pipeline',
    default_args=default_args,
    description='Pipeline de prediccion de shock hemorragico - Entrena, evalua, optimiza thresholds y visualiza multiples modelos',
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'shock', 'prediction', 'multi-model', 'threshold-optimization']
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
    
    # Steps 5-8: Model-specific tasks (one set per model)
    config = get_config()
    models = config.get('models')
    enabled_models = [name for name, cfg in models.items() if cfg['enabled']]
    
    model_tasks = {}
    for model_name in enabled_models:
        # Create tasks for this model
        train_task = PythonOperator(
            task_id=f'train_model_{model_name}',
            python_callable=step5_train_model,
            op_kwargs={'model_name': model_name}
        )
        
        optimize_threshold_task = PythonOperator(
            task_id=f'optimize_threshold_{model_name}',
            python_callable=step5b_optimize_threshold,
            op_kwargs={'model_name': model_name}
        )
        
        evaluate_task = PythonOperator(
            task_id=f'evaluate_model_{model_name}',
            python_callable=step6_evaluate_model,
            op_kwargs={'model_name': model_name}
        )
        
        compare_thresholds_task = PythonOperator(
            task_id=f'compare_thresholds_{model_name}',
            python_callable=step6b_compare_thresholds_on_test,
            op_kwargs={'model_name': model_name}
        )
        
        plots_task = PythonOperator(
            task_id=f'generate_plots_{model_name}',
            python_callable=step7_generate_evaluation_plots,
            op_kwargs={'model_name': model_name}
        )
        
        # Set dependencies for this model:
        # TRAIN → OPTIMIZE_THRESHOLD (train) → EVALUATE (test) → COMPARE (test) → PLOTS
        t3_create_dataset >> train_task >> optimize_threshold_task >> evaluate_task >> compare_thresholds_task >> plots_task
        
        model_tasks[model_name] = {
            'train': train_task,
            'optimize_threshold': optimize_threshold_task,
            'evaluate': evaluate_task,
            'compare_thresholds': compare_thresholds_task,
            'plots': plots_task
        }
    
    # Step 8: Cross-model comparison — runs once after ALL per-model plots tasks finish
    t8_compare = PythonOperator(
        task_id='generate_comparison_plots',
        python_callable=step8_generate_comparison_plots
    )
    for tasks in model_tasks.values():
        tasks['plots'] >> t8_compare

    # Dependencies for data preparation
    t1_validate >> t2_clean >> t3_create_dataset
    t3_create_dataset >> t4_eda
