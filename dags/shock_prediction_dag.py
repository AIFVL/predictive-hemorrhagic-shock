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
    """Step 5/8: Train Model (uses optimized hyperparameters if available)."""
    import pandas as pd
    from src.utils import get_config, logger, log_section
    from src.models.train import optimize_hyperparameters, cross_validate_model, train_model, save_model
    from src.features.base_features import get_target_variable
    
    config = get_config()
    log_section(f"STEP 5/8: TRAINING MODEL - {model_name.upper()}")
    
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
    scale_features = cv_config.get('scale_features', True)
    
    # Pull hyperparameter search results from previous step (preferred)
    ti: TaskInstance = kwargs.get('ti')
    xcom_payload = None
    if ti is not None:
        xcom_payload = ti.xcom_pull(task_ids=f'optimize_hyperparams_{model_name}')

    best_params = None
    search_results = {}
    if isinstance(xcom_payload, dict):
        best_params = xcom_payload.get('best_params')
        search_results = xcom_payload.get('search_results', {}) or {}

    # Fallback: if the optimize step didn't run / returned nothing, optimize here
    if not best_params:
        best_params, search_results = optimize_hyperparameters(
            X, y,
            model_config=model_config,
            model_name=model_name,
            scale_features=scale_features,
        )
    
    # Update model config with optimized parameters
    optimized_model_config = model_config.copy()
    optimized_model_config['params'] = best_params
    
    # Cross-validate with optimized parameters
    cv_results = cross_validate_model(
        X, y,
        model_config=optimized_model_config,
        model_name=model_name,
        n_folds=cv_config.get('n_folds', 5),
        scale_features=scale_features,
        random_state=random_state
    )
    
    # Train final model with optimized parameters
    pipeline, metadata = train_model(
        X, y,
        model_config=optimized_model_config,
        model_name=model_name,
        scale_features=scale_features
    )
    
    # Add optimization results to metadata
    metadata['cv_results'] = cv_results
    metadata['hyperparameter_search'] = search_results
    metadata['optimized_params'] = best_params
    
    # Save model
    save_model(pipeline, model_name, metadata)
    
    logger.success(f"Step 5/8: COMPLETED for {model_name}")
    return model_name


def step5a_optimize_hyperparams(model_name: str, **kwargs):
    """Step 5A/8: Optimize Hyperparameters (runs once, pushed to XCom)."""
    import pandas as pd
    from src.utils import get_config, logger, log_section, DataWriter
    from src.models.train import optimize_hyperparameters
    from src.features.base_features import get_target_variable

    config = get_config()
    log_section(f"STEP 5A/8: OPTIMIZING HYPERPARAMETERS - {model_name.upper()}")

    # Load training split
    train_split_path = Path(config.get_path('splits_dir')) / 'train.parquet'
    df = pd.read_parquet(train_split_path)

    target_variable = get_target_variable()
    y = df[target_variable]
    X = df.drop(columns=[target_variable])

    logger.info({"samples": len(df), "features": X.shape[1]})

    model_config = config.get_model(model_name)
    cv_config = config.get_cv_config()
    scale_features = cv_config.get('scale_features', True)

    best_params, search_results = optimize_hyperparameters(
        X,
        y,
        model_config=model_config,
        model_name=model_name,
        scale_features=scale_features,
    )

    # Persist best params as an artifact (easy to audit without opening metadata)
    artifact_dir = Path(config.get_path('output_base')) / 'hyperparameter_search' / model_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    DataWriter.write_json_file(best_params, str(artifact_dir / 'best_params.json'))
    DataWriter.write_json_file(search_results, str(artifact_dir / 'search_results.json'))

    logger.success(f"Step 5A/8: COMPLETED for {model_name}")
    return {"best_params": best_params, "search_results": search_results}


def step5b_optimize_threshold(model_name: str, **kwargs):
    """Step 5B/8: Optimize Classification Threshold on TRAIN Set"""
    import pandas as pd
    import numpy as np
    from src.utils import get_config, logger, log_section
    from src.models.train import load_model, update_model_metadata
    from src.models.evaluate import find_optimal_threshold_for_target_recall
    from src.features.base_features import get_target_variable
    
    config = get_config()
    log_section(f"STEP 5B/8: OPTIMIZING THRESHOLD ON TRAIN SET - {model_name.upper()}")
    
    # Load TRAIN split (NOT TEST!)
    train_split_path = Path(config.get_path('splits_dir')) / 'train.parquet'
    df_train = pd.read_parquet(train_split_path)
    
    # Get target variable from config
    target_variable = get_target_variable()
    
    y_train = df_train[target_variable]
    X_train = df_train.drop(columns=[target_variable])
    
    logger.info(f"Train set: {len(y_train)} samples, {int(y_train.sum())} positives ({y_train.mean():.1%})")
    
    # Load model
    pipeline = load_model(model_name)
    
    # Get probabilities on TRAIN set
    y_prob = pipeline.predict_proba(X_train)[:, 1]
    
    # Get threshold configuration
    threshold_config = config.get_threshold_optimization_config()
    target_recall = config.get_target_recall_for_model(model_name)
    
    logger.info(f"Finding optimal threshold for target recall >= {target_recall:.0%}")
    
    # Get search range
    search_config = threshold_config.get('search_thresholds')
    search_thresholds = np.arange(search_config[0], search_config[1], search_config[2]).tolist()
    
    # Get test thresholds for metadata
    test_thresholds = threshold_config.get('test_thresholds')
    
    # Find optimal threshold on TRAIN set
    optimal_result = find_optimal_threshold_for_target_recall(
        y_train, 
        y_prob, 
        target_recall=target_recall,
        thresholds=search_thresholds,
        test_thresholds=test_thresholds
    )
    
    # Extract for logging
    opt_point = optimal_result['operating_point']
    threshold = opt_point['threshold']
    metrics = opt_point['metrics']
    cm = opt_point['confusion_matrix']
    
    logger.info(f"OPTIMAL THRESHOLD FOUND (ON TRAIN): {threshold:.3f}")
    logger.info({
        "dataset": "TRAIN (optimization)",
        "target_recall": f">= {target_recall:.0%}",
        "threshold": threshold,
        "recall": f"{metrics['recall']:.3f} ({metrics['recall']:.1%})",
        "precision": f"{metrics['precision']:.3f} ({metrics['precision']:.1%})",
        "specificity": f"{metrics['specificity']:.3f} ({metrics['specificity']:.1%})",
        "f1_score": metrics['f1_score'],
        "f2_score": metrics['f2_score'],
        "kappa": metrics['kappa'],
        "accuracy": metrics['accuracy'],
        "confusion_matrix": cm
    })
    
    # Save threshold to model metadata
    update_model_metadata(model_name, {
        'threshold_optimization': optimal_result['threshold_optimization'],
        'operating_point': optimal_result['operating_point']
    })
    
    logger.success(f"Step 5B/8: COMPLETED for {model_name}")
    logger.info("Threshold optimized on TRAIN and saved to model metadata")
    
    return optimal_result


def step6_evaluate_model(model_name: str, **kwargs):
    """Step 6/8: Evaluate Model on Test Set with Optimal Threshold"""
    import pandas as pd
    import json
    from src.utils import get_config, logger, log_section
    from src.models.train import load_model
    from src.models.evaluate import evaluate_model, save_evaluation_results
    from src.features.base_features import get_target_variable
    
    config = get_config()
    log_section(f"STEP 6/8: EVALUATING MODEL ON TEST SET - {model_name.upper()}")
    
    # Load test split
    test_split_path = Path(config.get_path('splits_dir')) / 'test.parquet'
    df_test = pd.read_parquet(test_split_path)
    
    # Get target variable from config
    target_variable = get_target_variable()
    
    y_test = df_test[target_variable]
    X_test = df_test.drop(columns=[target_variable])
    
    logger.info(f"Test set: {len(y_test)} samples, {int(y_test.sum())} positives ({y_test.mean():.1%})")
    
    # Load model
    pipeline = load_model(model_name)
    
    # Load optimal threshold from metadata
    model_dir = Path(config.get_path('model_output', model_name=model_name)).parent
    metadata_path = model_dir / 'metadata.json'
    
    optimal_threshold = 0.5  # Default
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        if 'operating_point' in metadata:
            optimal_threshold = metadata['operating_point']['threshold']
            logger.info(f"Using optimal threshold from TRAIN optimization: {optimal_threshold:.3f}")
        else:
            logger.warning("No optimal threshold found in metadata, using default 0.5")
    else:
        logger.warning("Metadata file not found, using default threshold 0.5")
    
    # Get predictions with optimal threshold
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= optimal_threshold).astype(int)
    
    # Evaluate with both default and optimal thresholds
    results = evaluate_model(
        pipeline, X_test, y_test,
        dataset_name='test',
        use_optimal_threshold=False  # We'll add optimal results manually
    )
    
    # Calculate metrics with optimal threshold
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    optimal_metrics = {
        'threshold': optimal_threshold,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'kappa': cohen_kappa_score(y_test, y_pred),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }
    }
    
    # Add optimal threshold results to evaluation
    results['optimal_threshold_results'] = optimal_metrics
    results['model_name'] = model_name
    
    # Log results
    logger.info("TEST SET RESULTS (OPTIMAL THRESHOLD)")
    logger.info({
        "threshold": f"{optimal_threshold:.3f} (from TRAIN optimization)",
        "accuracy": f"{optimal_metrics['accuracy']:.4f}",
        "precision": f"{optimal_metrics['precision']:.4f}",
        "recall": f"{optimal_metrics['recall']:.4f} ⭐",
        "specificity": f"{optimal_metrics['specificity']:.4f}",
        "f1_score": f"{optimal_metrics['f1_score']:.4f}",
        "kappa": f"{optimal_metrics['kappa']:.4f}",
        "confusion_matrix": optimal_metrics['confusion_matrix']
    })
    
    # Save results
    save_evaluation_results(results, model_name)
    
    logger.success(f"Step 6/8: COMPLETED for {model_name}")
    logger.info("Model evaluated on TEST set (no optimization, only reporting)")


def step6b_compare_thresholds_on_test(model_name: str, **kwargs):
    """Step 6B/8: Compare Different Thresholds on Test Set (Reporting Only)"""
    import pandas as pd
    import numpy as np
    import json
    from src.utils import get_config, logger, log_section, DataWriter
    from src.models.train import load_model
    from src.models.evaluate import analyze_thresholds
    from src.features.base_features import get_target_variable
    
    config = get_config()
    log_section(f"STEP 6B/8: COMPARING THRESHOLDS ON TEST SET - {model_name.upper()}")
    
    # Load test split
    test_split_path = Path(config.get_path('splits_dir')) / 'test.parquet'
    df_test = pd.read_parquet(test_split_path)
    
    # Get target variable from config
    target_variable = get_target_variable()
    
    y_test = df_test[target_variable]
    X_test = df_test.drop(columns=[target_variable])
    
    # Load model
    pipeline = load_model(model_name)
    
    # Get probabilities
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Load optimal threshold from metadata
    model_dir = Path(config.get_path('model_output', model_name=model_name)).parent
    metadata_path = model_dir / 'metadata.json'
    
    optimal_threshold = 0.5
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        if 'operating_point' in metadata:
            optimal_threshold = metadata['operating_point']['threshold']
    
    # Get test thresholds from config
    threshold_config = config.get_threshold_optimization_config()
    test_thresholds = threshold_config.get('test_thresholds', [])
    
    # Add optimal threshold to comparison (if not already present)
    comparison_thresholds = list(test_thresholds)
    if optimal_threshold not in comparison_thresholds:
        comparison_thresholds.append(optimal_threshold)
    comparison_thresholds.sort()
    
    logger.info(f"Comparing thresholds on TEST set: {comparison_thresholds}")
    logger.info(f"Optimal threshold (from TRAIN): {optimal_threshold:.3f}")
    
    # Analyze all thresholds on TEST
    df_results = analyze_thresholds(y_test, y_prob, comparison_thresholds)
    
    # Mark which one is optimal
    df_results['is_optimal'] = df_results['threshold'].apply(
        lambda x: 'OPTIMAL' if abs(x - optimal_threshold) < 0.001 else ''
    )
    
    # Log results table
    logger.info("THRESHOLD COMPARISON ON TEST SET (REPORTING ONLY)")
    logger.info("\n" + df_results.to_string(index=False, float_format='%.3f'))
    
    # Save comparison results
    output_dir = Path(config.get_path('output_base')) / 'threshold_analysis' / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    threshold_csv_path = output_dir / 'threshold_comparison_test.csv'
    df_results.to_csv(threshold_csv_path, index=False)
    logger.info(f"Threshold comparison saved to: {threshold_csv_path}")
    
    # Save as JSON too
    comparison_dict = {
        'optimal_threshold': optimal_threshold,
        'comparison_thresholds': comparison_thresholds,
        'results': df_results.to_dict('records')
    }
    
    comparison_json_path = output_dir / 'threshold_comparison_test.json'
    DataWriter.write_json_file(comparison_dict, str(comparison_json_path))
    
    logger.success(f"Step 6B/8: COMPLETED for {model_name}")
    logger.info("Threshold comparison reported on TEST (no decisions made)")
    
    return comparison_dict


def step7_generate_evaluation_plots(model_name: str, **kwargs):
    """Step 7/8: Generate Evaluation Plots"""
    import pandas as pd
    from src.utils import get_config, logger, log_section
    from src.models.train import load_model
    from src.visualization.generate_plots import generate_all_plots
    from src.features.base_features import get_target_variable
    
    config = get_config()
    log_section(f"STEP 7/8: GENERATING EVALUATION PLOTS - {model_name.upper()}")
    
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
    logger.success(f"Step 7/8: COMPLETED for {model_name}")


def step8_generate_comparison_plots(**kwargs):
    """Step 8/8: Generate Cross-Model Comparison Plots"""
    from src.utils import get_config, logger, log_section
    from src.visualization.generate_plots import generate_comparison_plots

    config = get_config()
    log_section("STEP 8/8: GENERATING CROSS-MODEL COMPARISON PLOTS")

    plots = generate_comparison_plots()

    logger.info({"generated_comparison_plots": len(plots)})
    logger.success("Step 8/8: COMPLETED")
    return list(plots.keys())


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
    from src.utils import get_config
    model_tasks = {}
    for model_name in list(get_config().get_model_names()):
        # Create tasks for this model
        optimize_hyperparams_task = PythonOperator(
            task_id=f'optimize_hyperparams_{model_name}',
            python_callable=step5a_optimize_hyperparams,
            op_kwargs={'model_name': model_name}
        )

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
        # OPTIMIZE_HYPERPARAMS → TRAIN → OPTIMIZE_THRESHOLD (train) → EVALUATE (test) → COMPARE (test) → PLOTS
        t3_create_dataset >> optimize_hyperparams_task >> train_task >> optimize_threshold_task >> evaluate_task >> compare_thresholds_task >> plots_task
        
        model_tasks[model_name] = {
            'optimize_hyperparams': optimize_hyperparams_task,
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
