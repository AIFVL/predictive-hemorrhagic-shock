"""Pipeline step functions used by the Airflow DAG.

The DAG should orchestrate only. Business logic belongs here and in domain modules.
"""

from pathlib import Path
import json
from functools import wraps

import numpy as np
import pandas as pd
from airflow.models import TaskInstance

from src.utils import get_config, logger, log_section, DataLoader
from src.datasets.loaders import split_features_and_target
from src.data.validate import validate_data
from src.data.clean import clean_data
from src.datasets.make_dataset import (
    make_training_dataset,
    save_training_dataset,
    create_train_test_split,
    save_splits,
)
from src.features.base_features import (
    get_numerical_features,
    get_binary_features,
    get_target_variable,
)
from src.models.train import (
    train_complete_workflow,
    save_model,
    load_model,
    update_model_metadata,
)
from src.models.evaluate import (
    optimize_threshold_workflow,
    evaluate_complete_workflow,
    compare_thresholds_workflow,
    save_evaluation_results,
)
from src.visualization.generate_eda_plots import generate_all_eda_plots
from src.visualization.generate_plots import generate_all_plots, generate_comparison_plots


def _pipeline_step(title_template: str, success_template: str):
    """Decorator to standardize section/success/error logging for pipeline steps."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = dict(kwargs)
            if 'model_name' not in context and args and isinstance(args[0], str):
                context['model_name'] = args[0]

            title = title_template.format(**context)
            success = success_template.format(**context)

            log_section(title)
            try:
                result = func(*args, **kwargs)
            except Exception:
                logger.exception(f"{title} FAILED")
                raise

            logger.success(success)
            return result

        return wrapper

    return decorator


@_pipeline_step('STEP 1/7: VALIDATING RAW DATA', 'Step 1/7: COMPLETED')
def step1_validate_raw_data(**kwargs):
    """Step 1/7: Validate Raw Data."""
    config = get_config()

    df = DataLoader.load(config.get_path('raw_data'))
    _, report = validate_data(df)

    report_path = Path(config.get_path('output_base')) / 'validation_report.json'
    DataLoader.save(report, report_path)

    return report


@_pipeline_step('STEP 2/7: CLEANING DATA', 'Step 2/7: COMPLETED')
def step2_clean_data(**kwargs):
    """Step 2/7: Clean Data."""
    config = get_config()

    df = DataLoader.load(config.get_path('raw_data'))
    df_validated, _ = validate_data(df)
    df_cleaned, report = clean_data(df_validated)

    DataLoader.save(df_cleaned, config.get_path('cleaned_data'))

    report_path = Path(config.get_path('output_base')) / 'cleaning_report.json'
    DataLoader.save(report, report_path)

    return True


@_pipeline_step('STEP 3/7: CREATING TRAINING DATASET WITH FEATURE ENGINEERING', 'Step 3/7: COMPLETED')
def step3_create_training_dataset(**kwargs):
    """Step 3/7: Create Training Dataset with Feature Engineering."""
    config = get_config()

    df = DataLoader.load(config.get_path('cleaned_data'))

    X, y = make_training_dataset(df)
    save_training_dataset(X, y)

    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y,
        test_size=config.get('data_split.test_size'),
        stratify=config.get('data_split.stratify'),
        random_state=config.get('general_config.random_seed'),
    )

    save_splits(X_train, X_test, y_train, y_test)

    return True


@_pipeline_step('STEP 4/7: GENERATING EDA PLOTS', 'Step 4/7: COMPLETED')
def step4_generate_eda_plots(**kwargs):
    """Step 4/7: Generate EDA Plots."""
    config = get_config()

    df = DataLoader.load(config.get_path('cleaned_data'))

    numerical_features = get_numerical_features()
    binary_features = get_binary_features()
    target_variable = get_target_variable()

    numerical_cols = [c for c in numerical_features if c in df.columns]
    binary_cols = [c for c in binary_features if c in df.columns]

    plots = generate_all_eda_plots(df, numerical_cols, binary_cols, target_variable)

    logger.info({'generated_plots': len(plots)})


@_pipeline_step('STEP 5/8: TRAINING MODEL - {model_name}', 'Step 5/8: COMPLETED for {model_name}')
def step5_train_model(model_name: str, **kwargs):
    """Step 5/8: Train Model - orchestrates complete training workflow."""
    config = get_config()
    split_path = Path(config.get_path('splits_dir')) / 'train.parquet'
    df = DataLoader.load(split_path)
    X, y = split_features_and_target(df)

    logger.info({'samples': len(df), 'features': X.shape[1], 'positives': f"{y.mean():.1%}"})

    # Call encapsulated business logic
    pipeline, metadata = train_complete_workflow(X, y, model_name)

    # Steps layer handles persistence
    save_model(pipeline, model_name, metadata)
    return model_name


@_pipeline_step('STEP 5B/8: OPTIMIZING THRESHOLD ON TRAIN SET - {model_name}', 'Step 5B/8: COMPLETED for {model_name}')
def step5b_optimize_threshold(model_name: str, **kwargs):
    """Step 5B/8: Optimize Classification Threshold on TRAIN Set."""
    config = get_config()
    split_path = Path(config.get_path('splits_dir')) / 'train.parquet'
    df_train = DataLoader.load(split_path)
    X_train, y_train = split_features_and_target(df_train)

    pipeline = load_model(model_name)

    # Call encapsulated business logic
    optimal_result = optimize_threshold_workflow(pipeline, X_train, y_train, model_name)

    # Steps layer handles metadata persistence
    update_model_metadata(
        model_name,
        {
            'threshold_optimization': optimal_result['threshold_optimization'],
            'operating_point': optimal_result['operating_point'],
        },
    )

    return optimal_result


@_pipeline_step('STEP 6/8: EVALUATING MODEL ON TEST SET - {model_name}', 'Step 6/8: COMPLETED for {model_name}')
def step6_evaluate_model(model_name: str, **kwargs):
    """Step 6/8: Evaluate Model on Test Set with Optimal Threshold."""
    config = get_config()
    split_path = Path(config.get_path('splits_dir')) / 'test.parquet'
    df_test = DataLoader.load(split_path)
    X_test, y_test = split_features_and_target(df_test)

    logger.info({'samples': len(df_test), 'positives': f"{y_test.mean():.1%}"})

    pipeline = load_model(model_name)

    # Call encapsulated business logic
    results = evaluate_complete_workflow(pipeline, X_test, y_test, model_name)

    # Steps layer handles persistence
    save_evaluation_results(results, model_name)


@_pipeline_step('STEP 6B/8: COMPARING THRESHOLDS ON TEST SET - {model_name}', 'Step 6B/8: COMPLETED for {model_name}')
def step6b_compare_thresholds_on_test(model_name: str, **kwargs):
    """Step 6B/8: Compare Different Thresholds on Test Set (Reporting Only)."""
    config = get_config()
    split_path = Path(config.get_path('splits_dir')) / 'test.parquet'
    df_test = DataLoader.load(split_path)
    X_test, y_test = split_features_and_target(df_test)

    pipeline = load_model(model_name)

    # Call encapsulated business logic
    comparison_dict = compare_thresholds_workflow(pipeline, X_test, y_test, model_name)

    # Steps layer handles persistence
    output_dir = Path(config.get_path('output_base')) / 'models' / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_csv_path = output_dir / f'{model_name}_threshold.parquet'
    df_results = pd.DataFrame(comparison_dict['results'])
    DataLoader.save(df_results, threshold_csv_path)
    logger.info(f'Threshold comparison saved to: {threshold_csv_path}')

    comparison_json_path = output_dir / f'{model_name}_threshold.json'
    DataLoader.save(comparison_dict, comparison_json_path)

    return comparison_dict


@_pipeline_step('STEP 7/8: GENERATING EVALUATION PLOTS - {model_name}', 'Step 7/8: COMPLETED for {model_name}')
def step7_generate_evaluation_plots(model_name: str, **kwargs):
    """Step 7/8: Generate Evaluation Plots."""
    config = get_config()
    split_path = Path(config.get_path('splits_dir')) / 'test.parquet'
    df_test = DataLoader.load(split_path)
    X_test, y_test = split_features_and_target(df_test)
    
    train_path = Path(config.get_path('splits_dir')) / 'train.parquet'
    df_train = DataLoader.load(train_path)
    X_train, y_train = split_features_and_target(df_train)

    pipeline = load_model(model_name)
    plots = generate_all_plots(pipeline, X_test, y_test, X_train=X_train, y_train=y_train, model_name=model_name)

    logger.info({'generated_plots': len(plots)})


@_pipeline_step('STEP 8/8: GENERATING CROSS-MODEL COMPARISON PLOTS', 'Step 8/8: COMPLETED')
def step8_generate_comparison_plots(**kwargs):
    """Step 8/8: Generate Cross-Model Comparison Plots."""
    plots = generate_comparison_plots()

    logger.info({'generated_comparison_plots': len(plots)})
    return list(plots.keys())
