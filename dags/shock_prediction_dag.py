"""
DAG de Airflow para el pipeline de predicción de shock hemorrágico.
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator


# Añadir el path para que pueda importar el pipeline
sys.path.insert(0, '/opt/airflow')

# Definir los argumentos por defecto del DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}


def run_shock_pipeline(**kwargs):
    """Función para ejecutar el pipeline de shock."""
    # Importar aquí para evitar problemas de serialización de Airflow
    from simple_pipeline import SimpleShockPipeline
    
    # Parámetros del pipeline
    data_path = kwargs.get('data_path', '/opt/airflow/data/processed/shock.csv')
    output_dir = kwargs.get('output_dir', '/opt/airflow/output/shock_pipeline')
    target_col = kwargs.get('target_col', 'SHOCK')
    
    # Crear y ejecutar pipeline
    pipeline = SimpleShockPipeline(data_path=data_path, output_dir=output_dir)
    results = pipeline.run_pipeline(target_col=target_col)
    
    print("Pipeline completado exitosamente!")
    return results


def run_eda_pipeline(**kwargs):
    """Función para ejecutar el pipeline de EDA."""
    # Importar módulos necesarios
    import pandas as pd
    from pathlib import Path
    
    # Parámetros
    data_path = kwargs.get('data_path', '/opt/airflow/data/processed/shock.csv')
    output_dir = kwargs.get('output_dir', '/opt/airflow/output/eda')
    
    # Asegurar que el directorio de salida exista
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Realizando EDA en: {data_path}")
    
    # Carga de datos
    if str(data_path).endswith('.csv'):
        df = pd.read_csv(data_path)
    elif str(data_path).endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Formato de archivo no soportado: {data_path}")
    
    # Análisis básico
    print(f"Forma del dataset: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    print(f"Resumen estadístico:\n{df.describe()}")
    print(f"Valores nulos:\n{df.isnull().sum()}")
    
    # Guardar reporte de EDA
    report_path = f"{output_dir}/eda_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Reporte de EDA - {datetime.now()}\n")
        f.write(f"Forma del dataset: {df.shape}\n")
        f.write(f"Columnas: {list(df.columns)}\n")
        f.write(f"Valores nulos:\n{df.isnull().sum()}\n")
        f.write(f"Resumen estadístico:\n{df.describe()}\n")
    
    print(f"Reporte de EDA guardado en: {report_path}")
    return f"EDA completado y guardado en: {report_path}"


# Definir el DAG
dag = DAG(
    dag_id='shock_prediction_pipeline',
    default_args=default_args,
    description='Pipeline de predicción de shock hemorrágico',
    schedule_interval=timedelta(days=1),  # Se ejecuta diariamente
    catchup=False,
    tags=['ml', 'shock', 'prediction', 'eda']
)


# Definir tareas
t1 = PythonOperator(
    task_id='run_eda',
    python_callable=run_eda_pipeline,
    op_kwargs={'data_path': '/opt/airflow/data/processed/shock.csv'},
    dag=dag
)

t2 = PythonOperator(
    task_id='run_shock_prediction',
    python_callable=run_shock_pipeline,
    op_kwargs={
        'data_path': '/opt/airflow/data/processed/shock.csv',
        'output_dir': '/opt/airflow/output/shock_model',
        'target_col': 'SHOCK'
    },
    dag=dag
)

t3 = BashOperator(
    task_id='log_results',
    bash_command='echo "Pipeline completado: $(date)" >> /opt/airflow/logs/pipeline_status.log',
    dag=dag
)


# Definir dependencias
t1 >> t2 >> t3