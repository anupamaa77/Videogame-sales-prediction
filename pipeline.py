from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator

from data_ingestion import ingest_data
from data_cleaning import clean_vgsales_data
from great_expectation_raw import validate_raw_data
from great_expectation_clean import validate_clean_data
from redis_upload import run_full_saving_pipeline
from model_training import run_full_training_pipeline
from mlflow_logging import run_mlflow_logging_task


from datetime import datetime, timedelta

default_args = {
    "owner": "Anupama Rai",
    "depends_on_past": False,
    "email": ["Anupama.Rai@mail.bcu.ac.uk"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5)
}

with DAG(
    "final_project",
    default_args=default_args,
    description="""
        End-to-end podcast analytics and machine learning pipeline.
        Implements ingestion to a star schema in MariaDB, preprocessing and caching with Redis,
        model training and evaluation tracked via MLflow, and best model deployment using FastAPI.
        Includes analytics on listening behavior using SQL queries over fact and dimension tables.
        """,
    schedule_interval=timedelta(days=1),
    start_date=datetime(2021, 10, 10),
    catchup=False,
    tags=["example"],
) as dag:

    # Start containers
    start_course_container = BashOperator(
        task_id='run_course_container',
        bash_command='docker start course_container',
    )

    start_redis_store = BashOperator(
        task_id='run_redis_store',
        bash_command='docker start redis_store',
    )

    start_mlflow_ui = BashOperator(
        task_id='start_mlflow_ui',
        bash_command="""
        if ! lsof -i:5000; then
            nohup conda run -n final_project mlflow ui --host 0.0.0.0 --port 5000 > /tmp/mlflow_ui.log 2>&1 &
            echo "MLflow UI started"
        else
            echo "MLflow UI already running"
        fi
        """,
    )

    file_path = "/home/anupamarai24128432/vgsales_project/data/vgsales.csv"

    data_ingestion = PythonOperator(
        task_id="data_ingestion",
        python_callable=ingest_data,
    )

    data_validation_raw = PythonOperator(
        task_id="validate_raw_data",
        python_callable=validate_raw_data,
    )

    data_validation_clean = PythonOperator(
        task_id="validate_clean_data",
        python_callable=validate_clean_data,
    )

    data_cleaning = PythonOperator(
        task_id="data_cleaning",
        python_callable=clean_vgsales_data,
        op_args=[file_path],  # Optional, only if your function accepts it
    )

    # ✅ POST-VALIDATION WRAPPER FUNCTION
    def run_post_validation_preprocessing():
        import pandas as pd
        from post_validation_preprocessing import post_validation_preprocessing

        validated_path = "/home/anupamarai24128432/vgsales_project/data/validated_vgsales.csv"
        output_path = "/home/anupamarai24128432/vgsales_project/data/final_preprocessed_vgsales.csv"

        validated_df = pd.read_csv(validated_path)
        processed_df = post_validation_preprocessing(validated_df)
        processed_df.to_csv(output_path, index=False)

        print("✅ Post-validation preprocessing completed and saved.")

    post_validation_operator = PythonOperator(
        task_id="post_validation_preprocessing_task",
        python_callable=run_post_validation_preprocessing,
    )
    upload_to_redis = PythonOperator(
        task_id="upload_to_redis",
        python_callable=run_full_saving_pipeline,
    )
    train_model= PythonOperator(
        task_id="train_model",
        python_callable=run_full_training_pipeline,
    )
    mlflow_logging = PythonOperator(
    task_id="mlflow_logging",
    python_callable=run_mlflow_logging_task,
    ) 

    # DAG Task Dependency Chain
    [start_course_container, start_redis_store, start_mlflow_ui] >> data_ingestion >> data_validation_raw >> data_validation_clean >> data_cleaning >> post_validation_operator >> upload_to_redis >> train_model >> mlflow_logging
