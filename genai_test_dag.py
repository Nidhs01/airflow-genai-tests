from airflow import DAG
from airflow.providers.google.cloud.operators.gen_ai import TextGenerationOperator
from datetime import datetime

with DAG(
    dag_id="genai_test_dag",
    start_date=datetime(2026, 3, 20),
    schedule_interval=None,
    catchup=False,
) as dag:

    generate_text = TextGenerationOperator(
        task_id="generate_text",
        model="text-bison-001",  # pick a valid Gen AI model
        input="Hello from GSoC!"
    )

    generate_script = TextGenerationOperator(
        task_id="generate_script",
        model="text-bison-001",
        input="Write a short Python script that prints 'Hello World!'"
    )

    # DAG flow
    generate_text >> generate_script