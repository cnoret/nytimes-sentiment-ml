from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.utils.dates import days_ago
import requests

STREAMLIT_APP_URL = Variable.get("STREAMLIT_APP_URL")


def check_streamlit_app():
    response = requests.get(STREAMLIT_APP_URL)
    if response.status_code != 200:
        raise Exception(f"Erreur HTTP : {response.status_code}")


default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
    "catchup": False,
}

with DAG(
    "monitor_streamlit_slack",
    default_args=default_args,
    schedule_interval="*/30 * * * *",
    catchup=False,
) as dag:

    check_app = PythonOperator(
        task_id="check_streamlit_app",
        python_callable=check_streamlit_app,
    )

    slack_alert = SlackWebhookOperator(
        task_id="send_slack_alert",
        slack_webhook_conn_id="slack_default",
        message=f"⚠️ L'application Streamlit est indisponible : {STREAMLIT_APP_URL}",
        username="AirflowBot",
        trigger_rule="one_failed",
    )

    start = DummyOperator(task_id="start", dag=dag)
    end = DummyOperator(task_id="end", dag=dag)

    start >> check_app >> slack_alert >> end
