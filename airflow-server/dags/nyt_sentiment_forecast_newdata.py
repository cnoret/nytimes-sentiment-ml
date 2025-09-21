"""NYT Sentiment Forecast DAG with New Data"""

import os
from airflow import DAG
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import pandas as pd
import mlflow
from prophet import Prophet
import psycopg2

NEONDB_URI = Variable.get("NeonDBName")
MLFLOW_URI = Variable.get("MLFLOW_TRACKING_URI")
AWS_ACCESS_KEY_ID = Variable.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = Variable.get("AWS_SECRET_ACCESS_KEY")

# Default Arguments
default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
    "catchup": False,
}

# DAG Definition
dag = DAG(
    dag_id="nyt_sentiment_forecast_newdata",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
)


def fetch_data_from_s3(**context):
    """Fetch data from S3"""
    s3_hook = S3Hook(aws_conn_id="aws_s3")
    file_path = s3_hook.download_file(
        key="training/df_ny_times.json", bucket_name="nytimes-etl", local_path="/tmp"
    )
    context["task_instance"].xcom_push(key="s3_file", value=file_path)


fetch_s3_task = PythonOperator(
    task_id="fetch_data_from_s3", python_callable=fetch_data_from_s3, dag=dag
)


def load_recent_data_from_neondb(**context):
    """Load recent data from NeonDB"""
    conn = psycopg2.connect(Variable.get("NeonDBName"))
    query = "SELECT id, title, abstract, published_date FROM nyt_business_articles"
    df = pd.read_sql(query, conn)
    conn.close()

    recent_data_path = "/tmp/recent_neondb.csv"
    df.to_csv(recent_data_path, index=False)

    context["task_instance"].xcom_push(key="neon_file", value=recent_data_path)


load_neondb_task = PythonOperator(
    task_id="load_recent_data_from_neondb",
    python_callable=load_recent_data_from_neondb,
    dag=dag,
)


def preprocess_and_sentiment(**context):
    """Preprocess data and perform sentiment analysis (robuste & batché)."""
    import os
    import pandas as pd
    from airflow.exceptions import AirflowException
    from transformers import pipeline

    # Cache HF silencieux
    os.environ.setdefault("HF_HOME", "/tmp/huggingface_cache")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    # 1) Inputs depuis XCom
    ti = context["task_instance"]
    s3_file = ti.xcom_pull(key="s3_file", task_ids="fetch_data_from_s3")              # JSON path/string
    recent_file = ti.xcom_pull(key="neon_file", task_ids="load_recent_data_from_neondb")  # CSV path

    if not s3_file:
        raise AirflowException("XCom 's3_file' manquant (fetch_data_from_s3).")
    if not recent_file:
        raise AirflowException("XCom 'neon_file' manquant (load_recent_data_from_neondb).")

    # 2) Charger les données
    # pd.read_json accepte un chemin OU une string JSON
    df_s3 = pd.read_json(s3_file, lines=False)
    df_neon = pd.read_csv(recent_file)

    # 3) Harmoniser les colonnes et la section
    df_neon = df_neon.rename(columns={
        "title": "snippet",
        "abstract": "lead_paragraph",
        "published_date": "pub_date",
    })
    df_neon["section_name"] = "Business Day"

    df_combined = pd.concat([df_s3, df_neon], ignore_index=True)

    # Garde Business / Business Day et travaille sur une COPIE pour éviter les warnings
    df_business = df_combined.loc[
        df_combined["section_name"].isin(["Business", "Business Day"])
    ].copy()

    # 4) Dates propres
    df_business["pub_date"] = pd.to_datetime(df_business.get("pub_date"), errors="coerce", utc=True)
    invalid = df_business["pub_date"].isna().sum()
    if invalid:
        print(f"[preprocess] Lignes avec pub_date invalide: {invalid}")
        df_business = df_business.dropna(subset=["pub_date"])
    # supprime la timezone si présente
    try:
        df_business["pub_date"] = df_business["pub_date"].dt.tz_localize(None)
    except Exception:
        pass

    # 5) Construire le texte sans NaN et sans types float
    for col in ("snippet", "lead_paragraph"):
        if col not in df_business.columns:
            df_business[col] = ""
        df_business.loc[:, col] = df_business[col].fillna("").astype(str)

    df_business["text"] = (df_business["snippet"] + " " + df_business["lead_paragraph"]).str.strip()
    # Garde uniquement les textes non vides
    df_business = df_business.loc[df_business["text"].str.len() > 0].copy()

    if df_business.empty:
        raise AirflowException("Aucun texte à scorer après nettoyage.")

    # 6) Pipeline HF (multilingue) + batch
    pipe = pipeline(
        task="text-classification",
        model="tabularisai/multilingual-sentiment-analysis",
        device=-1,            # CPU
        truncation=True
    )

    texts = df_business["text"].astype(str).tolist()  # <- évite TypeError
    sentiments = []
    BATCH = 32
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        # Le pipeline gère la troncation; si tu veux forcer : [t[:512] for t in batch]
        out = pipe(batch)
        sentiments.extend(out)

    # 7) Assembler le résultat
    out_df = pd.DataFrame(sentiments)
    # normaliser les noms de colonnes si besoin
    if "label" not in out_df.columns and "labels" in out_df.columns:
        out_df = out_df.rename(columns={"labels": "label"})
    df_result = pd.concat([df_business.reset_index(drop=True), out_df], axis=1)

    # Score signé (optionnel)
    if "label" in df_result.columns and "score" in df_result.columns:
        df_result["sentiment_signed"] = df_result.apply(
            lambda r: r["score"] if str(r["label"]).upper().startswith("POS") else -r["score"],
            axis=1
        )

    # 8) Sauvegarde + XCom
    out_path = "/tmp/sentiments.csv"
    df_result.to_csv(out_path, index=False)
    ti.xcom_push(key="sentiments_file", value=out_path)

    print(f"[preprocess] Textes scorés: {len(df_result)} "
          f"| Pos≈{(df_result.get('label','').astype(str).str.upper().str.startswith('POS')).mean():.2%}")

preprocess_task = PythonOperator(
    task_id="preprocess_and_sentiment",
    python_callable=preprocess_and_sentiment,
    dag=dag,
)


def prophet_forecast():
    """Forecast sentiment using Prophet"""
    df_sentiments = pd.read_csv("/tmp/sentiments.csv")
    df_sentiments["date"] = pd.to_datetime(df_sentiments["pub_date"]).dt.date
    df_pivot = df_sentiments.pivot_table(
        index="date", columns="label", values="score", aggfunc="mean"
    ).reset_index()

    forecasts = []
    for label in df_pivot.columns[1:]:
        df_label = (
            df_pivot[["date", label]]
            .rename(columns={"date": "ds", label: "y"})
            .dropna()
        )
        model = Prophet()
        model.fit(df_label)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        forecast["label"] = label
        forecasts.append(forecast[["ds", "yhat", "label"]])

    pd.concat(forecasts).to_csv("/tmp/forecast.csv", index=False)


forecast_task = PythonOperator(
    task_id="prophet_forecast", python_callable=prophet_forecast, dag=dag
)


def store_forecasts_neondb():
    """Store forecasts in NeonDB"""
    conn = psycopg2.connect(NEONDB_URI)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiment_forecasts (
            id SERIAL PRIMARY KEY,
            date TIMESTAMP,
            sentiment_label TEXT,
            prediction FLOAT
        );
    """
    )

    df_forecast = pd.read_csv("/tmp/forecast.csv")
    for _, row in df_forecast.iterrows():
        cursor.execute(
            "INSERT INTO sentiment_forecasts (date, sentiment_label, prediction) VALUES (%s,%s,%s)",
            (row["ds"], row["label"], row["yhat"]),
        )
    conn.commit()
    cursor.close()
    conn.close()


store_task = PythonOperator(
    task_id="store_forecasts_neondb", python_callable=store_forecasts_neondb, dag=dag
)


def track_mlflow():
    """Track metrics and artifacts in MLflow"""
    mlflow.set_tracking_uri(Variable.get("MLFLOW_TRACKING_URI"))

    os.environ["AWS_ACCESS_KEY_ID"] = Variable.get("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = Variable.get("AWS_SECRET_ACCESS_KEY")

    mlflow.set_experiment("nyt-sentiment-forecast")

    with mlflow.start_run():
        df_forecast = pd.read_csv("/tmp/forecast.csv")

        for sentiment in df_forecast["label"].unique():
            avg_pred = df_forecast[df_forecast["label"] == sentiment]["yhat"].mean()
            mlflow.log_metric(f"average_prediction_{sentiment}", avg_pred)

        mlflow.log_artifact("/tmp/forecast.csv", "forecast_data")


mlflow_task = PythonOperator(
    task_id="track_mlflow", python_callable=track_mlflow, dag=dag
)

start = DummyOperator(task_id="start", dag=dag)
end = DummyOperator(task_id="end", dag=dag)

(
    start
    >> [fetch_s3_task, load_neondb_task]
    >> preprocess_task
    >> forecast_task
    >> store_task
    >> mlflow_task
    >> end
)
