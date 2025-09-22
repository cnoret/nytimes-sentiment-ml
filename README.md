# NYT Sentiment Forecast Project - Architecture Overview

This project implements an end-to-end MLOps pipeline for economic sentiment analysis
using data from the **New York Times API**.

## Components

1. **Source**
   - **New York Times API**: Provides the raw articles used for sentiment analysis.
   - Articles are ingested and stored temporarily for processing.

2. **Airflow DAGs**
   - **fetch_data_from_s3**: Retrieves training data snapshots from S3.
   - **load_recent_data_from_neondb**: Loads recent NYT business articles already stored in NeonDB.
   - **preprocess_and_sentiment**: Cleans and enriches the articles, then scores sentiment using a HuggingFace model.
   - **prophet_forecast**: Runs time series forecasting (Prophet) on sentiment trends.
   - **store_forecasts_neondb**: Stores the forecasts in NeonDB for long-term use.
   - **track_mlflow**: Logs metrics and artifacts (forecast results) into MLflow.
   - **monitor_streamlit_app**: Monitors the health of the Streamlit dashboard and reports to Slack.

3. **Temporary Files**
   - `sentiments.csv`: Intermediate file containing raw article texts with sentiment scores.
   - `forecast.csv`: Forecasted sentiment trends for the upcoming period.

4. **Storage and Tracking**
   - **NeonDB**: Stores structured tables of articles and forecasts.
   - **MLflow**: Experiment tracking and metric logging.
   - **S3 Bucket**: Stores artifacts such as models and data snapshots.

5. **CI/CD**
   - **Jenkins**: Automates the ML workflow (train, test, build, push artifacts).

6. **Visualization**
   - **Streamlit Dashboard**: Interactive interface to explore sentiment trends, forecasts, and metrics.
   - Receives data from forecast outputs, NeonDB, and MLflow.

## Data Flow Summary
- Articles come from the **NYT API**.
- Airflow orchestrates ingestion, preprocessing, sentiment scoring, forecasting, and storage.
- Intermediate files (`sentiments.csv`, `forecast.csv`) are created during the process.
- Forecasts and metrics are logged into **NeonDB** and **MLflow**, with artifacts stored on **S3**.
- **Jenkins** ensures CI/CD automation for the ML workflow.
- **Streamlit** provides a visualization layer, monitored by a dedicated Airflow DAG.
