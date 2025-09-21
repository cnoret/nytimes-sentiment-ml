# Library
import pandas as pd
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt
from prophet import Prophet


# Fonction pour charger les données
def load_data(url):
    df = pd.read_json(url)
    return df


# Fonction pour prétraiter les données
def preprocess_data(df):
    df_business = df[(df["section_name"] == "Business Day")]
    df_business = df_business.loc[:, ["snippet", "lead_paragraph", "pub_date"]]
    df_business["pub_date"] = pd.to_datetime(df_business["pub_date"])
    df_business["date"] = df_business["pub_date"].dt.strftime("%Y-%m-%d")
    df_business.drop("pub_date", axis=1, inplace=True)
    df_business["text"] = df_business["snippet"] + "_" + df_business["lead_paragraph"]
    df_business = df_business.reset_index(drop=True)
    return df_business


# Fonction pour appliquer l'analyse de sentiment
def apply_sentiment_analysis(df):
    pipe = pipeline(
        "text-classification", model="tabularisai/multilingual-sentiment-analysis"
    )
    data = []
    for text in df["text"]:
        result = pipe(text)
        data.append(result[0])  # Ajouter le premier (et seul) dictionnaire
    df_SA = pd.DataFrame(data)
    df_SA["date"] = df["date"]
    return df_SA


# Fonction pour prétraiter les scores de sentiment
def preprocess_sentiment_scores(df_SA):
    df_mean_scores = df_SA.groupby(["date", "label"], as_index=False)["score"].mean()
    df_pivot = df_SA.pivot_table(
        index="date", columns="label", values="score", aggfunc="mean"
    )
    df_pivot = df_pivot.fillna(0)
    return df_pivot


# Fonction pour diviser les données en sous-ensembles par label
def split_data_by_label(df_pivot):
    df_Very_Negative = df_pivot[["Very Negative"]].reset_index()
    df_Very_Negative.columns = ["ds", "y"]
    df_Negative = df_pivot[["Negative"]].reset_index()
    df_Negative.columns = ["ds", "y"]
    df_Neutral = df_pivot[["Neutral"]].reset_index()
    df_Neutral.columns = ["ds", "y"]
    df_Positive = df_pivot[["Positive"]].reset_index()
    df_Positive.columns = ["ds", "y"]
    df_Very_Positive = df_pivot[["Very Positive"]].reset_index()
    df_Very_Positive.columns = ["ds", "y"]
    return df_Very_Negative, df_Negative, df_Neutral, df_Positive, df_Very_Positive


# Fonction pour entraîner un modèle Prophet et faire des prédictions
def train_and_predict(df, label):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    forecast["label"] = label
    return forecast


# Fonction pour visualiser les prédictions
def plot_forecast(forecast, label):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(forecast["ds"], forecast["yhat"], label="Prédiction", color="blue")
    ax.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        color="lightblue",
        alpha=0.3,
        label="Intervalle de confiance",
    )
    ax.set_title(f"Prédictions pour le label : {label}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur prédite")
    ax.legend()
    plt.show()


# Fonction pour visualiser toutes les prédictions ensemble
def plot_all_forecasts(all_forecasts):
    plt.figure(figsize=(12, 8))
    for label, group in all_forecasts.groupby("label"):
        plt.plot(group["ds"], group["yhat"], label=label)
    plt.title("Prédictions pour tous les labels")
    plt.xlabel("Date")
    plt.ylabel("Valeur prédite")
    plt.legend()
    plt.show()


# Fonction principale pour exécuter le pipeline
def main():
    # Charger les données
    url = "https://lead-project-ny-times-ml-ops.s3.eu-west-3.amazonaws.com/df_ny_times.json"
    df_ny_times = load_data(url)

    # Prétraiter les données
    df_business = preprocess_data(df_ny_times)

    # Appliquer l'analyse de sentiment
    df_SA_business = apply_sentiment_analysis(df_business)

    # Prétraiter les scores de sentiment
    df_pivot = preprocess_sentiment_scores(df_SA_business)

    # Diviser les données en sous-ensembles par label
    df_Very_Negative, df_Negative, df_Neutral, df_Positive, df_Very_Positive = (
        split_data_by_label(df_pivot)
    )

    # Entraîner les modèles et faire des prédictions
    forecast_Very_Negative = train_and_predict(df_Very_Negative, "Very Negative")
    forecast_Negative = train_and_predict(df_Negative, "Negative")
    forecast_Neutral = train_and_predict(df_Neutral, "Neutral")
    forecast_Positive = train_and_predict(df_Positive, "Positive")
    forecast_Very_Positive = train_and_predict(df_Very_Positive, "Very Positive")

    # Visualiser les prédictions pour chaque label
    plot_forecast(forecast_Very_Negative, "Very Negative")
    plot_forecast(forecast_Negative, "Negative")
    plot_forecast(forecast_Neutral, "Neutral")
    plot_forecast(forecast_Positive, "Positive")
    plot_forecast(forecast_Very_Positive, "Very Positive")

    # Combiner toutes les prédictions
    all_forecasts = pd.concat(
        [
            forecast_Very_Negative,
            forecast_Negative,
            forecast_Neutral,
            forecast_Positive,
            forecast_Very_Positive,
        ]
    )

    # Visualiser toutes les prédictions ensemble
    plot_all_forecasts(all_forecasts)


# Exécuter le pipeline
if __name__ == "__main__":
    main()
