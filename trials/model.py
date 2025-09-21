#Library
import pandas as pd
import numpy as np
from transformers import pipeline
import plotly.express as px
import matplotlib.pyplot as plt
import requests
import json
from prophet import Prophet
import mlflow

#Loading data
df_ny_times=pd.read_json('https://lead-project-ny-times-ml-ops.s3.eu-west-3.amazonaws.com/df_ny_times.json')

#Preprocess_1
df_business = df_ny_times[(df_ny_times['section_name'] == 'Business Day')]
df_business=df_business.loc[:,['snippet','lead_paragraph','pub_date']]
df_business['pub_date'] = pd.to_datetime(df_business['pub_date'])
df_business['date'] = df_business['pub_date'].dt.strftime('%Y-%m-%d')
df_business.drop('pub_date',axis=1,inplace=True)
df_business['text']=df_business['snippet']+'_'+df_business['lead_paragraph']
df_business = df_business.reset_index(drop=True)

#Model_1: Sentiment Analysis
pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
## Appliquer le modèle de classification à chaque texte
data = []
for text in df_business['text']:
    result = pipe(text)
    data.append(result[0])  # Ajouter le premier (et seul) dictionnaire
## Créer un DataFrame avec les résultats de la classification
df_SA_business = pd.DataFrame(data)
## Ajouter la colonne 'date' de df_business à df_SA_business
df_SA_business['date'] = df_business['date']

#Preprocess_2
## Grouper par 'date' et 'label', puis calculer la moyenne des scores
df_mean_scores = df_SA_business.groupby(['date', 'label'], as_index=False)['score'].mean()
## Créer un tableau pivot avec la moyenne des scores par date et label
df_pivot = df_SA_business.pivot_table(index='date', columns='label', values='score', aggfunc='mean')
## Remplacer les NaN par des zéros
df_pivot = df_pivot.fillna(0)

#Split Dataset en 5 sous-dataseet en fonction du label
df_SA_business_Very_Negative = df_pivot[['Very Negative']]
df_SA_business_Negative = df_pivot[['Negative']]
df_SA_business_Neutral = df_pivot[['Neutral']]
df_SA_business_Positive = df_pivot[['Positive']]
df_SA_business_Very_Very_Positive = df_pivot[['Very Positive']]

#Mise en fonme Time Series Prophet
## Réinitialiser l'index pour avoir 'date' en colonne
df_SA_business_Very_Negative = df_pivot[['Very Negative']].reset_index()  
df_SA_business_Very_Negative.columns = ['ds', 'y']
df_SA_business_Negative = df_pivot[['Negative']].reset_index()
df_SA_business_Negative.columns = ['ds', 'y']
df_SA_business_Neutral = df_pivot[['Neutral']].reset_index()
df_SA_business_Neutral.columns = ['ds', 'y']
df_SA_business_Positive = df_pivot[['Positive']].reset_index()
df_SA_business_Positive.columns = ['ds', 'y']
df_SA_business_Very_Positive = df_pivot[['Very Positive']].reset_index()
df_SA_business_Very_Positive.columns = ['ds', 'y']

#Model_2: Time Series - Prophet
## Fonction pour entraîner un modèle et faire des prédictions
def train_and_predict(df, column_name):
    # Créer et entraîner le modèle
    model = Prophet()
    model.fit(df)
    # Créer un DataFrame pour les prédictions (par exemple, 30 jours dans le futur)
    future = model.make_future_dataframe(periods=30)
    # Faire des prédictions
    forecast = model.predict(future)
    # Ajouter une colonne pour identifier le label
    forecast['label'] = column_name
    return forecast

# Appliquer la fonction à chaque DataFrame
forecast_Very_Negative = train_and_predict(df_SA_business_Very_Negative, 'Very Negative')
forecast_Negative = train_and_predict(df_SA_business_Negative, 'Negative')
forecast_Neutral = train_and_predict(df_SA_business_Neutral, 'Neutral')
forecast_Positive = train_and_predict(df_SA_business_Positive, 'Positive')
forecast_Very_Positive = train_and_predict(df_SA_business_Very_Positive, 'Very Positive')

#Visualizations
def plot_forecast(forecast, label):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(forecast['ds'], forecast['yhat'], label='Prédiction', color='blue')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.3, label='Intervalle de confiance')
    ax.set_title(f'Prédictions pour le label : {label}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Valeur prédite')
    ax.legend()
    plt.show()
# Visualiser les prédictions pour chaque label
plot_forecast(forecast_Very_Negative, 'Very Negative')
plot_forecast(forecast_Negative, 'Negative')
plot_forecast(forecast_Neutral, 'Neutral')
plot_forecast(forecast_Positive, 'Positive')
plot_forecast(forecast_Very_Positive, 'Very Positive')

# Combiner toutes les prédictions
all_forecasts = pd.concat([
    forecast_Very_Negative,
    forecast_Negative,
    forecast_Neutral,
    forecast_Positive,
    forecast_Very_Positive
])

#Visualization_2
plt.figure(figsize=(12, 8))
for label, group in all_forecasts.groupby('label'):
    plt.plot(group['ds'], group['yhat'], label=label)
plt.title('Prédictions pour tous les labels')
plt.xlabel('Date')
plt.ylabel('Valeur prédite')
plt.legend()
plt.show()