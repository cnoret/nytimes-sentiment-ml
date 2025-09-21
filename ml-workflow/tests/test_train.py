import pytest
import pandas as pd
import numpy as np
from datetime import datetime

# Importe le module entier pour pouvoir mocker ses fonctions
import app.train


@pytest.fixture
def mock_ny_times_data():
    """Crée un DataFrame factice pour simuler les données brutes NY Times"""
    return pd.DataFrame(
        {
            "section_name": [
                "Business Day",
                "Business Day",
                "Politics",
                "Business Day",
            ],
            "snippet": [
                "Economic news 1",
                "Stock market up",
                "Political debate",
                "Tech company profits",
            ],
            "lead_paragraph": [
                "Details about economy",
                "Stock details",
                "Political details",
                "Tech details",
            ],
            "pub_date": [
                "2023-01-01T12:00:00Z",
                "2023-01-02T12:00:00Z",
                "2023-01-03T12:00:00Z",
                "2023-01-04T12:00:00Z",
            ],
        }
    )


@pytest.fixture
def mock_business_data():
    """Crée un DataFrame factice pour simuler les données prétraitées business"""
    return pd.DataFrame(
        {
            "text": [
                "Economic news 1_Details about economy",
                "Stock market up_Stock details",
                "Tech company profits_Tech details",
            ],
            "date": ["2023-01-01", "2023-01-02", "2023-01-04"],
        }
    )


@pytest.fixture
def mock_sentiment_data():
    """Crée un DataFrame factice pour simuler les résultats d'analyse de sentiment"""
    return pd.DataFrame(
        {
            "label": [
                "Positive",
                "Very Positive",
                "Neutral",
                "Positive",
                "Very Positive",
            ],
            "score": [0.85, 0.92, 0.55, 0.78, 0.94],
            "date": [
                "2023-01-01",
                "2023-01-01",
                "2023-01-02",
                "2023-01-02",
                "2023-01-03",
            ],
        }
    )


@pytest.fixture
def mock_pivot_data():
    """Crée un DataFrame factice pour simuler les données pivot de sentiment"""
    df = pd.DataFrame(
        index=["2023-01-01", "2023-01-02", "2023-01-03"],
        columns=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
    )
    df.loc["2023-01-01"] = [0.1, 0.2, 0.3, 0.85, 0.92]
    df.loc["2023-01-02"] = [0.0, 0.1, 0.55, 0.78, 0.0]
    df.loc["2023-01-03"] = [0.2, 0.0, 0.0, 0.0, 0.94]
    return df


def test_load_data(monkeypatch, mock_ny_times_data):
    # Mock pour la fonction pd.read_json
    def mock_read_json(url, *args, **kwargs):
        return mock_ny_times_data

    # Applique le mock à pd.read_json
    monkeypatch.setattr(pd, "read_json", mock_read_json)

    # Teste la fonction
    result = app.train.load_data("dummy_url")

    # Vérifie les résultats
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.equals(mock_ny_times_data)


def test_preprocess_data(mock_ny_times_data):
    # Teste la fonction directement
    result = app.train.preprocess_data(mock_ny_times_data)

    # Vérifie les résultats
    assert isinstance(result, pd.DataFrame)
    assert "text" in result.columns
    assert "date" in result.columns
    assert "snippet" in result.columns  # Ces colonnes sont conservées
    assert "lead_paragraph" in result.columns  # Ces colonnes sont conservées
    assert "pub_date" not in result.columns  # Celle-ci est supprimée
    assert "section_name" not in result.columns  # Celle-ci n'est pas sélectionnée
    assert len(result) == 3  # Seulement les lignes "Business Day"


def test_apply_sentiment_analysis(monkeypatch, mock_business_data):
    # Mock pour pipeline de transformers
    class MockPipeline:
        def __call__(self, text):
            # Renvoie un résultat factice basé sur le texte
            if "Economic" in text:
                return [{"label": "Positive", "score": 0.85}]
            elif "Stock" in text:
                return [{"label": "Very Positive", "score": 0.92}]
            else:
                return [{"label": "Neutral", "score": 0.55}]

    # Mock pour la fonction pipeline
    def mock_pipeline(*args, **kwargs):
        return MockPipeline()

    # Applique le mock
    monkeypatch.setattr("app.train.pipeline", mock_pipeline)

    # Teste la fonction
    result = app.train.apply_sentiment_analysis(mock_business_data)

    # Vérifie les résultats
    assert isinstance(result, pd.DataFrame)
    assert "label" in result.columns
    assert "score" in result.columns
    assert "date" in result.columns
    assert len(result) == len(mock_business_data)


def test_preprocess_sentiment_scores(mock_sentiment_data):
    # Teste la fonction directement
    result = app.train.preprocess_sentiment_scores(mock_sentiment_data)

    # Vérifie les résultats
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == "date"
    assert all(
        label in result.columns for label in ["Positive", "Very Positive", "Neutral"]
    )
    assert len(result) == len(mock_sentiment_data["date"].unique())


def test_split_data_by_label(mock_pivot_data):
    # Teste la fonction directement
    result = app.train.split_data_by_label(mock_pivot_data)

    # Vérifie les résultats
    assert isinstance(result, tuple)
    assert len(result) == 5
    for df in result:
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["ds", "y"]
        assert len(df) == len(mock_pivot_data)


def test_train_and_predict(monkeypatch):
    # Créer des données factices pour Prophet
    test_df = pd.DataFrame(
        {"ds": pd.date_range(start="2023-01-01", periods=10), "y": np.random.rand(10)}
    )

    # Mock pour Prophet
    class MockProphet:
        def fit(self, df):
            pass

        def make_future_dataframe(self, periods):
            return pd.DataFrame(
                {"ds": pd.date_range(start="2023-01-01", periods=10 + periods)}
            )

        def predict(self, future):
            # Crée un DataFrame de prédiction factice
            forecast = pd.DataFrame(
                {
                    "ds": future["ds"],
                    "yhat": np.random.rand(len(future)),
                    "yhat_lower": np.random.rand(len(future)),
                    "yhat_upper": np.random.rand(len(future)),
                }
            )
            return forecast

    # Mock pour le constructeur Prophet
    def mock_prophet(*args, **kwargs):
        return MockProphet()

    # Applique le mock
    monkeypatch.setattr("app.train.Prophet", mock_prophet)

    # Teste la fonction
    result = app.train.train_and_predict(test_df, "Test Label")

    # Vérifie les résultats
    assert isinstance(result, pd.DataFrame)
    assert "ds" in result.columns
    assert "yhat" in result.columns
    assert "label" in result.columns
    assert result["label"].iloc[0] == "Test Label"
