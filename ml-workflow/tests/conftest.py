import pytest
import pandas as pd


@pytest.fixture
def get_url():
    url = os.getenv("NEONDB_URI")
    return url


@pytest.fixture
def mock_data():
    # Crée un DataFrame factice pour les tests
    return pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01", periods=10),
            "text": ["texte exemple"] * 10,
            # Ajoute les autres colonnes nécessaires
        }
    )
