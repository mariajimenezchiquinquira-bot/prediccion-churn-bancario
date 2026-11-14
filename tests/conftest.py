# --------------------------------------------------------------------------
# Datos compartidos por las pruebas: el dataset real (una sola vez) y datos
# sintéticos
# --------------------------------------------------------------------------
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.datos import data_prep as dp


# Dataset real, cargado y limpio (una sola vez para todas las pruebas)
@pytest.fixture(scope="session")
def df():
    return dp.clean_data(dp.load_data())


# Partición 80/20 del dataset completo
@pytest.fixture(scope="session")
def datos(df):
    return dp.prepare_data(df)


# Partición de una muestra pequeña, para pruebas rápidas de entrenamiento
@pytest.fixture(scope="session")
def datos_pequenos(df):
    return dp.prepare_data(df.sample(1200, random_state=0))


# Datos sintéticos desbalanceados y un modelo sencillo ya entrenado
@pytest.fixture(scope="session")
def sintetico():
    X, y = make_classification(n_samples=1500, n_features=8, weights=[0.84], random_state=0)
    modelo = LogisticRegression(max_iter=500).fit(X[:1000], y[:1000])
    return modelo, X[1000:], y[1000:], X[:1000], y[:1000]
