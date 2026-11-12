"""Carga, limpieza y preparación de datos para el proyecto de churn bancario.

Uso típico desde el notebook:

    from src.datos import data_prep as dp
    df = dp.clean_data(dp.load_data())
    datos = dp.prepare_data(df)          # split estratificado + preprocesador
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Constantes del proyecto
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank_churn.xlsx"

ID_COL = "clientnum"
TARGET_ORIGINAL = "attrition_flag"
TARGET = "churn"                      # 1 = cliente se fue, 0 = permaneció
POSITIVE_LABEL = "Attrited Customer"
NEGATIVE_LABEL = "Existing Customer"

CATEGORICAL_COLS = [
    "gender",
    "education_level",
    "marital_status",
    "income_category",
    "card_category",
]

# avg_open_to_buy = credit_limit - total_revolving_bal (diferencia exacta de 0
# en los 10127 registros), por lo que no aporta información nueva.
# avg_utilization_ratio = total_revolving_bal / credit_limit (diferencia menor
# a 0.0005, solo redondeo). Al entrenar con y sin ella el desempeño fue igual,
# así que se quita para tener un modelo más simple.
REDUNDANT_COLS = ["avg_open_to_buy", "avg_utilization_ratio"]

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Carga y limpieza
# --------------------------------------------------------------------------
def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Lee el Excel original sin modificarlo."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo en {path}. "
            "Verifica que bank_churn.xlsx esté en data/raw/."
        )
    return pd.read_excel(path)


def clean_data(df: pd.DataFrame, drop_redundant: bool = True) -> pd.DataFrame:
    """Limpia el dataset y crea la variable objetivo binaria `churn`.

    Decisiones (documentar en el informe):
    - `attrition_flag` se convierte a 0/1 (1 = Attrited Customer).
    - `clientnum` pasa a ser el índice: identifica al cliente pero no es feature.
    - Las categorías "Unknown" se conservan como una categoría propia, ya que
      eliminarlas descartaría entre 7% y 15% de los clientes.
    - No se eliminan outliers: son pocos y pueden ser clientes reales.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    faltantes = {ID_COL, TARGET_ORIGINAL} - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas esperadas: {faltantes}")

    valores = set(df[TARGET_ORIGINAL].dropna().unique())
    if not valores <= {POSITIVE_LABEL, NEGATIVE_LABEL}:
        raise ValueError(f"Valores inesperados en {TARGET_ORIGINAL}: {valores}")

    # Texto limpio
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("string").str.strip()

    # Duplicados por cliente (se informa cuántos registros se eliminan)
    antes = len(df)
    df = df.drop_duplicates(subset=ID_COL)
    print(f"Registros duplicados eliminados: {antes - len(df)}")

    # Variable objetivo
    df[TARGET] = (df[TARGET_ORIGINAL] == POSITIVE_LABEL).astype(int)
    df = df.drop(columns=TARGET_ORIGINAL).set_index(ID_COL)

    if drop_redundant:
        df = df.drop(columns=[c for c in REDUNDANT_COLS if c in df.columns])

    return df


def get_feature_lists(df: pd.DataFrame):
    """Devuelve (columnas_numericas, columnas_categoricas) sin el target."""
    categoricas = [c for c in CATEGORICAL_COLS if c in df.columns]
    numericas = [c for c in df.columns if c not in categoricas + [TARGET]]
    return numericas, categoricas


# --------------------------------------------------------------------------
# Split y preprocesamiento
# --------------------------------------------------------------------------
def build_preprocessor(numericas, categoricas, scale: bool = True) -> ColumnTransformer:
    """ColumnTransformer: escala numéricas (opcional) y codifica categóricas.

    - SVM necesita `scale=True`.
    - Los ensambles basados en árboles no lo necesitan, pero no les afecta.
    """
    num = StandardScaler() if scale else "passthrough"
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    pre = ColumnTransformer(
        transformers=[("num", num, numericas), ("cat", cat, categoricas)],
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")
    return pre


@dataclass
class DataBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    numericas: list
    categoricas: list

    def preprocessor(self, scale: bool = True) -> ColumnTransformer:
        """Preprocesador nuevo (sin ajustar) para usar dentro de un Pipeline."""
        return build_preprocessor(self.numericas, self.categoricas, scale=scale)


def prepare_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> DataBundle:
    """Split estratificado train/test.

    El preprocesador NO se ajusta aquí: debe ir dentro de un Pipeline para que
    se ajuste solo con los datos de entrenamiento de cada fold (evita fuga de
    información).
    """
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    numericas, categoricas = get_feature_lists(df)
    return DataBundle(X_train, X_test, y_train, y_test, numericas, categoricas)