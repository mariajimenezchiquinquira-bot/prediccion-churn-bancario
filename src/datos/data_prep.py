# --------------------------------------------------------------------------
# Carga, limpieza y preparación de datos
#
# Uso típico desde el notebook:
#     from src.datos import data_prep as dp
#     df = dp.clean_data(dp.load_data())
#     datos = dp.prepare_data(df)          # split estratificado 80/20
# --------------------------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from IPython.display import display
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
TARGET = "churn"                  # 1 = cliente se fue, 0 = permaneció
POSITIVE_LABEL = "Attrited Customer"
NEGATIVE_LABEL = "Existing Customer"

CATEGORICAL_COLS = [
    "gender",
    "education_level",
    "marital_status",
    "income_category",
    "card_category",
]

REDUNDANT_COLS = ["avg_open_to_buy", "avg_utilization_ratio"]

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Carga y limpieza
# --------------------------------------------------------------------------

def load_data(path=DATA_PATH) -> pd.DataFrame:
    # Leo el Excel original sin modificarlo
    path = Path(path)

    # Reviso que el archivo exista, si no, aviso dónde debe estar
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo en {path}. "
            "Verifica que bank_churn.xlsx esté en data/raw/."
        )
    return pd.read_excel(path)


# --------------------------------------------------------------------------
# verificar_valores_numericos: reviso la calidad de las variables numéricas
# - Muestro el mínimo y el máximo de cada variable.
# - Si alguna tiene valores negativos, me detengo con un error, porque en este
#   negocio (cupos, saldos, montos y conteos) no tendrían sentido.
# - Los valores muy altos NO se eliminan: son clientes reales.
# --------------------------------------------------------------------------
def verificar_valores_numericos(df: pd.DataFrame) -> pd.DataFrame:
    # Tomo solo las variables numéricas (sin el ID del cliente)
    numericas = df.select_dtypes("number").drop(columns=[ID_COL], errors="ignore")

    # Calculo el rango (mínimo y máximo) de cada variable
    rango = numericas.agg(["min", "max"]).T
    rango.columns = ["minimo", "maximo"]

    # Si hay valores negativos, me detengo y aviso en cuáles variables
    con_negativos = rango.index[rango["minimo"] < 0].tolist()
    if con_negativos:
        raise ValueError(f"Hay valores negativos en: {con_negativos}")

    print("Valores negativos encontrados: 0")
    return rango


# --------------------------------------------------------------------------
# clean_data: limpio el dataset y creo la variable objetivo binaria `churn`
#
# Decisiones:
# - `attrition_flag` se convierte a 0/1 (1 = Attrited Customer).
# - `clientnum` pasa a ser el índice
# - Las categorías "Unknown" se conservan como una categoría propia, ya que
#   eliminarlas descartaría entre 7% y 
# 15% de los clientes.
# --------------------------------------------------------------------------

def clean_data(df: pd.DataFrame, drop_redundant: bool = True) -> pd.DataFrame:
    # Trabajo sobre una copia para no tocar los datos originales
    df = df.copy()

    # Dejo los nombres de las variables en minúscula y sin espacios
    df.columns = df.columns.str.strip().str.lower()

    # Reviso que estén las variables clave (ID y variable objetivo)
    faltantes = {ID_COL, TARGET_ORIGINAL} - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas esperadas: {faltantes}")

    # Me aseguro que la variable objetivo solo contenga los dos textos esperados
    valores = set(df[TARGET_ORIGINAL].dropna().unique())
    if not valores <= {POSITIVE_LABEL, NEGATIVE_LABEL}:
        raise ValueError(f"Valores inesperados en {TARGET_ORIGINAL}: {valores}")

    # Quito espacios sobrantes en las variables categóricas
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("string").str.strip()

    # Duplicados por cliente (informo cuántos registros se eliminan)
    antes = len(df)
    df = df.drop_duplicates(subset=ID_COL)
    print(f"Registros duplicados eliminados: {antes - len(df)}")

    # Reviso que no haya valores negativos y muestro el rango de cada variable
    display(verificar_valores_numericos(df))

    # Variable objetivo: 1 = cliente se fue, 0 = cliente se quedó
    df[TARGET] = (df[TARGET_ORIGINAL] == POSITIVE_LABEL).astype(int)

    # Quito la variable original y dejo el ID del cliente como índice
    df = df.drop(columns=TARGET_ORIGINAL).set_index(ID_COL)

    # Quito las variables redundantes
    if drop_redundant:
        df = df.drop(columns=[c for c in REDUNDANT_COLS if c in df.columns])

    return df


def get_feature_lists(df: pd.DataFrame):
    # Devuelvo dos listas: variables numéricas y variables categóricas (sin la objetivo)
    categoricas = [c for c in CATEGORICAL_COLS if c in df.columns]
    numericas = [c for c in df.columns if c not in categoricas + [TARGET]]
    return numericas, categoricas


# --------------------------------------------------------------------------
# Split y preprocesamiento
# --------------------------------------------------------------------------

# build_preprocessor: escala las numéricas (opcional) y codifica las categóricas
# - El SVM necesita scale=True.
# - Los ensambles basados en árboles no lo necesitan, pero no les afecta.

def build_preprocessor(numericas, categoricas, scale: bool = True) -> ColumnTransformer:
    # Numéricas: las escalo si scale=True, si no las dejo pasar igual
    num = StandardScaler() if scale else "passthrough"

    # Categóricas: one-hot (una variable 0/1 por cada categoría)
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # Junto las dos transformaciones en un solo preprocesador
    pre = ColumnTransformer(
        transformers=[("num", num, numericas), ("cat", cat, categoricas)],
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")
    return pre


# DataBundle: paquete que guarda juntos los datos de entrenamiento, los de
# prueba y las listas de variables
@dataclass
class DataBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    numericas: list
    categoricas: list

    def preprocessor(self, scale: bool = True) -> ColumnTransformer:
        # Devuelvo un preprocesador nuevo (sin ajustar) para usar dentro de un Pipeline
        return build_preprocessor(self.numericas, self.categoricas, scale=scale)


# prepare_data: hago el split estratificado train/test (80/20)
# El preprocesador NO se ajusta aquí: va dentro de un Pipeline para que se
# ajuste solo con los datos de entrenamiento de cada fold (evita fuga de información).
def prepare_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> DataBundle:
    # Separo las variables (X) de la variable objetivo (y)
    X = df.drop(columns=TARGET)
    y = df[TARGET]

    # Divido en entrenamiento y prueba, manteniendo la misma proporción de churn
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Guardo las listas de variables y devuelvo todo en el paquete
    numericas, categoricas = get_feature_lists(df)
    return DataBundle(X_train, X_test, y_train, y_test, numericas, categoricas)