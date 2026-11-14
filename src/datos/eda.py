# --------------------------------------------------------------------------
# Funciones del análisis exploratorio (EDA) del proyecto de churn.
#
# Cada gráfico se guarda en reports/figures/ (para el informe) y también se
# muestra en el notebook.
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.datos.data_prep import CATEGORICAL_COLS, TARGET
from src.utils import guardar_y_mostrar

ETIQUETAS = {0: "Permanece", 1: "Se fue"}


# --------------------------------------------------------------------------
# Tablas
# --------------------------------------------------------------------------
# Resumo el tipo, los nulos y los valores únicos de cada variable
def resumen_general(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tipo": df.dtypes.astype(str),
            "nulos": df.isna().sum(),
            "unicos": df.nunique(),
        }
    )


# Cuento los 'Unknown' de cada variable categórica y comparo su tasa de
# churn con la del resto
def tabla_unknown(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for col in CATEGORICAL_COLS:
        es_unk = df[col] == "Unknown"
        if es_unk.any():
            filas.append(
                {
                    "variable": col,
                    "n_unknown": int(es_unk.sum()),
                    "pct_unknown": round(100 * es_unk.mean(), 1),
                    "churn_unknown_%": round(100 * df.loc[es_unk, TARGET].mean(), 1),
                    "churn_resto_%": round(100 * df.loc[~es_unk, TARGET].mean(), 1),
                }
            )
    return pd.DataFrame(filas)


# Calculo el número de clientes y la tasa de churn (%) de cada categoría
def tabla_churn_categorias(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for col in CATEGORICAL_COLS:
        g = df.groupby(col, observed=True)[TARGET].agg(["size", "mean"])
        filas.append(pd.DataFrame({
            "variable": col, "categoria": g.index.astype(str),
            "clientes": g["size"].values, "churn_%": (100 * g["mean"]).round(1).values,
        }))
    return pd.concat(filas, ignore_index=True)


# Calculo la correlación de cada variable numérica con churn, de mayor a
# menor magnitud
def correlacion_con_target(df: pd.DataFrame) -> pd.Series:
    corr = df.select_dtypes("number").corr()[TARGET].drop(TARGET)
    return corr.sort_values(key=abs, ascending=False).round(3)


# --------------------------------------------------------------------------
# Gráficos
# --------------------------------------------------------------------------
def plot_balance_clases(df: pd.DataFrame, guardar: bool = True):
    conteo = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    barras = ax.bar([ETIQUETAS[i] for i in conteo.index], conteo.values,
                    color=["#4C72B0", "#C44E52"])
    for b, v in zip(barras, conteo.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v / len(df):.1%})",
                ha="center", va="bottom")
    ax.set_title("Balance de clases")
    ax.set_ylabel("Clientes")
    ax.set_ylim(0, conteo.max() * 1.18)
    guardar_y_mostrar(fig, "eda/balance_clases", guardar)


# Variables continuas (las de pocos valores se ven mejor en plot_churn_por_valor)
VARS_CONTINUAS = ["customer_age", "months_on_book", "credit_limit", "total_revolving_bal",
                  "total_amt_chng_q4_q1", "total_trans_amt", "total_trans_ct",
                  "total_ct_chng_q4_q1"]


# Dibujo un boxplot de cada variable continua, separado por churn
def plot_numericas_por_churn(df: pd.DataFrame, cols=VARS_CONTINUAS, guardar: bool = True):
    n_cols = 4
    n_filas = -(-len(cols) // n_cols)
    fig, axes = plt.subplots(n_filas, n_cols, figsize=(4 * n_cols, 3.2 * n_filas))
    for ax, col in zip(axes.ravel(), cols):
        sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, legend=False,
                    palette=["#4C72B0", "#C44E52"], ax=ax)
        ax.set_xticks([0, 1], [ETIQUETAS[0], ETIQUETAS[1]])
        ax.set_xlabel("")
        ax.set_title(col, fontsize=10)
    for ax in axes.ravel()[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Variables numéricas según churn", y=1.01)
    fig.tight_layout()
    guardar_y_mostrar(fig, "eda/numericas_por_churn", guardar)


# Dibujo la tasa de churn de cada categoría, con la tasa global como
# referencia y el número de clientes (n)
def plot_categoricas_churn(df: pd.DataFrame, guardar: bool = True):
    cols = CATEGORICAL_COLS
    global_rate = df[TARGET].mean()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), cols):
        t = df.groupby(col, observed=True)[TARGET].agg(["mean", "size"]).sort_values("mean")
        # "$" se escapa para que matplotlib no lo lea como fórmula (ej. "$40K - $60K")
        etiquetas = [f"{c} (n={n})".replace("$", r"\$") for c, n in zip(t.index.astype(str), t["size"])]
        sns.barplot(x=t["mean"].values, y=etiquetas, color="#8172B3", ax=ax)
        ax.axvline(global_rate, color="red", linestyle="--", label=f"Global {global_rate:.1%}")
        ax.set_title(f"Churn por {col}", fontsize=10)
        ax.set_xlabel("Tasa de churn")
        ax.set_ylabel("")
        ax.set_xlim(0, t["mean"].max() * 1.6)   # espacio a la derecha para la leyenda
        ax.legend(fontsize=8, loc="lower right")
    for ax in axes.ravel()[len(cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    guardar_y_mostrar(fig, "eda/categoricas_churn", guardar)


def plot_correlaciones(df: pd.DataFrame, guardar: bool = True):
    corr = df.select_dtypes("number").corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # solo el triángulo de abajo
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 8}, ax=ax)
    ax.set_title("Matriz de correlación")
    guardar_y_mostrar(fig, "eda/correlaciones", guardar)


# --------------------------------------------------------------------------
# Churn por cada valor (variables con pocos valores) y transacciones
# --------------------------------------------------------------------------
VARS_POCOS_VALORES = ["months_inactive_12_mon", "contacts_count_12_mon",
                      "total_relationship_count"]


# Dibujo la tasa de churn de cada valor de la variable, con el número de
# clientes (n)
def plot_churn_por_valor(df: pd.DataFrame, cols=VARS_POCOS_VALORES, guardar: bool = True):
    tasa_global = df[TARGET].mean()
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4), sharey=True)
    for ax, col in zip(axes, cols):
        t = df.groupby(col)[TARGET].agg(["mean", "size"])
        ax.bar(t.index.astype(str), t["mean"], color="#C44E52")
        for x, (m, n) in enumerate(zip(t["mean"], t["size"])):
            ax.text(x, m, f"{m:.0%}\nn={n}", ha="center", va="bottom", fontsize=8)
        ax.axhline(tasa_global, color="gray", linestyle="--", label=f"Global {tasa_global:.1%}")
        ax.set(title=col, xlabel="Valor", ylim=(0, 1.2))
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Tasa de churn")
    fig.suptitle("Tasa de churn por cada valor", y=1.02)
    fig.tight_layout()
    guardar_y_mostrar(fig, "eda/churn_por_valor", guardar)


# Comparo la distribución del número de transacciones (total_trans_ct) entre
# quienes se fueron y quienes se quedaron
def plot_distribucion_transacciones(df: pd.DataFrame, guardar: bool = True):
    fig, ax = plt.subplots(figsize=(7, 4))
    for valor, color in [(0, "#4C72B0"), (1, "#C44E52")]:
        x = df.loc[df[TARGET] == valor, "total_trans_ct"]
        ax.hist(x, bins=30, density=True, alpha=0.6, color=color,
                label=f"{ETIQUETAS[valor]} (mediana {x.median():.0f})")
    ax.set(title="Número de transacciones según churn", xlabel="total_trans_ct",
           ylabel="Densidad")
    ax.legend()
    guardar_y_mostrar(fig, "eda/distribucion_transacciones", guardar)
