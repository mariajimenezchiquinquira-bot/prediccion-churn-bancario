# --------------------------------------------------------------------------
# Métricas, selección de umbral y gráficos comunes a todos los modelos.
#
# Cada modelo evaluado produce un diccionario de resultado (ver
# `evaluar_modelo`). Con varios resultados en {nombre: resultado} comparo
# tablas y curvas de cualquier modelo.
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.datos.data_prep import RANDOM_STATE
from src.utils import guardar_y_mostrar


# --------------------------------------------------------------------------
# Puntajes y umbrales
# --------------------------------------------------------------------------
# Probabilidad de la clase 'se fue' (todos los modelos del proyecto tienen
# predict_proba)
def puntaje(modelo, X) -> np.ndarray:
    return modelo.predict_proba(X)[:, 1]


# Elijo el umbral de decisión:
# - 'f1': el que maximiza F1.
# - 'recall': el más alto (más preciso) que aún alcanza `recall_objetivo`.
def elegir_umbral(y, score, criterio: str = "f1", recall_objetivo: float = 0.80) -> float:
    prec, rec, thr = precision_recall_curve(y, score)
    prec, rec = prec[:-1], rec[:-1]
    if criterio == "f1":
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
        return float(thr[np.argmax(f1)])
    if criterio == "recall":
        ok = rec >= recall_objetivo
        if not ok.any():
            raise ValueError(f"Ningún umbral alcanza recall >= {recall_objetivo}")
        return float(thr[ok].max())
    raise ValueError("criterio debe ser 'f1' o 'recall'")


# Probabilidades de la clase 'se fue' fuera de muestra (validación cruzada)
def _puntajes_cv(modelo, X, y, cv: int = 5, n_jobs: int = -1) -> np.ndarray:
    skf = StratifiedKFold(cv, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(modelo, X, y, cv=skf, method="predict_proba", n_jobs=n_jobs)[:, 1]


# Elijo el umbral con predicciones fuera de muestra (CV) del entrenamiento,
# para no ajustarlo mirando la prueba
def umbral_por_cv(modelo, X, y, cv: int = 5, criterio: str = "f1",
                  recall_objetivo: float = 0.80, n_jobs: int = -1) -> float:
    return elegir_umbral(y, _puntajes_cv(modelo, X, y, cv, n_jobs), criterio, recall_objetivo)


# --------------------------------------------------------------------------
# Evaluación
# --------------------------------------------------------------------------
# Evalúo un modelo ya entrenado y devuelvo un diccionario de resultado
def evaluar_modelo(nombre: str, modelo, X, y, umbral: float = 0.5) -> dict:
    y = np.asarray(y)
    score = puntaje(modelo, X)
    pred = (score >= umbral).astype(int)
    metricas = {
        "accuracy": accuracy_score(y, pred),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
    }
    return {"nombre": nombre, "umbral": umbral, "y_true": y,
            "y_score": score, "y_pred": pred, "metricas": metricas}


# Entreno el modelo con `entrenar()` y elijo su umbral con CV del
# entrenamiento. Devuelvo (modelo, umbral).
#
# Es la parte lenta, por eso se guarda con utils.cacheado.
def entrenar_con_umbral(entrenar, datos, n_jobs: int = -1):
    modelo = entrenar()
    return modelo, umbral_por_cv(modelo, datos.X_train, datos.y_train, n_jobs=n_jobs)


# Elijo el umbral con CV del entrenamiento (si no lo recibo), evalúo en
# prueba y guardo el modelo y su resultado
def registrar_modelo(nombre: str, modelo, datos, resultados: dict, modelos: dict,
                     n_jobs: int = -1, umbral: float | None = None) -> None:
    if umbral is None:
        umbral = umbral_por_cv(modelo, datos.X_train, datos.y_train, n_jobs=n_jobs)
    resultados[nombre] = evaluar_modelo(nombre, modelo, datos.X_test, datos.y_test, umbral)
    modelos[nombre] = modelo


# Armo la tabla con las métricas de todos los modelos, ordenada por PR-AUC
# (el mejor arriba)
def tabla_comparativa(resultados: dict) -> pd.DataFrame:
    filas = {n: {**r["metricas"], "umbral": r["umbral"]} for n, r in resultados.items()}
    return pd.DataFrame(filas).T.sort_values("pr_auc", ascending=False).round(3)


# Ordeno a los clientes por riesgo y mido cuántas fugas captura cada decil.
#
# Responde: si el banco solo puede contactar al 10% de los clientes, ¿qué
# porcentaje de las fugas reales alcanza?
def tabla_lift(resultado: dict, n_bins: int = 10) -> pd.DataFrame:
    y, s = resultado["y_true"], resultado["y_score"]
    orden = np.argsort(-s, kind="stable")
    y_ord = y[orden]
    base = y.mean()
    filas, acum = [], 0
    for i, idx in enumerate(np.array_split(np.arange(len(y)), n_bins), start=1):
        fugas = int(y_ord[idx].sum())
        acum += fugas
        filas.append({
            "decil": i,
            "clientes": len(idx),
            "fugas": fugas,
            "tasa_churn_%": round(100 * fugas / len(idx), 1),
            "captura_acum_%": round(100 * acum / y.sum(), 1),
            "lift": round((fugas / len(idx)) / base, 2),
        })
    return pd.DataFrame(filas)


# Calculo la importancia por permutación sobre las variables originales
# (antes del pipeline)
def importancia_permutacion(modelo, X, y, scoring: str = "average_precision",
                            n_repeats: int = 5, n_jobs: int = -1) -> pd.DataFrame:
    r = permutation_importance(modelo, X, y, scoring=scoring, n_repeats=n_repeats,
                               random_state=RANDOM_STATE, n_jobs=n_jobs)
    out = pd.DataFrame({"variable": X.columns,
                        "caida_media": r.importances_mean,
                        "desv": r.importances_std})
    return out.sort_values("caida_media", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Gráficos
# --------------------------------------------------------------------------
# Dibujo las curvas ROC y Precisión-Recall de uno o varios modelos
def plot_curvas(resultados: dict, guardar: bool = True, archivo: str = "svm/curvas_roc_pr"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for nombre, r in resultados.items():
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_score"])
        ax1.plot(fpr, tpr, label=f"{nombre} (AUC={r['metricas']['roc_auc']:.3f})")
        p, rc, _ = precision_recall_curve(r["y_true"], r["y_score"])
        ax2.plot(rc, p, label=f"{nombre} (PR-AUC={r['metricas']['pr_auc']:.3f})")
    base = next(iter(resultados.values()))["y_true"].mean()
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set(title="Curva ROC", xlabel="Tasa de falsos positivos",
            ylabel="Tasa de verdaderos positivos")
    ax2.axhline(base, color="k", linestyle="--", alpha=0.5, label=f"Azar ({base:.3f})")
    ax2.set(title="Curva Precisión-Recall", xlabel="Recall", ylabel="Precisión")
    ax1.legend(loc="lower right"); ax2.legend(loc="lower left")
    fig.tight_layout()
    guardar_y_mostrar(fig, archivo, guardar)


def plot_matriz_confusion(resultado: dict, guardar: bool = True, archivo: str | None = None):
    cm = confusion_matrix(resultado["y_true"], resultado["y_pred"])
    pct = cm / cm.sum(axis=1, keepdims=True)
    etiquetas = np.array([[f"{cm[i, j]}\n({pct[i, j]:.1%})" for j in range(2)] for i in range(2)])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=etiquetas, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Permanece", "Se fue"], yticklabels=["Permanece", "Se fue"])
    ax.set(title=f"Matriz de confusión - {resultado['nombre']}\n(umbral={resultado['umbral']:.3f})",
           xlabel="Predicho", ylabel="Real")
    guardar_y_mostrar(fig, archivo or f"comparacion/confusion_{resultado['nombre'].lower().replace(' ', '_')}", guardar)


def plot_importancia(imp: pd.DataFrame, top: int = 12, guardar: bool = True,
                     archivo: str = "svm/importancia_permutacion"):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(d["variable"], d["caida_media"], xerr=d["desv"], color="#55A868")
    ax.set(title="Importancia por permutación (caída de PR-AUC)", xlabel="Caída media")
    fig.tight_layout()
    guardar_y_mostrar(fig, archivo, guardar)


# Dibujo la curva de calibración: probabilidad predicha contra proporción
# real de fugas.
#
# Cerca de la diagonal, la probabilidad se puede leer como probabilidad real
# de fuga. El Brier score va en la leyenda (menor es mejor).
def plot_calibracion(resultados: dict, guardar: bool = True, archivo: str = "comparacion/calibracion"):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Calibración perfecta")
    for nombre, r in resultados.items():
        real, pred = calibration_curve(r["y_true"], r["y_score"], n_bins=8, strategy="quantile")
        brier = brier_score_loss(r["y_true"], r["y_score"])
        ax.plot(pred, real, marker="o", label=f"{nombre} (Brier {brier:.3f})")
    ax.set(title="Curva de calibración (set de prueba)",
           xlabel="Probabilidad predicha de fuga", ylabel="Proporción real de fugas")
    ax.legend(fontsize=8)
    guardar_y_mostrar(fig, archivo, guardar)


# --------------------------------------------------------------------------
# Comparación entre modelos y decisión de negocio
# --------------------------------------------------------------------------
# Calculo PR-AUC y F1 de cada modelo con su intervalo de confianza
# (bootstrap en la prueba).
#
# Remuestreo los clientes de prueba con reemplazo, los mismos para todos los
# modelos. Si los intervalos de dos modelos se traslapan, la diferencia
# entre ellos no es confiable.
def tabla_intervalos(resultados: dict, n_boot: int = 1000, nivel: float = 0.95) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    y = np.asarray(next(iter(resultados.values()))["y_true"])
    muestras = rng.integers(0, len(y), size=(n_boot, len(y)))
    inf, sup = 100 * (1 - nivel) / 2, 100 * (1 + nivel) / 2
    filas = {}
    for nombre, r in resultados.items():
        ap = [average_precision_score(y[i], r["y_score"][i]) for i in muestras]
        f1 = [f1_score(y[i], r["y_pred"][i], zero_division=0) for i in muestras]
        filas[nombre] = {
            "pr_auc": r["metricas"]["pr_auc"],
            "pr_auc_inf": np.percentile(ap, inf), "pr_auc_sup": np.percentile(ap, sup),
            "f1": r["metricas"]["f1"],
            "f1_inf": np.percentile(f1, inf), "f1_sup": np.percentile(f1, sup),
        }
    return pd.DataFrame(filas).T.sort_values("pr_auc", ascending=False).round(3)


# Dibujo el PR-AUC de cada modelo con su intervalo de confianza del 95%
# (sale de tabla_intervalos)
def plot_intervalos(tabla: pd.DataFrame, guardar: bool = True, archivo: str = "comparacion/intervalos_confianza"):
    d = tabla.iloc[::-1]
    pos = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(d["pr_auc"], pos, xerr=[d["pr_auc"] - d["pr_auc_inf"], d["pr_auc_sup"] - d["pr_auc"]],
                fmt="o", capsize=4, color="#4C72B0", label="PR-AUC e intervalo de confianza del 95%")
    ax.set_yticks(pos, d.index)
    ax.set(title="PR-AUC por modelo con intervalo de confianza (set de prueba)",
           xlabel="PR-AUC", ylabel="Modelo", ylim=(-0.6, len(d) - 0.4))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
    fig.tight_layout()
    guardar_y_mostrar(fig, archivo, guardar)


# Dibujo la curva de ganancia acumulada: % de fugas capturadas al contactar
# al X% de clientes de mayor riesgo
def plot_ganancia(resultados: dict, guardar: bool = True, archivo: str = "comparacion/curva_ganancia"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for nombre, r in resultados.items():
        y = np.asarray(r["y_true"])
        captura = np.cumsum(y[np.argsort(-r["y_score"], kind="stable")]) / y.sum()
        ax.plot(100 * np.arange(1, len(y) + 1) / len(y), 100 * captura, label=nombre)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.5, label="Al azar")
    for x in (10, 20, 30):
        ax.axvline(x, color="gray", linestyle=":", alpha=0.5)
    ax.set(title="Curva de ganancia acumulada (set de prueba)",
           xlabel="% de clientes contactados (de mayor a menor riesgo)",
           ylabel="% de fugas capturadas")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    guardar_y_mostrar(fig, archivo, guardar)


# Comparo umbrales de decisión: a cuántos clientes se contacta y cuántas
# fugas se detectan.
#
# Los umbrales se eligen con CV del entrenamiento (no con la prueba): el que
# maximiza F1 y el más alto que alcanza cada recall objetivo.
def tabla_umbrales(modelo, datos, recalls=(0.80, 0.90), n_jobs: int = -1) -> pd.DataFrame:
    y_tr = np.asarray(datos.y_train)
    s_cv = _puntajes_cv(modelo, datos.X_train, y_tr, n_jobs=n_jobs)
    umbrales = {"Máximo F1": elegir_umbral(y_tr, s_cv, "f1")}
    for r in recalls:
        umbrales[f"Recall objetivo {r:.0%}"] = elegir_umbral(y_tr, s_cv, "recall", r)
    y, s = np.asarray(datos.y_test), puntaje(modelo, datos.X_test)
    filas = {}
    for nombre, u in umbrales.items():
        pred = s >= u
        contactados, detectadas = int(pred.sum()), int((pred & (y == 1)).sum())
        filas[nombre] = {
            "umbral": round(u, 3), "clientes_contactados": contactados,
            "contactados_%": round(100 * contactados / len(y), 1),
            "fugas_detectadas": detectadas, "recall": round(detectadas / y.sum(), 3),
            "precision": round(detectadas / max(contactados, 1), 3),
            "falsas_alarmas": contactados - detectadas,
        }
    return pd.DataFrame.from_dict(filas, orient="index")


# Metas de negocio del banco (se pueden cambiar). Cada meta dice si el indicador
# debe ser al menos ese valor (">=") o a lo sumo ese valor ("<=").
METAS = {
    "recall": (">=", 0.85, "Detectar al menos el 85% de las fugas"),
    "precision": (">=", 0.80, "Que al menos 8 de cada 10 contactados se iban a ir"),
    "contactados_%": ("<=", 20.0, "Contactar como máximo al 20% de los clientes"),
    "captura_20%": (">=", 90.0, "Con el 20% de mayor riesgo, alcanzar al menos el 90% de las fugas"),
    "pr_auc": (">=", 0.90, "Buena capacidad de ordenar clientes por riesgo (PR-AUC)"),
}


# Calculo los indicadores de negocio de un modelo en la prueba (con su
# umbral elegido)
def indicadores_negocio(resultado: dict) -> dict:
    y, s, pred = resultado["y_true"], resultado["y_score"], resultado["y_pred"]
    top = np.argsort(-s, kind="stable")[: int(round(0.20 * len(y)))]
    return {
        "recall": resultado["metricas"]["recall"],
        "precision": resultado["metricas"]["precision"],
        "contactados_%": 100 * pred.mean(),
        "captura_20%": 100 * y[top].sum() / y.sum(),
        "pr_auc": resultado["metricas"]["pr_auc"],
    }


# Devuelvo True si el valor cumple la meta del indicador
def cumple_meta(indicador: str, valor: float, metas: dict = METAS) -> bool:
    signo, meta, _ = metas[indicador]
    return valor >= meta if signo == ">=" else valor <= meta


# Comparo las metas del banco con el resultado de cada modelo (el primer
# renglón es la meta).
#
# La última columna cuenta cuántas metas cumple cada modelo.
def tabla_indicadores(resultados: dict, metas: dict = METAS) -> pd.DataFrame:
    filas = {"Meta": {k: f"{s} {m:g}" for k, (s, m, _) in metas.items()} | {"metas_cumplidas": f"{len(metas)} de {len(metas)}"}}
    for nombre, r in resultados.items():
        ind = indicadores_negocio(r)
        cumplidas = sum(cumple_meta(k, v, metas) for k, v in ind.items())
        filas[nombre] = {k: round(v, 3 if k in ("recall", "precision", "pr_auc") else 1)
                         for k, v in ind.items()} | {"metas_cumplidas": f"{cumplidas} de {len(metas)}"}
    return pd.DataFrame(filas).T
