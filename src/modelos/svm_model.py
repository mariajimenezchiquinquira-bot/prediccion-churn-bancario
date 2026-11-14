# --------------------------------------------------------------------------
# Modelo SVM (Support Vector Machine) para predecir churn.
#
# Flujo típico desde el notebook:
#
#     from src.modelos import svm_model as svm
#     busqueda = svm.buscar_hiperparametros(datos)
#     svm.resumen_busqueda(busqueda)
#     modelo = svm.entrenar_final(datos, svm.mejores_parametros(busqueda))
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.modelos import busqueda as bq
from src.datos.data_prep import DataBundle, RANDOM_STATE, build_preprocessor
from src.utils import guardar_y_mostrar

# Malla de hiperparámetros:
# - kernel rbf: C (penalización de errores) y gamma (alcance de cada punto).
# - kernel linear: solo C (no tiene gamma).
# - kernel poly: grado del polinomio, C y gamma.
# - kernel sigmoid: C y gamma.
# - class_weight: "balanced" (más peso a la clase minoritaria) o None (sin peso).
#   Se prueba porque el umbral se ajusta después, así que el peso puede no hacer falta.
PESOS = ["balanced", None]
PARAM_GRID = [
    {
        "svc__kernel": ["rbf"],
        "svc__C": [1, 5, 20, 50, 100],
        "svc__gamma": ["scale", 0.01, 0.03, 0.1],
        "svc__class_weight": PESOS,
    },
    {
        "svc__kernel": ["linear"],
        "svc__C": [0.01, 0.1, 1],
        "svc__class_weight": PESOS,
    },
    {
        "svc__kernel": ["poly"],
        "svc__degree": [2, 3],
        "svc__C": [1, 5],
        "svc__gamma": ["scale"],
        "svc__class_weight": PESOS,
    },
    {
        "svc__kernel": ["sigmoid"],
        "svc__C": [1, 5],
        "svc__gamma": ["scale"],
        "svc__class_weight": PESOS,
    },
]


# Creo el SVC con los valores por defecto del proyecto (rbf, C=1,
# class_weight balanced) + params.
#
# Lo usan este módulo y el Stacking de ensemble_models, para definir el SVM
# en un solo lugar.
def crear_svc(**params) -> SVC:
    base = dict(kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced", random_state=RANDOM_STATE)
    return SVC(**{**base, **params})


# Armo el pipeline: preprocesamiento (con escalado) + SVC.
#
# Con calibrar=True envuelvo el SVC en CalibratedClassifierCV (sigmoide de
# Platt), que habilita predict_proba. No uso SVC(probability=True) porque
# está deprecado desde scikit-learn 1.9.
#
# El escalado es obligatorio en SVM: sin él, variables como credit_limit
# (miles) dominarían a otras como dependent_count (0 a 5).
#
# class_weight='balanced' penaliza más los errores sobre la clase
# minoritaria.
def construir_pipeline(datos: DataBundle, calibrar: bool = False, **params_svc) -> Pipeline:
    svc = crear_svc(**params_svc)
    if calibrar:
        svc = CalibratedClassifierCV(svc, method="sigmoid", cv=5, ensemble=False)
    return Pipeline([
        ("pre", datos.preprocessor(scale=True)),
        ("svc", svc),
    ])


# Hago un grid search con validación cruzada estratificada solo sobre el
# entrenamiento
def buscar_hiperparametros(datos: DataBundle, param_grid=None, cv: int = 5,
                           n_jobs: int = -1, verbose: int = 1) -> GridSearchCV:
    return bq.ejecutar_busqueda(GridSearchCV, construir_pipeline(datos), datos, cv, n_jobs,
                                verbose, param_grid=param_grid or PARAM_GRID)


# Devuelvo los mejores hiperparámetros sin el prefijo 'svc__'
def mejores_parametros(busqueda: GridSearchCV) -> dict:
    return bq.mejores_parametros(busqueda, "svc__")


# Muestro las mejores combinaciones según PR-AUC promedio en validación
# cruzada
def resumen_busqueda(busqueda: GridSearchCV, top: int = 10) -> pd.DataFrame:
    r = pd.DataFrame(busqueda.cv_results_)
    cols = {
        "param_svc__kernel": "kernel",
        "param_svc__C": "C",
        "param_svc__gamma": "gamma",
        "param_svc__degree": "grado",
        "param_svc__class_weight": "class_weight",
        "mean_test_pr_auc": "pr_auc",
        "std_test_pr_auc": "pr_auc_std",
        "mean_test_roc_auc": "roc_auc",
        "mean_test_recall": "recall",
        "mean_test_f1": "f1",
        "mean_fit_time": "seg_ajuste",
    }
    out = r[[c for c in cols if c in r]].rename(columns=cols)
    for c in ("gamma", "grado"):
        if c in out:
            out[c] = out[c].fillna("-")
    if "class_weight" in out:
        out["class_weight"] = out["class_weight"].fillna("ninguno")
    return out.sort_values("pr_auc", ascending=False).head(top).round(4).reset_index(drop=True)


# Reentreno con los mejores parámetros sobre todo el entrenamiento, con el
# SVC calibrado para que sus probabilidades se lean como probabilidad real
# de fuga
def entrenar_final(datos: DataBundle, params: dict) -> Pipeline:
    return bq.entrenar_final(construir_pipeline(datos, calibrar=True, **params), datos)


# Dibujo el PR-AUC en validación cruzada de cada combinación (C, gamma) del
# kernel RBF, con el class_weight de la mejor combinación de la búsqueda
def plot_heatmap_cv(busqueda: GridSearchCV, guardar: bool = True):
    r = pd.DataFrame(busqueda.cv_results_)
    cw = busqueda.best_params_.get("svc__class_weight")
    col = r["param_svc__class_weight"]
    r = r[(r["param_svc__kernel"] == "rbf") & (col.isna() if cw is None else col == cw)].copy()
    r["gamma"] = r["param_svc__gamma"].astype(str)
    r["C"] = r["param_svc__C"].astype(float)
    tabla = r.pivot(index="C", columns="gamma", values="mean_test_pr_auc").astype(float)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(tabla, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    # Marco la mejor combinación de la búsqueda con un recuadro rojo
    mejor = busqueda.best_params_
    if mejor.get("svc__kernel") == "rbf":
        fila = list(tabla.index).index(float(mejor["svc__C"]))
        colu = list(tabla.columns).index(str(mejor["svc__gamma"]))
        ax.add_patch(plt.Rectangle((colu, fila), 1, 1, fill=False, edgecolor="red", lw=3))
    ax.set(title=f"SVM RBF: PR-AUC en validación cruzada (class_weight={cw})",
           xlabel="gamma", ylabel="C")
    guardar_y_mostrar(fig, "svm/heatmap_cv", guardar)


# Muestro la mejor combinación de cada kernel, según PR-AUC promedio en
# validación cruzada
def tabla_por_kernel(busqueda: GridSearchCV) -> pd.DataFrame:
    r = pd.DataFrame(busqueda.cv_results_)
    mejor = r.loc[r.groupby("param_svc__kernel")["mean_test_pr_auc"].idxmax()]
    out = pd.DataFrame({
        "kernel": mejor["param_svc__kernel"].values,
        "pr_auc": mejor["mean_test_pr_auc"].values,
        "pr_auc_std": mejor["std_test_pr_auc"].values,
        "roc_auc": mejor["mean_test_roc_auc"].values,
        "recall": mejor["mean_test_recall"].values,
        "f1": mejor["mean_test_f1"].values,
        "mejores_parametros": [", ".join(f"{k.replace('svc__', '')}={v}" for k, v in p.items()
                                         if k != "svc__kernel") for p in mejor["params"]],
    })
    return out.sort_values("pr_auc", ascending=False).round(4).reset_index(drop=True)


# Dibujo el mejor PR-AUC de cada kernel, con su desviación en validación
# cruzada
def plot_kernels(busqueda: GridSearchCV, guardar: bool = True):
    t = tabla_por_kernel(busqueda)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(t["kernel"], t["pr_auc"], yerr=t["pr_auc_std"], capsize=4, color="#4C72B0",
           label="PR-AUC (barra: desviación)")
    for x, v in enumerate(t["pr_auc"]):
        ax.text(x, 0.02, f"{v:.3f}", ha="center", color="white", fontsize=9)
    ax.set(title="SVM: mejor PR-AUC por kernel", xlabel="Kernel",
           ylabel="PR-AUC en validación cruzada", ylim=(0, 1))
    ax.legend(loc="upper right")
    guardar_y_mostrar(fig, "svm/kernels", guardar)


# Demuestro el SVM con más de dos clases (uno contra uno) sobre `variable`.
#
# No es el problema del proyecto (el churn es binario), solo muestra la
# técnica. Uno las clases muy pequeñas y evalúo con validación cruzada de 5
# partes.
def demo_multiclase(datos: DataBundle, df: pd.DataFrame, variable: str = "card_category",
                    unir: tuple = ("Gold", "Platinum"), guardar: bool = True) -> pd.DataFrame:
    X = pd.concat([datos.X_train, datos.X_test]).drop(columns=variable)
    y = df.loc[X.index, variable].replace({c: "/".join(unir) for c in unir})
    pre = build_preprocessor(datos.numericas, [c for c in datos.categoricas if c != variable])
    modelo = Pipeline([("pre", pre),
                       ("svc", SVC(kernel="rbf", C=5, class_weight="balanced",
                                   decision_function_shape="ovo"))])
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    pred = cross_val_predict(modelo, X, y, cv=cv, n_jobs=-1)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ConfusionMatrixDisplay.from_predictions(y, pred, ax=ax, cmap="Blues", colorbar=False)
    ax.set(title=f"SVM con 3 clases: {variable} (validación cruzada)",
           xlabel="Clase predicha", ylabel="Clase real")
    guardar_y_mostrar(fig, "svm/multiclase", guardar)
    tabla = pd.DataFrame(classification_report(y, pred, output_dict=True)).T
    tabla = tabla.drop(["accuracy", "weighted avg"]).round(3)
    tabla["support"] = tabla["support"].astype(int)
    return tabla.rename(columns={"precision": "precisión", "support": "clientes"})
