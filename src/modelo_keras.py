# --------------------------------------------------------------------------
# Red neuronal de Keras con la misma interfaz que los modelos de scikit-
# learn.
#
# Existe porque las funciones del proyecto (umbral por CV, evaluación,
# curvas) esperan un modelo con fit y predict_proba. Esta clase envuelve una
# red de Keras para que funcione igual que un SVM o un LightGBM y se pueda
# usar dentro de un Pipeline, sin fuga de información.
#
# Uso típico desde el notebook:
#
#     from sklearn.pipeline import Pipeline
#     from src import modelo_keras as mk
#     red = Pipeline([("pre", datos.preprocessor()), ("red", mk.RedNeuronal())])
# --------------------------------------------------------------------------
import math

import keras
import numpy as np
import pandas as pd
from keras import layers, regularizers
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.datos.data_prep import RANDOM_STATE
from src.utils import guardar_y_mostrar


# Red feed-forward (keras.Sequential) para clasificación binaria.
#
# Parámetros principales:
# - capas: neuronas de cada capa oculta, por ejemplo (32, 16).
# - dropout: fracción de neuronas que se apaga en cada capa oculta
#   (regularización).
# - l2: castigo L2 a los pesos (regularización).
# - optimizador: "adam", "sgd" (con momentum y decaimiento) o "rmsprop".
# - paciencia: épocas sin mejorar antes de parar (parada temprana).
class RedNeuronal(ClassifierMixin, BaseEstimator):
    def __init__(self, capas=(32, 16), activacion="relu", dropout=0.0, l2=0.0,
                 optimizador="adam", lr=0.001, momentum=0.0, decaimiento=0.0,
                 epocas=100, batch_size=64, paciencia=10, random_state=42):
        self.capas = capas
        self.activacion = activacion
        self.dropout = dropout
        self.l2 = l2
        self.optimizador = optimizador
        self.lr = lr
        self.momentum = momentum
        self.decaimiento = decaimiento
        self.epocas = epocas
        self.batch_size = batch_size
        self.paciencia = paciencia
        self.random_state = random_state

    def _optimizador(self, pasos_por_epoca):
        # Si hay decaimiento, la tasa de aprendizaje baja un poco en cada época
        lr = self.lr
        if self.decaimiento > 0:
            lr = keras.optimizers.schedules.InverseTimeDecay(
                self.lr, decay_steps=pasos_por_epoca, decay_rate=self.decaimiento)
        if self.optimizador == "sgd":
            return keras.optimizers.SGD(lr, momentum=self.momentum)
        if self.optimizador == "rmsprop":
            return keras.optimizers.RMSprop(lr)
        if self.optimizador == "adam":
            return keras.optimizers.Adam(lr)
        raise ValueError("optimizador debe ser 'adam', 'sgd' o 'rmsprop'")

    def fit(self, X, y):
        keras.utils.set_random_seed(self.random_state)
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        self.classes_ = np.array([0, 1])

        modelo = keras.Sequential([keras.Input(shape=(X.shape[1],))])
        for n in self.capas:
            modelo.add(layers.Dense(
                n, activation=self.activacion,
                kernel_regularizer=regularizers.l2(self.l2) if self.l2 else None))
            if self.dropout:
                modelo.add(layers.Dropout(self.dropout))
        modelo.add(layers.Dense(1, activation="sigmoid"))

        pasos = math.ceil(len(X) * 0.85 / self.batch_size)
        modelo.compile(
            optimizer=self._optimizador(pasos),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")])

        # Separa 15% del entrenamiento para vigilar la PR-AUC y parar a tiempo
        parada = keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=self.paciencia,
            restore_best_weights=True)
        self.historia_ = modelo.fit(
            X, y, validation_split=0.15, epochs=self.epocas,
            batch_size=self.batch_size, callbacks=[parada], verbose=0)
        self.modelo_ = modelo
        return self

    def predict_proba(self, X):
        p = self.modelo_.predict(np.asarray(X, dtype="float32"), verbose=0).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------
# Funciones de apoyo para el notebook
# --------------------------------------------------------------------------
ESPACIO = {
    "red__capas": [(16,), (32,), (64,), (32, 16), (64, 32), (64, 32, 16)],
    "red__dropout": [0.0, 0.1, 0.2, 0.3],
    "red__l2": [0.0, 1e-4, 1e-3, 1e-2],
    "red__optimizador": ["adam", "rmsprop", "sgd"],
    "red__lr": [0.0003, 0.001, 0.003, 0.01],
    "red__momentum": [0.0, 0.9],
    "red__decaimiento": [0.0, 0.01],
    "red__batch_size": [32, 64, 128],
}


# Armo el preprocesador + la red, para que el escalado se aprenda solo con
# el entrenamiento
def construir_pipeline(datos, **params) -> Pipeline:
    return Pipeline([("pre", datos.preprocessor()), ("red", RedNeuronal(**params))])


# Entreno la red (la base si no hay parámetros) con todo el entrenamiento
def entrenar_final(datos, params=None) -> Pipeline:
    return construir_pipeline(datos, **(params or {})).fit(datos.X_train, datos.y_train)


# Hago una búsqueda al azar con validación cruzada solo sobre el
# entrenamiento (PR-AUC)
def buscar_hiperparametros(datos, n_iter: int = 15, cv: int = 5) -> RandomizedSearchCV:
    busqueda = RandomizedSearchCV(
        construir_pipeline(datos), ESPACIO, n_iter=n_iter, scoring="average_precision",
        cv=StratifiedKFold(cv, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE, n_jobs=1, verbose=1, refit=False)
    return busqueda.fit(datos.X_train, datos.y_train)


# Devuelvo los mejores hiperparámetros sin el prefijo 'red__'
def mejores_parametros(busqueda) -> dict:
    return {k.replace("red__", ""): v for k, v in busqueda.best_params_.items()}


# Muestro las mejores combinaciones con su PR-AUC promedio en validación
# cruzada
def resumen_busqueda(busqueda, top: int = 5) -> pd.DataFrame:
    res = pd.DataFrame(busqueda.cv_results_).sort_values("rank_test_score")
    return res.filter(regex="param_|mean_test_score|std_test_score").head(top)


# Dibujo el error y el PR-AUC por época, en entrenamiento y validación
def plot_aprendizaje(modelo, archivo: str = "redes/aprendizaje"):
    import matplotlib.pyplot as plt
    h = modelo.named_steps["red"].historia_.history
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    for eje, medida, titulo in zip(ejes, ["loss", "pr_auc"], ["Error (pérdida)", "PR-AUC"]):
        eje.plot(h[medida], label="Entrenamiento")
        eje.plot(h["val_" + medida], label="Validación")
        eje.set_title(titulo)
        eje.set_xlabel("Época")
        eje.set_ylabel(titulo)
        eje.legend()
    fig.suptitle("Aprendizaje de la red por época")
    guardar_y_mostrar(fig, archivo)


# Comparo la red sin regularizar, con L2, con Dropout y con ambos
def tabla_regularizacion(datos, params: dict) -> pd.DataFrame:
    base = {**params, "l2": 0.0, "dropout": 0.0}
    variantes = {
        "Sin regularización": base,
        "Solo L2": {**base, "l2": 1e-3},
        "Solo Dropout": {**base, "dropout": 0.3},
        "L2 y Dropout": {**base, "l2": 1e-3, "dropout": 0.3},
    }
    filas = []
    for nombre, p in variantes.items():
        m = entrenar_final(datos, p)
        h = m.named_steps["red"].historia_.history
        mejor = int(np.argmax(h["val_pr_auc"]))
        filas.append({
            "configuración": nombre,
            "épocas": len(h["loss"]),
            "pr_auc_entrenamiento": h["pr_auc"][mejor],
            "pr_auc_validación": h["val_pr_auc"][mejor],
            "pr_auc_prueba": average_precision_score(
                datos.y_test, m.predict_proba(datos.X_test)[:, 1]),
        })
    tabla = pd.DataFrame(filas).set_index("configuración")
    tabla["brecha"] = tabla["pr_auc_entrenamiento"] - tabla["pr_auc_validación"]
    return tabla.round(3)


OPTIMIZADORES = {"SGD": "sgd", "RMSprop": "rmsprop", "Adam": "adam"}


# Entreno la red final cambiando solo el optimizador (SGD, RMSprop y Adam).
#
# Todos usan el método del gradiente; cambia cómo lo usan para corregir los
# pesos. Devuelvo (tabla, historias) con el aprendizaje de cada época.
def comparar_optimizadores(datos, params: dict):
    filas, historias = [], {}
    for nombre, opt in OPTIMIZADORES.items():
        m = entrenar_final(datos, {**params, "optimizador": opt, "momentum": 0.0})
        h = m.named_steps["red"].historia_.history
        historias[nombre] = {k: list(v) for k, v in h.items()}
        mejor = int(np.argmax(h["val_pr_auc"]))
        filas.append({
            "optimizador": nombre,
            "épocas": len(h["loss"]),
            "pr_auc_validación": h["val_pr_auc"][mejor],
            "pr_auc_prueba": average_precision_score(
                datos.y_test, m.predict_proba(datos.X_test)[:, 1]),
        })
    return pd.DataFrame(filas).set_index("optimizador").round(3), historias


# Dibujo el error y el PR-AUC de validación por época para cada optimizador
def plot_optimizadores(historias: dict, archivo: str = "redes/optimizadores"):
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    for eje, medida, titulo in zip(ejes, ["val_loss", "val_pr_auc"],
                                   ["Error de validación (pérdida)", "PR-AUC de validación"]):
        for nombre, h in historias.items():
            eje.plot(h[medida], label=nombre)
        eje.set_title(titulo)
        eje.set_xlabel("Época")
        eje.set_ylabel(titulo)
        eje.legend()
    fig.suptitle("Aprendizaje de la red según el optimizador")
    guardar_y_mostrar(fig, archivo)
