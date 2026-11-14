# --------------------------------------------------------------------------
# Funciones compartidas por todos los módulos: carpeta de figuras, estilo y
# guardado
# --------------------------------------------------------------------------
import shutil

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.datos.data_prep import PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = MODELS_DIR / "cache"
sns.set_theme(style="whitegrid")


# Guardo la figura en reports/figures/ (si guardar=True), la muestro y la
# cierro
def guardar_y_mostrar(fig, nombre: str, guardar: bool = True):
    if guardar:
        ruta = FIG_DIR / f"{nombre}.png"  # nombre puede traer carpeta, ej. "svm/kernels"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# Guardo cada modelo entrenado y la tabla de resultados en models/, para no
# reentrenar
def guardar_modelos(modelos: dict, resultados: dict):
    carpeta = MODELS_DIR / "finales"  # los modelos van en models/finales/, aparte de resultados y caché
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre, modelo in modelos.items():
        joblib.dump(modelo, carpeta / f"{nombre.lower().replace(' ', '_')}.joblib", compress=3)
    joblib.dump(resultados, MODELS_DIR / "resultados.joblib", compress=3)


# Cargo los modelos y los resultados guardados con guardar_modelos
def cargar_modelos():
    resultados = joblib.load(MODELS_DIR / "resultados.joblib")
    carpeta = MODELS_DIR / "finales"
    modelos = {n: joblib.load(carpeta / f"{n.lower().replace(' ', '_')}.joblib") for n in resultados}
    return modelos, resultados


# Si el resultado ya está guardado en models/cache/, lo cargo; si no, lo
# calculo y lo guardo.
#
# `calcular` es una función sin argumentos (por ejemplo, una lambda) que
# solo se ejecuta cuando no hay nada guardado.
#
# Con forzar=True lo vuelvo a calcular y reemplazo lo guardado.
def cacheado(nombre: str, calcular, forzar: bool = False):
    archivo = CACHE_DIR / f"{nombre.lower().replace(' ', '_')}.joblib"
    if archivo.exists() and not forzar:
        print(f"Cargado desde models/cache: {nombre}")
        return joblib.load(archivo)
    resultado = calcular()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(resultado, archivo, compress=3)
    return resultado


# Borro models/cache/ (lo uso si cambian los datos o el código de los
# modelos)
def borrar_cache():
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
