"""Utilidades compartidas por todos los módulos: carpeta de figuras, estilo y guardado."""
import shutil

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.datos.data_prep import PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = MODELS_DIR / "cache"
sns.set_theme(style="whitegrid")


def guardar_y_mostrar(fig, nombre: str, guardar: bool = True):
    """Guarda la figura en reports/figures/<tema>/ (si guardar=True), la muestra y la cierra."""
    if guardar:
        ruta = FIG_DIR / f"{nombre}.png"  # nombre puede traer carpeta, ej. "svm/kernels"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def guardar_modelos(modelos: dict, resultados: dict):
    """Guarda cada modelo entrenado y la tabla de resultados en models/, para no reentrenar."""
    carpeta = MODELS_DIR / "finales"  # los modelos van en models/finales/, aparte de resultados y caché
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre, modelo in modelos.items():
        joblib.dump(modelo, carpeta / f"{nombre.lower().replace(' ', '_')}.joblib", compress=3)
    joblib.dump(resultados, MODELS_DIR / "resultados.joblib", compress=3)


def cargar_modelos():
    """Devuelve (modelos, resultados) guardados con guardar_modelos."""
    resultados = joblib.load(MODELS_DIR / "resultados.joblib")
    carpeta = MODELS_DIR / "finales"
    modelos = {n: joblib.load(carpeta / f"{n.lower().replace(' ', '_')}.joblib") for n in resultados}
    return modelos, resultados


def cacheado(nombre: str, calcular, forzar: bool = False):
    """Devuelve el resultado guardado en models/cache/ si ya existe; si no, lo calcula y lo guarda.

    `calcular` es una función sin argumentos (por ejemplo, una lambda) que solo se ejecuta
    cuando no hay nada guardado. Así las búsquedas y los entrenamientos largos se hacen una vez.
    Con `forzar=True` se vuelve a calcular y se reemplaza lo guardado.
    """
    archivo = CACHE_DIR / f"{nombre.lower().replace(' ', '_')}.joblib"
    if archivo.exists() and not forzar:
        print(f"Cargado desde models/cache: {nombre}")
        return joblib.load(archivo)
    resultado = calcular()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(resultado, archivo, compress=3)
    return resultado


def borrar_cache():
    """Borra models/cache/. Usarla si cambian los datos o el código de los modelos."""
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
