"""Pasos completos del notebook main en una sola función cada uno.

Cada función agrupa lo que antes eran varias líneas del notebook (cargar,
buscar, entrenar, evaluar). Los nombres de caché (`models/cache/`) son los
mismos de siempre, así que los resultados guardados siguen valiendo.
"""
from src.datos import data_prep as dp
from src.modelos import ensemble_models as ens
from src import evaluation as ev
from src import modelo_keras as mk
from src.modelos import regresion_logistica as rl
from src.modelos import svm_model as svm
from src import utils

try:
    from IPython.display import display
except ImportError:          # fuera de Jupyter
    display = print


def cargar_datos():
    """Carga el Excel, lo limpia e imprime su tamaño."""
    df = dp.clean_data(dp.load_data())
    print(df.shape)
    return df


def particionar(df):
    """Separa entrenamiento y prueba (80/20 estratificado) e imprime los tamaños."""
    datos = dp.prepare_data(df)
    print(datos.X_train.shape, datos.X_test.shape)
    return datos


def _final(nombre_cache, entrenar, datos, nombre, resultados, modelos=None, n_jobs=-1):
    """Entrena (o carga de caché), elige el umbral con CV y evalúa en prueba."""
    modelo, umbral = utils.cacheado(nombre_cache, lambda: ev.entrenar_con_umbral(
        entrenar, datos, n_jobs=n_jobs))
    ev.registrar_modelo(nombre, modelo, datos, resultados,
                        {} if modelos is None else modelos, umbral=umbral)
    return modelo, umbral


def buscar_svm(datos):
    """Busca los mejores hiperparámetros del SVM (o los carga de caché) y muestra el resumen."""
    busqueda = utils.cacheado("busqueda_svm", lambda: svm.buscar_hiperparametros(datos))
    display(svm.resumen_busqueda(busqueda))
    return busqueda


def elegir_svm(busqueda):
    """Muestra el heatmap de validación cruzada y devuelve los mejores parámetros del SVM."""
    svm.plot_heatmap_cv(busqueda)
    params = svm.mejores_parametros(busqueda)
    print(params)
    return params


def svm_final(datos, params_svm):
    """SVM final: entrena, elige el umbral y evalúa en prueba."""
    resultados = {}
    modelo, umbral = _final("final_svm", lambda: svm.entrenar_final(datos, params_svm),
                            datos, "SVM", resultados)
    return modelo, umbral, resultados


def regresion_logistica(datos, resultados):
    """Regresión logística: busca parámetros, entrena, evalúa y muestra coeficientes."""
    busqueda = utils.cacheado("busqueda_regresion_logistica",
                              lambda: rl.buscar_hiperparametros(datos))
    display(rl.resumen_busqueda(busqueda, top=5))
    params = rl.mejores_parametros(busqueda)
    print(params)
    modelo, _ = _final("final_regresion_logistica", lambda: rl.entrenar_final(datos, params),
                       datos, "Regresion logistica", resultados)
    display(rl.coeficientes(modelo))
    return modelo


def buscar_ensambles(datos, n_iter=20):
    """Busca los mejores hiperparámetros de cada ensamble y devuelve {nombre: params}."""
    mejores = {}
    for nombre in ens.NOMBRES:
        busq = utils.cacheado(f"busqueda_{nombre}",
                              lambda: ens.buscar_hiperparametros(nombre, datos, n_iter=n_iter))
        print(f"\n=== {nombre} ===")
        display(ens.resumen_busqueda(busq, top=5))
        mejores[nombre] = ens.mejores_parametros(busq)
    return mejores


def entrenar_ensambles(datos, mejores_ens, resultados, modelos):
    """Entrena los ensambles finales y los evalúa en prueba (llena resultados y modelos)."""
    for nombre in ens.NOMBRES:
        _final(f"final_{nombre}", lambda: ens.entrenar_final(nombre, datos, mejores_ens[nombre]),
               datos, nombre, resultados, modelos)


def stacking(datos, params_svm, mejores_ens, resultados, modelos):
    """Stacking de SVM + RF + XGBoost + LightGBM; devuelve la tabla comparativa de todos."""
    params = {"SVM": params_svm, **{n: mejores_ens[n] for n in ens.BASE_STACKING}}
    _final("final_stacking", lambda: ens.entrenar_stacking(datos, params),
           datos, "Stacking", resultados, modelos, n_jobs=1)
    return ev.tabla_comparativa(resultados)


def red_base(datos, resultados, modelos):
    """Red neuronal base (sin ajustar): entrena, elige el umbral y evalúa en prueba."""
    return _final("final_red_base", lambda: mk.entrenar_final(datos), datos,
                  "Red base", resultados, modelos, n_jobs=1)[0]


def buscar_red(datos):
    """Búsqueda de hiperparámetros de la red; devuelve los mejores parámetros."""
    busqueda = utils.cacheado("busqueda_red", lambda: mk.buscar_hiperparametros(datos))
    print(f"Mejor PR-AUC en validación cruzada: {busqueda.best_score_:.3f}")
    display(mk.resumen_busqueda(busqueda, top=5))
    return mk.mejores_parametros(busqueda)


def red_final(datos, params_red, resultados, modelos):
    """Red neuronal final con los mejores parámetros: entrena y evalúa en prueba."""
    return _final("final_red_neuronal", lambda: mk.entrenar_final(datos, params_red), datos,
                  "Red neuronal", resultados, modelos, n_jobs=1)[0]


def optimizadores(datos, params_red):
    """Compara SGD, RMSprop y Adam en la red final: grafica y muestra la tabla."""
    tabla, historias = utils.cacheado("optimizadores_red",
                                      lambda: mk.comparar_optimizadores(datos, params_red))
    mk.plot_optimizadores(historias)
    display(tabla)
