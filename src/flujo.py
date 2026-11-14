# --------------------------------------------------------------------------
# Pasos del notebook main, cada uno agrupado en una sola función (cargar,
# buscar, entrenar y evaluar).
#
# Los nombres de caché (models/cache/) son los de siempre, así que los
# resultados guardados siguen valiendo.
# --------------------------------------------------------------------------
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


# Cargo el Excel, lo limpio e imprimo su tamaño
def cargar_datos():
    df = dp.clean_data(dp.load_data())
    print(df.shape)
    return df


# Separo entrenamiento y prueba (80/20 estratificado) e imprimo los tamaños
def particionar(df):
    datos = dp.prepare_data(df)
    print(datos.X_train.shape, datos.X_test.shape)
    return datos


# Entreno (o cargo de caché), elijo el umbral con CV y evalúo en prueba
def _final(nombre_cache, entrenar, datos, nombre, resultados, modelos=None, n_jobs=-1):
    modelo, umbral = utils.cacheado(nombre_cache, lambda: ev.entrenar_con_umbral(
        entrenar, datos, n_jobs=n_jobs))
    ev.registrar_modelo(nombre, modelo, datos, resultados,
                        {} if modelos is None else modelos, umbral=umbral)
    return modelo, umbral


# Busco los mejores hiperparámetros del SVM (o los cargo de caché) y muestro
# el resumen
def buscar_svm(datos):
    busqueda = utils.cacheado("busqueda_svm", lambda: svm.buscar_hiperparametros(datos))
    display(svm.resumen_busqueda(busqueda))
    return busqueda


# Muestro el heatmap de validación cruzada y devuelvo los mejores parámetros
# del SVM
def elegir_svm(busqueda):
    svm.plot_heatmap_cv(busqueda)
    params = svm.mejores_parametros(busqueda)
    print(params)
    return params


# SVM final: entreno, elijo el umbral y evalúo en prueba
def svm_final(datos, params_svm):
    resultados = {}
    modelo, umbral = _final("final_svm", lambda: svm.entrenar_final(datos, params_svm),
                            datos, "SVM", resultados)
    return modelo, umbral, resultados


# Regresión logística: busco parámetros, entreno, evalúo y muestro los
# coeficientes
def regresion_logistica(datos, resultados):
    busqueda = utils.cacheado("busqueda_regresion_logistica",
                              lambda: rl.buscar_hiperparametros(datos))
    display(rl.resumen_busqueda(busqueda, top=5))
    params = rl.mejores_parametros(busqueda)
    print(params)
    modelo, _ = _final("final_regresion_logistica", lambda: rl.entrenar_final(datos, params),
                       datos, "Regresion logistica", resultados)
    display(rl.coeficientes(modelo))
    return modelo


# Busco los mejores hiperparámetros de cada ensamble y devuelvo {nombre:
# parámetros}
def buscar_ensambles(datos, n_iter=20):
    mejores = {}
    for nombre in ens.NOMBRES:
        busq = utils.cacheado(f"busqueda_{nombre}",
                              lambda: ens.buscar_hiperparametros(nombre, datos, n_iter=n_iter))
        print(f"\n=== {nombre} ===")
        display(ens.resumen_busqueda(busq, top=5))
        mejores[nombre] = ens.mejores_parametros(busq)
    return mejores


# Entreno los ensambles finales y los evalúo en prueba (lleno resultados y
# modelos)
def entrenar_ensambles(datos, mejores_ens, resultados, modelos):
    for nombre in ens.NOMBRES:
        _final(f"final_{nombre}", lambda: ens.entrenar_final(nombre, datos, mejores_ens[nombre]),
               datos, nombre, resultados, modelos)


# Stacking de SVM + RF + XGBoost + LightGBM: devuelvo la tabla comparativa
# de todos los modelos
def stacking(datos, params_svm, mejores_ens, resultados, modelos):
    params = {"SVM": params_svm, **{n: mejores_ens[n] for n in ens.BASE_STACKING}}
    _final("final_stacking", lambda: ens.entrenar_stacking(datos, params),
           datos, "Stacking", resultados, modelos, n_jobs=1)
    return ev.tabla_comparativa(resultados)


# Red neuronal base (sin ajustar): entreno, elijo el umbral y evalúo en
# prueba
def red_base(datos, resultados, modelos):
    return _final("final_red_base", lambda: mk.entrenar_final(datos), datos,
                  "Red base", resultados, modelos, n_jobs=1)[0]


# Busco los hiperparámetros de la red y devuelvo los mejores
def buscar_red(datos):
    busqueda = utils.cacheado("busqueda_red", lambda: mk.buscar_hiperparametros(datos))
    print(f"Mejor PR-AUC en validación cruzada: {busqueda.best_score_:.3f}")
    display(mk.resumen_busqueda(busqueda, top=5))
    return mk.mejores_parametros(busqueda)


# Red neuronal final con los mejores parámetros: entreno y evalúo en prueba
def red_final(datos, params_red, resultados, modelos):
    return _final("final_red_neuronal", lambda: mk.entrenar_final(datos, params_red), datos,
                  "Red neuronal", resultados, modelos, n_jobs=1)[0]


# Comparo SGD, RMSprop y Adam en la red final: grafico y muestro la tabla
def optimizadores(datos, params_red):
    tabla, historias = utils.cacheado("optimizadores_red",
                                      lambda: mk.comparar_optimizadores(datos, params_red))
    mk.plot_optimizadores(historias)
    display(tabla)
