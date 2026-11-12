# Predicción de churn en un banco

Proyecto final de Métodos y Aplicaciones de Analítica II, Pontificia Universidad Javeriana.

## Integrantes

- María José Jiménez
- Ana Sofía Carranza

## Problema

Un banco perdió 15% de sus clientes activos en el último trimestre. El objetivo es construir un modelo que identifique a los clientes con mayor probabilidad de retirarse, para priorizar a quiénes ofrecerles beneficios (sin cuota de manejo, mejores tasas, retiros ilimitados, etc.).

- Variable objetivo: `attrition_flag` (se convierte en `churn`: 1 = se fue, 0 = permaneció).
- Datos: 10.127 clientes y 21 variables originales demográficas, económicas y de actividad financiera. Los modelos usan 17 variables predictoras.
- Se eliminan `avg_open_to_buy` (es exactamente el cupo menos el saldo) y `avg_utilization_ratio` (es el saldo dividido entre el cupo). Con y sin esa segunda variable el desempeño fue prácticamente igual, así que se quita para tener un modelo más simple y fácil de explicar.
- Solo 16% de los clientes se fue, por eso se evalúa con PR-AUC, recall y F1, y no solo con accuracy.

## Modelos

- SVM (se comparan los kernels RBF, polinomial, lineal y sigmoide, y se elige RBF). Como el churn tiene solo dos clases, se agrega una demostración corta del SVM con más de dos clases (categoría de tarjeta).
- Regresión logística (modelo de referencia)
- Bagging
- Random Forest
- XGBoost
- LightGBM
- Stacking (SVM + Random Forest, XGBoost y LightGBM)
- Redes neuronales con Keras (una red base y una ajustada con búsqueda de hiperparámetros), dentro de `notebooks/main.ipynb`.

Las probabilidades de los modelos se calibran y se verifican con una curva de calibración y el Brier score.

## Resultados en el set de prueba

| Modelo | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Regresión logística | 0,671 | 0,686 | 0,915 | 0,737 |
| SVM | 0,763 | 0,780 | 0,965 | 0,870 |
| Bagging | 0,883 | 0,876 | 0,984 | 0,936 |
| Random Forest | 0,868 | 0,877 | 0,988 | 0,953 |
| XGBoost | 0,880 | 0,911 | 0,993 | 0,968 |
| LightGBM | 0,902 | 0,909 | 0,992 | 0,966 |
| Stacking | 0,911 | 0,914 | 0,992 | 0,967 |
| Red neuronal (Keras) | 0,760 | 0,801 | 0,969 | 0,886 |

**Modelo recomendado: LightGBM.** Está empatado técnicamente con XGBoost y Stacking, y se elige por simplicidad y rapidez. Con él, contactando al 20% de los clientes de mayor riesgo se alcanza el 95,7% de las fugas.

## Indicadores de negocio

Metas propuestas para el banco, medidas en el set de prueba (se pueden cambiar en `METAS`, dentro de `src/evaluation.py`):

| Indicador | Meta | LightGBM |
|---|---|---|
| Recall | al menos 0,85 | 0,902 |
| Precisión | al menos 0,80 | 0,916 |
| Clientes contactados | como máximo 20% | 15,8% |
| Fugas alcanzadas con el 20% de mayor riesgo | al menos 90% | 95,7% |
| PR-AUC | al menos 0,90 | 0,966 |

Bagging, Random Forest, XGBoost, LightGBM y Stacking cumplen las cinco metas. El SVM, las redes neuronales y la regresión logística no.

## Estructura del proyecto

```
.
├── data/
│   └── raw/
│       └── bank_churn.xlsx      # datos originales, sin modificar
├── extras/
│   └── pycaret/                 # comparación con PyCaret (Python 3.11, entorno propio)
├── models/                      # se crea al correr main.ipynb
│   ├── finales/                 # los 9 modelos entrenados (.joblib)
│   ├── resultados.joblib        # tabla de resultados de todos los modelos
│   └── cache/                   # pasos lentos guardados (no se sube a GitHub)
├── notebooks/
│   └── main.ipynb               # notebook principal, llama al código de src
├── reports/
│   ├── figures/                 # gráficos del informe, una carpeta por parte del código
│   │   ├── eda/                 # exploración de los datos
│   │   ├── svm/                 # SVM: kernels, matriz de confusión, curvas, importancia
│   │   ├── ensambles/           # LightGBM: matriz de confusión e importancia
│   │   ├── redes/               # red neuronal: aprendizaje, optimizadores, importancia
│   │   └── comparacion/         # comparación de todos los modelos
│   └── informe/                 # informe en Word (se agrega al terminarlo)
├── src/
│   ├── datos/                   # carga, limpieza y exploración de los datos
│   │   ├── data_prep.py         # carga, limpieza y partición de los datos
│   │   └── eda.py               # tablas y gráficos exploratorios
│   ├── modelos/                 # modelos de clasificación
│   │   ├── svm_model.py         # modelo SVM
│   │   ├── regresion_logistica.py   # modelo de referencia
│   │   ├── ensemble_models.py   # Random Forest, XGBoost, LightGBM, Bagging y Stacking
│   │   └── busqueda.py          # piezas comunes de la búsqueda de hiperparámetros
│   ├── modelo_keras.py          # red neuronal de Keras con la interfaz de scikit-learn
│   ├── evaluation.py            # métricas, umbrales, lift y gráficos de evaluación
│   ├── flujo.py                 # pasos completos del notebook (cargar, buscar, entrenar, evaluar)
│   └── utils.py                 # carpeta de figuras, estilo, caché y guardado de modelos
├── tests/                       # pruebas del código y de los resultados (pytest)
├── pytest.ini
├── requirements.txt             # librerías con versiones exactas
└── README.md
```

## Cómo correrlo

1. Clonar el repositorio y abrir la carpeta en Visual Studio Code.
2. Crear y activar un entorno virtual con Python 3.11 (TensorFlow lo necesita):

   ```
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```

3. Instalar las librerías:

   ```
   pip install -r requirements.txt
   ```

4. Abrir `notebooks/main.ipynb` y elegir el entorno `.venv` como kernel.

### Primera vez

Ejecutar todas las celdas (Run All). Tarda unos 25 minutos, sobre todo por las búsquedas de hiperparámetros y las redes neuronales. Los pasos lentos quedan guardados en `models/cache/` y los modelos finales en `models/finales/`. Los gráficos se guardan solos en `reports/figures/`, dentro de la carpeta de su tema.

### Si ya se corrió una vez

Run All vuelve a ser rápido, porque los pasos lentos se cargan de `models/cache/`. Si cambian los datos o el código de los modelos, hay que ejecutar `utils.borrar_cache()` y volver a correr todo una vez.

También se pueden cargar solo los modelos guardados, sin reentrenar:

1. Ejecutar las tres primeras celdas de código: configuración, carga y limpieza, y partición de los datos.
2. Ir a la sección "Cargar los modelos guardados" (se encuentra en el panel Esquema u Outline de VS Code), hacer clic derecho en su celda de código y elegir Execute Cell and Below.

Esa celda carga los modelos y sus resultados con `modelos, resultados = utils.cargar_modelos()`.

La comparación con PyCaret vive en `extras/pycaret/`, con su propio entorno y su propio README con los pasos para correrla.

## Pruebas

La carpeta `tests/` tiene pruebas automáticas del código (datos y partición, métricas y umbrales, caché, entrenamiento de cada modelo) y de los resultados guardados (que estén todos los modelos y que el modelo recomendado cumpla las metas del banco). Con el entorno activado:

```
pip install -r requirements.txt
pytest
```

Las pruebas de los resultados necesitan haber corrido `main.ipynb` completo una vez; si no, se saltan solas.
