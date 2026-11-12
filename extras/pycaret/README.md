# Comparación externa con PyCaret

PyCaret solo sirve para comprobar de forma externa los resultados de mi código. Va en su propio entorno, aparte del `.venv311` del proyecto principal (que usa pandas 3 y no es compatible con PyCaret). No se mezclan.

## Qué hay en esta carpeta

- `comparacion_pycaret.ipynb`: el notebook de la comparación.
- `cache_pycaret.json`: resultados ya calculados, para que el notebook corra en segundos. Se sube a GitHub a propósito.
- `requirements.txt`: las librerías del entorno de PyCaret.

## Pasos para crear el entorno

Desde la carpeta `extras/pycaret`, en Git Bash:

1. Instalar Python 3.11: `py install 3.11`
2. Crear el entorno: `py -3.11 -m venv .venv`
3. Activarlo: `source .venv/Scripts/activate`
4. Instalar las librerías: `pip install -r requirements.txt`
5. Registrarlo como kernel: `python -m ipykernel install --user --name pycaret --display-name "PyCaret"`
6. Abrir `comparacion_pycaret.ipynb` y elegir el kernel **PyCaret** en Select Kernel (en Python Environments o en Jupyter Kernel).

## Cómo correrlo

Con el kernel **PyCaret**, ejecutar todas las celdas (Run All). Con `cache_pycaret.json` presente tarda segundos y muestra mensajes "Cargado desde cache_pycaret.json".

Si algún día cambian los datos o la configuración, borrar `cache_pycaret.json` y correr todo una vez. Esa primera corrida tarda unos 14 minutos (la parte de ajuste, sobre todo el SVM) y vuelve a crear el archivo.

## Notas

- Uso `lightgbm==4.5.0` porque con la 4.7.0 se cierra por un error de memoria en Windows y PyCaret lo salta en silencio.
- El notebook usa `src/` y los datos del proyecto principal, no los copia.
- No se suben a GitHub el entorno `.venv` ni `logs.log` (los ignora el `.gitignore`).
