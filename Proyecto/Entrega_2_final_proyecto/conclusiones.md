# Conclusiones de la Entrega Parcial 2 (MLOps)

## 1. Uso de Herramientas de Tracking y Despliegue

La implementación de **MLflow** para el tracking y registro del modelo mejoró significativamente el desarrollo al:
* **Asegurar la trazabilidad:** Se tienen registros claros de los hiperparámetros y las métricas de cada entrenamiento (incluida la búsqueda Optuna) dentro del DAG.
* **Simplificar el despliegue:** A travez del uso de volumenes, se puede compartir el mejor modelo entrenado a travez de airflow, con nuestra backend, permitiendo una rapida conexion entre ambas
* **Analisis por entrenamiento:** Sumado a asegurar la trazabilidad, vemos que va mas alla, ya que, como generamos carpetas por fechas, y dentro de ellas guardamos los resultados de mlflow, podemos analizar como va cambiando el desempeño de nuestro modelo en el tiempo, y con el aumento de datos

## 2. Desafíos e Intereses del Despliegue con Gradio/FastAPI

**Desafíos:**
* **Manejo del Pipeline Completo:** El principal desafío fue idear la forma de que nuestra app pudiese acceder de manera mas automatica al modelo entrenado, el cual se actualizara si recibe nuevos datos
* **Dockerización:** Configurar los Dockerfiles para que incluyeran las dependencias correctas (especialmente las científicas como `scikit-learn` y `lightgbm` en un entorno *slim* de Python), y generar el compose entre frontend y backend, ademas de generar las funciones necesarias para poder predecir, y que tanto backend como frontend se pudiesen comunicar entre si


**Intereses:**
* **Modularidad:** La separación limpia entre la lógica de negocio (FastAPI) y la interfaz de usuario (Gradio) permite escalar o cambiar la interfaz sin tocar la API de predicción, lo cual puede llevar a interfaces mas intuitivas
* **Predicciones individuales:** El hecho de poder utilizar la interfaz de gradio para poder realizar predicciones individuales, y visualizar el resultado, resulta muy util cuando solo se quiere saber un dato, ya que cualquier otra forma de hacerlo, es decir de forma manual, conlleva un proceso mas engorroso

## 3. Aporte de Airflow a la Robustez y Escalabilidad

**Airflow** aporta robustez al:
* **Automatizar el Ciclo de Vida:** Orquesta el flujo desde la preparación de datos hasta la predicción, eliminando pasos manuales.
* **Manejo Condicional:** Si bien no se implemento en esta entrega, el hecho de poder genrar branches permite mas ideas o caminos, como lo que notamos en nuestra idea de estrategia para el re entrenamiento, el cual si se cumple un caso sigue tal camino, pero si se cumple el otro sigue el otro camino, y esto, es justamente lo que nos permite hacer Airflow
* **Analisis de tareas:** A travez de la interfaz de airflow podemos analizar el desempeño de cada tarea por separado, asi como los print que tenemos en ellas a forma de benchmark, esto nos permite poder analizar de manera correcta cada una de ellas, viendo su tiempo de ejecucion, errores que se generan, entre otros, lo que nos permite generar un mejor analisis y enfocar mas directamente nuestros esfuerzos a corregir ciertas tareas especificas, u optimizar tareas cuyo tiempo de ejecucion pudiese ser mas alto de lo esperado

## 4. Mejoras Futuras del Flujo

Si se tuviesemos más tiempo y recursos:

* **Monitoreo Avanzado:** Se añadirían tareas de monitoreo de rendimiento post-predicción (e.g., drift en métricas de negocio, latencia de la API) utilizando herramientas como Prometheus/Grafana.
* **Métricas Adicionales:** Se registrarían métricas post-predicción más específicas del negocio (e.g., Top-N F1, Cobertura, Popularidad de las recomendaciones).
* **Segmentación del Entrenamiento:** En lugar de reentrenar con todos los datos, se implementaría una lógica para solo reentrenar con el último trimestre, con el *dataset* de *drift* o con los datos que verifiquen cierto periodo (por ejemplo con solo el ultimo año de atos) para hacer el proceso más rápido y enfocado en la data reciente, ya que si agregamos datos sin borrar, quizas datos de hace 5-10 años, pueden no tener relevancia ahora por ejemplo
* **Despliegue de Mlflow**: Como se puede notar, si bien se usa mlflow, se usa un enfoque quizas mas local, o de guardar los archivos en carpetas, pero por lo que pudimos ver, se puede generar un contenedor de mlflow para almacenar los datos, lo cual puede ser utili para un mejor acceso y analisis de los mismos. Ademas, esto influye tambien a como se pueden compartir los modelos entre airflow y la app, ya que quizas se podria crear un compose entre todo
* **Optimizacion**: Si bien nuestras funciones y nuestro dearrollo cumple con lo requerido, siempre hay espacio para optimizacion, y dado lo grande de nuestros codigos, no podemos negar la posibilidad de que estos se puedan optimizar, bajando asi el tiempo de ejecucion 
* **Automatizacion**: Si bien ya se puede inferir, una cosa que nos gustaria ver si se puede automatizar es la conexion entre nuestra app y airflow, para depender menos de los volumenes, quizas evitar el hecho de que la data aparece magicamente en nuestro directorio, y asumir que se sube a un github por ejemplo, poniendo una tarea de descarga de esta, lo que evita tener que estar poniendo a mano la data dentro del archviso Data por ejemplo
