# Conclusiones de la Entrega Parcial 2 (MLOps)

## 1. Uso de Herramientas de Tracking y Despliegue

La implementación de **MLflow** para el tracking y registro del modelo mejoró significativamente el desarrollo al:
* **Asegurar la trazabilidad:** Se tienen registros claros de los hiperparámetros y las métricas de cada entrenamiento (incluida la búsqueda Optuna) dentro del DAG.
* **Simplificar el despliegue:** La aplicación FastAPI puede cargar la versión del "Modelo en Staging" directamente desde MLflow (`models:/sodai_drinks_lgbm_model/Staging`), independientemente de dónde se guardó el archivo físico. Esto desacopla el desarrollo del despliegue.

## 2. Desafíos e Intereses del Despliegue con Gradio/FastAPI

**Desafíos:**
* **Manejo del Pipeline Completo:** El principal desafío fue asegurar que el pipeline completo (Ingeniería + Preprocesamiento + Clasificador) fuera persistido correctamente por MLflow y que FastAPI pudiera manejar la carga y transformación de los datos de entrada (*raw features*) antes de pasarlos al modelo.
* **Dockerización:** Configurar los Dockerfiles para que incluyeran las dependencias correctas (especialmente las científicas como `scikit-learn` y `lightgbm` en un entorno *slim* de Python).
* **Comunicación entre Contenedores:** Usar `docker-compose` para orquestar la red y asegurar que Gradio y FastAPI se comunicaran correctamente a través de los nombres de servicio (`http://backend:8000`).

**Intereses:**
* **Modularidad:** La separación limpia entre la lógica de negocio (FastAPI) y la interfaz de usuario (Gradio) permite escalar o cambiar la interfaz sin tocar la API de predicción.

## 3. Aporte de Airflow a la Robustez y Escalabilidad

**Airflow** aporta robustez al:
* **Automatizar el Ciclo de Vida:** Orquesta el flujo desde la preparación de datos hasta la predicción, eliminando pasos manuales.
* **Manejo Condicional:** El uso de `BranchPythonOperator` permite implementar la lógica de reentrenamiento solo cuando se detecta *drift* (usando `drift_detection.py`), lo que es crucial para un entorno productivo eficiente y automatizado.
* **Tolerancia a Fallos:** Las configuraciones de *retries* (reintentos) y el flujo de tareas garantizan que el proceso se maneje de forma robusta ante fallos transitorios.

## 4. Mejoras Futuras del Flujo

Si se tuviera más tiempo y recursos:

* **Monitoreo Avanzado:** Se añadirían tareas de monitoreo de rendimiento post-predicción (e.g., drift en métricas de negocio, latencia de la API) utilizando herramientas como Prometheus/Grafana.
* **Métricas Adicionales:** Se registrarían métricas post-predicción más específicas del negocio (e.g., Top-N F1, Cobertura, Popularidad de las recomendaciones).
* **Segmentación del Entrenamiento:** En lugar de reentrenar con todos los datos, se implementaría una lógica para solo reentrenar con el último trimestre o el *dataset* de *drift* para hacer el proceso más rápido y enfocado en la data reciente.
* **Despliegue Blue/Green:** Se implementaría una estrategia de despliegue más segura (e.g., Blue/Green o Canary Release) en el DAG para el paso del modelo a **Production** después de una validación exhaustiva.