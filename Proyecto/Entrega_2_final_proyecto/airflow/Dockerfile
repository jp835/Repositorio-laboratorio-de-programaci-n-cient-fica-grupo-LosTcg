FROM python:3.10-slim


RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV AIRFLOW_HOME=/opt/airflow
ENV PATH="$PATH:/opt/airflow/.local/bin"
ENV PYTHONPATH="$AIRFLOW_HOME/scripts:$PYTHONPATH"


RUN mkdir -p $AIRFLOW_HOME/dags $AIRFLOW_HOME/logs $AIRFLOW_HOME/plugins
WORKDIR $AIRFLOW_HOME


RUN pip install --no-cache-dir \
    apache-airflow==2.10.2 \
    scikit-learn \
    pandas \
    numpy \
    joblib \
    lightgbm \
    mlflow \
    feature-engine \
    optuna

COPY ./dags $AIRFLOW_HOME/dags

COPY ./Modelos $AIRFLOW_HOME/Modelos
COPY ./Data $AIRFLOW_HOME/Data
COPY ./scripts $AIRFLOW_HOME/scripts
COPY ./Predicciones $AIRFLOW_HOME/Predicciones



RUN airflow db init && \
    airflow users create \
        --username admin \
        --password admin \
        --firstname Air \
        --lastname Flow \
        --role Admin \
        --email admin

EXPOSE 8080


CMD ["sh", "-c", "airflow scheduler & airflow webserver --port 8080"]