FROM python:3.10-slim

# Establecer la ruta de trabajo
WORKDIR /app/backend

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar y instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    mlflow[sklearn] \
    lightgbm \
    shap \
    feature_engine \
    'pandas<2.2.0'

# Copiar todo el código de la aplicación
COPY . .

# Exponer el puerto de FastAPI
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
