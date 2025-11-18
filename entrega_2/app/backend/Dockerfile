# app/backend/Dockerfile
FROM python:3.11-slim

# Establecer la ruta de trabajo
WORKDIR /app

# Instalar dependencias del sistema y Python
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de dependencia
COPY requirements.txt .

# Instalar librerías de Python
RUN pip install --no-cache-dir -r requirements.txt \
    # Instalar MLflow con el sabor de sklearn y lightgbm
    && pip install mlflow[sklearn] \
    && pip install lightgbm \
    && pip install shap \
    && pip install 'pandas<2.2.0' # Evitar conflictos con librerías más recientes

# Copiar el código de la aplicación
COPY main.py .

# Exponer el puerto de FastAPI
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]