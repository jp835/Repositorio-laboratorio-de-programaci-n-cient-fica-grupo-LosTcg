# app/frontend/Dockerfile
FROM python:3.10-slim

WORKDIR /app/frontend

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY app.py .

# El backend se expone en la red Docker, pero la app Gradio se expone al host
# a través del puerto 8501
EXPOSE 7860

# Comando de ejecución
CMD ["python", "app.py"]