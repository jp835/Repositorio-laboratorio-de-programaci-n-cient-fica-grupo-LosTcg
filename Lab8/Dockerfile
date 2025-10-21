
FROM python:3.10-slim

# Establece directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos necesarios
COPY . .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto de FastAPI (por defecto uvicorn corre en 8000)
EXPOSE 8000

# Comando para correr la API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
