FROM python:3.11-slim

WORKDIR /app

# Dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código
COPY . .

# Variables útiles
ENV PYTHONUNBUFFERED=1

# Comando de arranque (forma shell -> expande )
CMD gunicorn -b : wsgi:app
