# 1. Utiliser une image Python avec les outils de base
FROM python:3.10-slim

# 2. Définir le dossier de travail dans le conteneur
WORKDIR /app

# 3. Empêcher Python de créer des fichiers .pyc (plus propre)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 4. INSTALLATION DES DÉPENDANCES SYSTÈME
# C'est crucial pour PIL (Pillow) et OpenCV qui gèrent tes images
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. INSTALLATION DES LIBRAIRIES PYTHON
# On copie d'abord le requirements pour profiter du cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. COPIE DU PROJET
# On copie tout le contenu de ton dossier actuel dans /app
COPY . .

# 7. RÉGLAGE DES VARIABLES D'ENVIRONNEMENT (Spécifique Gradio)
# Indique à Gradio d'écouter sur toutes les interfaces sur le port 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# 8. EXPOSITION DU PORT
EXPOSE 7860

# 9. COMMANDE DE LANCEMENT
# Utilise bien le chemin vers ton script principal
CMD ["python", "scripts/launching_app.py"]