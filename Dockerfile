# DOCKERFILE POUR L'APPLICATION GRADIO


# IMAGE DE BASE
# Utilise une image officielle Python légère (version 3.10)
# contenant Python déjà installé.
FROM python:3.10-slim



# Tous les fichiers du projet seront placés dans /app
# à l'intérieur du conteneur Docker.
WORKDIR /app



# VARIABLES PYTHON
# Empêche Python de créer des fichiers .pyc
# (fichiers compilés inutiles ici)
ENV PYTHONDONTWRITEBYTECODE=1

# Force l'affichage immédiat des logs dans le terminal
# (pratique pour voir les prints et erreurs en direct)
ENV PYTHONUNBUFFERED=1



# DÉPENDANCES SYSTÈME : 

# Mise à jour des paquets Linux puis installation de librairies
# nécessaires pour l'affichage et traitement d'images (Pillow, OpenCV, ...)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*



# INSTALLATION DES LIBRAIRIES PYTHON : 
# On copie d'abord uniquement requirements.txt
# pour profiter du cache Docker si le code change

COPY requirements.txt .

# Installe toutes les dépendances Python du projet : gradio, torch, pandas, numpy, pillow, etc.
RUN pip install --no-cache-dir -r requirements.txt



# COPIE DU PROJET
# Copie tout le contenu du dépôt Git dans /app (code source, scripts, main.py, etc.).
COPY . .



# VARIABLES GRADIO
# Force Gradio à marcher sur toutes les interfaces réseau
# nécessaire pour accéder à l'application depuis le navigateur
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Définit le port utilisé par Gradio
ENV GRADIO_SERVER_PORT=7860

# Indique que le conteneur utilise le port 7860.
EXPOSE 7860



# COMMANDE DE LANCEMENT
# Quand on exécute le conteneur, lance l'application principale
CMD ["python", "main.py"]