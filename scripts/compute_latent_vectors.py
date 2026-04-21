#Ce script est conçu pour calculer les vecteurs latents associés à chaque image du dataset CelebA
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd
from source.latent_space import compute_latent_vectors
from source.prepa_dataset import load_attributes


# Charger le fichier d'attributs CelebA : 
df = load_attributes("dataset/list_attr_celeba.txt")

#Calculer les vecteurs latents pour toutes les images
df_latent = compute_latent_vectors(df,"dataset/img_align_celeba")

#Sauvegarder le DataFrame avec la colonne 'latent'
df_latent.to_pickle("dataset/df_latents.pkl")

print("Sauvegarde terminée : dataset/df_latents.pkl")
print(f"Nombre d'images traitées :{len(df_latent)}")
