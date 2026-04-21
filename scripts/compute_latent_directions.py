#Ce script est conçu pour calculer les directions latentes associées à différents attributs faciaux
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from source.latent_space import compute_directions

# Features
features = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
             'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
             'Male','Mouth_Slightly_Open','Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
             'Rosy_Cheeks','Sideburns','Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie','Young' ]


# Charger le DataFrame contenant les latents
df_latent = pd.read_pickle("dataset/df_latents.pkl")

# Calculer les directions latentes
dico_directions = compute_directions(df_latent, features)

# Sauvegarder le dictionnaire des directions
np.save("dataset/directions_latentes.npy", dico_directions, allow_pickle=True)

print("Sauvegarde terminée : dataset/directions_latentes.npy")
print("Nombre de directions calculées :", len(dico_directions))

# Contôle rapide
premier_attribut = list(dico_directions.keys())[0]
print(f'Premier attribut : {premier_attribut}') #devrait être 5_o_Clock_Shadow
print(f'Forme de sa direction {dico_directions[premier_attribut].shape}')  #devrait être (128,1) car 128 dim et une colonne