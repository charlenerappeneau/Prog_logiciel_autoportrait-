import numpy as np #bibliotheque pour manipuler tableaux numeriques
from PIL import Image #bibliotheque pour manipulation d'images
import os #import pour manipuler les chemins de fichiers
from source.attributes_mapping import (SEXE_MAP, HAIR_COLOR_MAP, HAIR_TYPE_MAP, PILOSITY_MAP, FACIAL_FEATURES_MAP, ACCESSORIES_MAP) #import des dictionnaires de mapping entre les réponses du questionnaire et les attributs CelebA

#-----------------------------------------------------------------------------
# Reconstruction d'images  
# Sert à reconstruire une image à partir de son vecteur latent, 
# ou à faire une interpolation/fusion entre deux ou plus images dans l'espace latent
#-----------------------------------------------------------------------------
import torch #import pour manipuler les tenseurs et charger le modèle VAE
from torchvision import transforms # import pour les transformations d'images 

from sys import path
path.append('source/vaemodels-igimu')
from source.vae_pytorch import VAE # assuming it is physically linked in code

import sys
# Mapping du module pour corriger l'erreur de torch.load qui cherche 'vae_pytorch' 
sys.modules['vae_pytorch'] = sys.modules['source.vae_pytorch']

# L'erreur indique que le modèle a été sauvegardé avec torch.save(model) 
# au lieu de torch.save(model.state_dict())
# On charge donc directement l'objet modèle entier
model = torch.load('source/vaemodels-igimu/vae_model_30.pth', map_location='cpu', weights_only=False)
model.eval()

transform = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128), transforms.ToTensor()])


def encode(image):
    '''
    Cette fonction prend une image en entrée et retourne son vecteur latent en utilisant l'encodeur du VAE.
    Paramètre :
        image (PIL.Image) : image à encoder
    Retour :
        latent (numpy.ndarray) : vecteur latent de l'image
    '''
    if image.mode != 'RGB':
        image = image.convert('RGB')
    tensor = transform(image).unsqueeze(0)
    mu, log_var = model.encode(tensor)
    # Pour l'inférence/reconstruction, nous utilisons la moyenne (mu)
    # plutôt que `reparameterize` qui ajoute du bruit aléatoire.
    return mu.detach().numpy()

def decode(latent, shape=None):
    '''
    Cette fonction prend un vecteur latent en entrée et retourne l'image reconstruite correspondante en utilisant le décodeur du VAE.
    Paramètres :
        latent (numpy.ndarray) : vecteur latent à décoder
        shape (tuple) : forme de l'image d'origine (hauteur, largeur, canaux), utilisée pour redimensionner l'image reconstruite si nécessaire
    Retour :
        image (PIL.Image) : image reconstruite à partir du vecteur latent
    '''
    z = torch.from_numpy(latent).float()
    out = model.decode(z).reshape(-1, 3, 128, 128).squeeze(0)
    out_img = out.detach().permute(1, 2, 0).numpy() * 255.0
    out_img = np.clip(out_img, 0, 255)
    return Image.fromarray(out_img.astype(np.uint8))

def reconstruct_image(image):
    """
    Cette fonction prend une image en entrée, l'encode dans l'espace latent, puis la décode pour obtenir une reconstruction de l'image.
    Paramètre :
        image (PIL.Image) : image à reconstruire
    Retour :
        reconstructed (PIL.Image) : image reconstruite à partir de l'image d'entrée
    """
    if image is None:
        return None

    img = np.array(image) #conversion image en tableau numpy

    latent = encode(image)
    reconstructed = decode(latent, img.shape)

    return reconstructed

#-----------------------------------------------------------------------------
# Interpolation d'images
#-----------------------------------------------------------------------------
def interpolate_images(img1, img2, t=0.5):
    """
    Cette fonction prend deux images en entrée, les encode dans l'espace latent, puis effectue une interpolation linéaire entre les deux vecteurs latents pour obtenir une image intermédiaire.

    Paramètres :
        img1 (PIL.Image) : première image
        img2 (PIL.Image) : deuxième image
        t (float) : paramètre d'interpolation entre 0 et 1, où 0 correspond à img1 et 1 correspond à img2. Par défaut, t=0.5 correspond à une interpolation à mi-chemin.
    Retour :
        interpolated (PIL.Image) : image résultante de l'interpolation entre img1 et img2
    """
    if img1 is None or img2 is None:
        return None

    # S'assurer que les deux images ont la même taille
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)

    z1 = encode(img1)
    z2 = encode(img2)

    z_interp = (1 - t) * z1 + t * z2

    shape = np.array(img1).shape
    return decode(z_interp, shape)

#-----------------------------------------------------------------------------
# Fusion d'images
#-----------------------------------------------------------------------------
def fusion_images(img1, img2, img3, w1=1/3, w2=1/3, w3=1/3):
    """
    Cette fonction prend trois images en entrée, les encode dans l'espace latent, puis effectue une interpolation pondérée entre les trois vecteurs latents pour obtenir une image fusionnée.
    Paramètres :
        img1 (PIL.Image) : première image
        img2 (PIL.Image) : deuxième image
        img3 (PIL.Image) : troisième image
        w1 (float) : poids de la première image dans la fusion, par défaut 1/3
        w2 (float) : poids de la deuxième image dans la fusion, par défaut 1/3
        w3 (float) : poids de la troisième image dans la fusion, par défaut 1/3
    Retour :
        fused (PIL.Image) : image résultante de la fusion entre img1, img2 et img3  
    """
    if img1 is None or img2 is None or img3 is None:
        return None

    # S'assurer que les images ont la même taille
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)
    if img1.size != img3.size:
        img3 = img3.resize(img1.size)

    z1 = encode(img1)
    z2 = encode(img2)
    z3 = encode(img3)

    # Normalisation des poids pour assurer une somme de 1
    total_w = w1 + w2 + w3
    w1, w2, w3 = w1/total_w, w2/total_w, w3/total_w

    z_fusion = w1 * z1 + w2 * z2 + w3 * z3

    shape = np.array(img1).shape
    return decode(z_fusion, shape)



#def blend_images(img1, img2, alpha=0.5):
    """
#    Mélange classique (Pixel blending) d'images
#    """
#    if img1 is None or img2 is None:
#        return None
    
#    # S'assurer que les deux images ont la même taille
#    if img1.size != img2.size:
#        img2 = img2.resize(img1.size)
        
#    img1_arr = np.array(img1).astype(float)
 #   img2_arr = np.array(img2).astype(float)
    
 #   blended = (1 - alpha) * img1_arr + alpha * img2_arr
 #   return Image.fromarray(blended.astype(np.uint8))


# -------------------------------------------------------------------------------------------------------
# Modification d’un attribut dans l’espace latent (mutation, ajout/suppression caractéristique)
# -------------------------------------------------------------------------------------------------------

#Chargement des attributs (CelebA) : 

def load_attributs(txt_file='list_attr_celeba.txt'):
    '''    
    Cette fonction charge les attributs du dataset CelebA à partir du fichier list_attr_celeba.txt et retourne un dictionnaire avec les attributs de chaque image.
    Paramètres :
        txt_file (str) : chemin vers le fichier list_attr_celeba.txt
    Retour :
        images_id (list) : liste des noms d'images
        dico_attributs (dict) : dictionnaire où la clé est le nom de l'image et la valeur est une liste des attributs correspondants à cette image (1 ou -1)
    '''    
    dico_attributs = {}
    images_id = []

    with open(txt_file, 'r') as file:
        lines = file.readlines()

        # ignorer les deux premières lignes (header CelebA)
        lines = lines[2:]

        for line in lines:
            parties = line.strip().split()
            image_id = parties[0]
            images_id.append(image_id)

            features = [int(x) for x in parties[1:]]
            dico_attributs[image_id] = features

    return images_id, dico_attributs



# Calcul des directions latentes : 
def compute_latent_vectors(df, image_folder, max_images=None):
    '''
    Cette fonction prend en entrée un DataFrame contenant au moins une colonne 'image_id' avec les noms des images, ainsi qu'un dossier contenant les images correspondantes. 
    Elle calcule le vecteur latent de chaque image en utilisant l'encodeur du VAE et ajoute une nouvelle colonne 'latent' au DataFrame avec ces vecteurs latents.

    Paramètres :
        df : pandas.DataFrame
            DataFrame contenant au minimum la colonne 'image_id'
        image_folder : str
            Dossier contenant les images CelebA
        max_images : int ou None
            Si renseigné, limite le nombre d'images traitées
            (utile pour les tests)

    Retour :
        df_latent : pandas.DataFrame
            Même DataFrame que df, avec une nouvelle colonne 'latent'
            contenant les vecteurs latents
    '''
    # Copie du DataFrame pour ne pas modifier l’original
    df_latent = df.copy()

    # Si max_images est donné, on garde seulement les premières lignes (utile pour tester la fct)
    if max_images is not None:
        df_latent = df_latent.iloc[:max_images].copy()

    # Liste qui contiendra les latents
    liste_latents = []
    # Boucle sur chaque image
    for image_id in df_latent["image_id"]:
        image_path = os.path.join(image_folder, image_id)
        try:
            # Ouvre image
            image = Image.open(image_path).convert("RGB")
            # Encode dans l’espace latent
            latent = encode(image)
            # Retire dimension batch : (1,128) -> (128,)
            latent = latent.squeeze()
            liste_latents.append(latent)

        except Exception as e:
            print(f"Erreur avec {image_id} : {e}")
            # On met None si erreur
            liste_latents.append(None)
    # Ajout de la colonne latent
    df_latent["latent"] = liste_latents

    return df_latent


def compute_directions(df_latent, feature_names):
    '''
    Cette fonction calculeles directions latentes des attributs à partir d'un DataFrame
    contenant :
        - les colonnes d'attributs CelebA (-1 / 1)
        - une colonne 'latent' contenant les vecteurs latents

    Parametres : 
        df_latent : pandas.DataFrame
            DataFrame contenant :
            - image_id
            - les attributs CelebA
            - la colonne 'latent'

        feature_names : list[str]
            Liste des attributs pour lesquels on veut calculer une direction

    Retour: 
        dico_directions : dict
            Dictionnaire :
            clé = nom de l'attribut
            valeur = vecteur direction correspondant
    '''
    dico_directions = {}

    # On enlève les lignes où le latent est manquant (au cas où)
    df_clean = df_latent[df_latent["latent"].notna()].copy()

    # Boucle sur chaque attribut demandé
    for feature in feature_names:
        # Vérifie que la colonne existe bien (au cas où)
        if feature not in df_clean.columns:
            print(f"Attribut absent du DataFrame : {feature}")
            continue

        # Groupe des images avec l'attribut
        df_with = df_clean[df_clean[feature] == 1]

        # Groupe des images sans l'attribut
        df_without = df_clean[df_clean[feature] == -1]

        # Si un des deux groupes est vide, on ne peut pas calculer de direction
        if len(df_with) == 0 or len(df_without) == 0:
            print(f"Attribut ignoré (groupe vide) : {feature}")
            continue

        # On récupère les vecteurs latents sous forme de matrice numpy
        latents_with = np.stack(df_with["latent"].values)
        latents_without = np.stack(df_without["latent"].values)

        # Moyennes des deux groupes
        mean_with = np.mean(latents_with, axis=0)
        mean_without = np.mean(latents_without, axis=0)

        # Direction latente
        direction = mean_with - mean_without
        # Stockage dans le dictionnaire final
        dico_directions[feature] = direction

    return dico_directions



def build_modifications(couleur_chev=None, type_chev=None, pilosite=None, visage=None, accessoires=None):
    '''
    Cette fonction construit la liste des modifications à appliquer dans l'espace latent
    à partir des réponses du questionnaire 2.

    Paramètres :
        couleur_chev (str) : choix de couleur de cheveux (ex: "Blond", "Brun", "Noir", "Rouge", "Aucun changement")
        type_chev (str) : choix de type de cheveux (ex: "Lisse", "Bouclés", "Ondulés", "Aucun changement")
        pilosite (list[str]) : liste des choix de pilosité (ex: ["Barbe", "Moustache"]) 
    Retour :
        modifications (list[tuple]) : liste de tuples (attribut, action, alpha) à appliquer dans l'espace latent
    '''
    modifications = []

    # Couleur des cheveux
    if couleur_chev and couleur_chev != "Aucun changement":
        if couleur_chev in HAIR_COLOR_MAP:
            attr = HAIR_COLOR_MAP[couleur_chev]
            modifications.append((attr, "ajout", 2))
    # Type de cheveux
    if type_chev and type_chev != "Aucun changement":
        if type_chev in HAIR_TYPE_MAP:
            attr = HAIR_TYPE_MAP[type_chev]
            modifications.append((attr, "ajout", 2))
    # Pilosité
    if pilosite:
        for element in pilosite:
            if element in PILOSITY_MAP:
                attr = PILOSITY_MAP[element]
                modifications.append((attr, "ajout", 2))
    # Visage
    if visage:
        for element in visage:
            if element in FACIAL_FEATURES_MAP:
                attr = FACIAL_FEATURES_MAP[element]
                modifications.append((attr, "ajout", 2))
    # Accessoires
    if accessoires:
        for element in accessoires:
            if element in ACCESSORIES_MAP : 
                attr = ACCESSORIES_MAP[element]
                modifications.append((attr, "ajout", 2))
    return modifications


# Modification de l'attribut 

def modif_attribut(latent_original, dico_direction, attribut, modification, alpha=1.0): 
    '''
    Cette fonction modifie un vecteur latent en ajoutant ou en soustrayant une direction latente correspondant à un attribut spécifique.

    Paramètres :
        latent_original (numpy.ndarray) : vecteur latent de l'image à modifier
        dico_direction (dict) : dictionnaire des directions latentes
        attribut (str) : nom de l'attribut à modifier
        modification (str) : type de modification ('ajout' ou 'suppression')
        alpha (float) : intensité de la modification (par défaut 1.0, peut être ajusté pour des modifications plus ou moins fortes)
     Retour :
        nouv_latent (numpy.ndarray) : nouveau vecteur latent après modification de l'attribut
    '''
    if attribut not in dico_direction:
        raise ValueError(f"Attribut {attribut} non trouvé")

    direction = dico_direction[attribut]

    if modification == 'ajout':
        nouv_latent = latent_original + alpha * direction
    elif modification == 'suppression':
        nouv_latent = latent_original - alpha * direction
    else:
        raise ValueError("modification doit être 'ajout' ou 'suppression'")

    return nouv_latent


































#autre version ? : on propose à l'utilisateur de choisir plusieurs attributs à modifier parmi une liste (voir si on donne les noms tels quels dans la liste features ou alors on donne d'autres noms avec explication et on fait le lien nous même)
#pour faire plusieurs modifications de plusieurs attributs -> boucles 

def modify_image_attributes(image, dico_direction, modifications):
    """
    Encodes an image, applies a set of attribute modifications in the latent space, and decodes the result.
    
    Paramètres :
        image (PIL.Image) : L'image de base à modifier.
        dico_direction (dict) : Dictionnaire des vecteurs de direction précalculés pour chaque attribut.
        modifications (list) : Liste de tuples contenant les modifications souhaiées
                               (attribut, action, intensité/alpha).
                               Exemple: [('Smiling', 'ajout', 1.5), ('Eyeglasses', 'suppression', 1.0)]
    Retour :
        PIL.Image : L'image générée avec les nouveaux attributs.
    """
    if image is None:
        return None
        
    # 1. Encodage de l'image dans l'espace latent
    latent = encode(image)
    
    # 2. Boucle sur les modifications pour modifier le vecteur latent
    latent_modifie = latent.copy()
    
    for modif in modifications:
        attribut = modif[0]
        action = modif[1]
        alpha = modif[2] if len(modif) > 2 else 1.0
        
        if attribut in dico_direction:
            latent_modifie = modif_attribut(latent_modifie, dico_direction, attribut, action, alpha)
        else:
            print(f"Attention: l'attribut '{attribut}' n'existe pas ou n'a pas de vecteur direction calculé.")

    # 3. Décodage du nouveau vecteur latent
    shape = np.array(image).shape
    image_modifiee = decode(latent_modifie, shape)
    
    return image_modifiee



def load_directions(file_path="dataset/directions_latentes.npy"):
    '''
    Charge le dictionnaire des directions latentes sauvegardé.
    '''
    dico_direction = np.load(file_path, allow_pickle=True).item()
    return dico_direction





#




 






    




