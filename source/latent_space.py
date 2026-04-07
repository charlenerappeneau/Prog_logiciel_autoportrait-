import numpy as np #bibliotheque pour manipuler tableaux numeriques
from PIL import Image #bibliotheque pour manipulation d'images

#-----------------------------------------------------------------------------
# Reconstruction d'images  
#-----------------------------------------------------------------------------
def encode(image):
    """Placeholder encodeur (image -> vecteur latent)"""
    img = np.array(image)
    return img.flatten() / 255.0  # simulation

def decode(latent, shape):
    """Placeholder decodeur (vecteur latent -> image)"""
    img = (latent * 255.0).reshape(shape).astype(np.uint8)
    return Image.fromarray(img)
 

def reconstruct_image(image):
    """
    Cette fonction prend une image en entrée
    Placeholder pour reconstruction avec VAE
    """
    if image is None:
        return None

    img = np.array(image) #conversion image en tableau numpy

    latent = encode(image)
    reconstructed = decode(latent, img_array.shape)

    return reconstructed

#-----------------------------------------------------------------------------
# Interpolation d'images
#-----------------------------------------------------------------------------
def interpolate_images(img1, img2, t=0.5):
    """
    Interpolation dans l'espace latent entre deux images
    t=0.5 correspond à une interpolation à mi-chemin

    """
    if img1 is None or img2 is None:
        return None

    z1 = encode(img1)
    z2 = encode(img2)

    z_interp = (1 - t) * z1 + t * z2

    shape = np.array(img1).shape
    return decode(z_interp, shape)


# -------------------------------------------------------------------------------------------------------
# Modification d’un attribut dans l’espace latent (mutation, ajout/suppression caractéristique)
# -------------------------------------------------------------------------------------------------------

#Chargement des attributs (CelebA)

def load_attributs(txt_file='list_attr_celeba.txt'):
    '''    
    Charge les attributs en binaire (-1 ou 1) pour chaque image
    Retour:
        images_id : liste des ids
        dico_attributs : dict {image_id: [features]}
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



# Calcul des directions latentes


features = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
             'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
             'Male','Mouth_Slightly_Open','Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
             'Rosy_Cheeks','Sideburns','Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie','Young' ]



def compute_directions(features, images_id, dico_attributs, vecteurs_latents): 
    '''
    Calcule les directions pour chaque caractéristique (chaque feature)
    
    Parametres : 
        features : liste de str de tous les attributs
        images_id : liste des id de toutes les images
        vecteurs_latents : array de dimensions (nbr_images,nbr de latent_dim) qui contient les vecteurs latents de chaque image après AE
        dico_attributs : dict{image_id : [features]} qui contient tous les attributs de chaque image (en binaires : -1 ou 1)

    Retour : 
        dico_direction : dict{feature : vecteur_direction} qui contient pour chaque feature (Blond_hair,...) le vecteur de direction correspondant 
    
    ''' 
    dico_direction = {}
    
    for i, feature in enumerate(features) : #pour chaque attribut
        with_feature = []
        without_feature = []
        
        for j, id in enumerate(images_id): #on regarde si l'attribut est présent ou pas dans chaque image 
            if dico_attributs['id'][i] == 1 : #l'attribut est présent dans l'image 
                with_feature.append(vecteurs_latents[j]) #on ajoute le vecteur latent correspondant à cette image dans la liste des 'with_features'
            elif dico_attributs['id'][i] == -1 : #l'attribut n'est pas présent dans l'image 
                without_feature.append(vecteurs_latents[j]) #on ajoute le vecteur latent correspondant à cette image dans la liste des 'without_features'
        
        # éviter erreurs si listes vides
        if len(with_feature) == 0 or len(without_feature) == 0:
            continue

        # Moyennes des vecteurs latents dans with_feature et without_features 
        moy_with = np.mean(with_feature, axis = 0)
        moy_without = np.mean(without_feature, axis = 0)

        # Computing la direction en faisant la soustraction de with et de without : 
        direction = moy_with - moy_without
        dico_direction['feature'] = direction
    
    return dico_direction


# Modification de l'attribut 

def modif_attribut(latent_original, dico_direction, attribut, modification, alpha=1.0): 
    '''
    Ajoute ou supprime un attribut dans le vecteur latent
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









## CODE Salomé

#-----------------------------------------------------------------------------
# Mélange d'images 
#-----------------------------------------------------------------------------

#additioner vect latents, en faisant moyennes des vecteurs latents de plusieurs images, etc.







 






    




