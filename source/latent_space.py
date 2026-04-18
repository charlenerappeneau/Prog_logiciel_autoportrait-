import numpy as np #bibliotheque pour manipuler tableaux numeriques
from PIL import Image #bibliotheque pour manipulation d'images

#-----------------------------------------------------------------------------
# Reconstruction d'images  
#-----------------------------------------------------------------------------
import torch
from torchvision import transforms
from vae_pytorch import VAE # assuming it is physically linked in code

# initialize model
model = VAE()
model.load_state_dict(torch.load('vaemodels-igimu/vae_model_30.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128), transforms.ToTensor()])

def encode(image):
    tensor = transform(image).unsqueeze(0)
    mu, log_var = model.encode(tensor)
    z = model.reparameterize(mu, log_var)
    return z.detach().numpy()

def decode(latent, shape=None):
    z = torch.from_numpy(latent)
    out = model.decode(z).reshape(-1, 3, 128, 128).squeeze(0)
    out_img = out.detach().permute(1, 2, 0).numpy() * 255.0
    return Image.fromarray(out_img.astype(np.uint8))

def reconstruct_image(image):
    """
    Cette fonction prend une image en entrée
    Reconstruction avec VAE
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
    Interpolation dans l'espace latent entre deux images
    t=0.5 correspond à une interpolation à mi-chemin
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

def blend_images(img1, img2, alpha=0.5):
    """
    Mélange classique (Pixel blending) d'images
    """
    if img1 is None or img2 is None:
        return None
    
    # S'assurer que les deux images ont la même taille
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)
        
    img1_arr = np.array(img1).astype(float)
    img2_arr = np.array(img2).astype(float)
    
    blended = (1 - alpha) * img1_arr + alpha * img2_arr
    return Image.fromarray(blended.astype(np.uint8))


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
    Calcule les directions pour chaque caractéristique de façon vectorisée (plus rapide)
    ''' 
    dico_direction = {}
    
    # Convertir en tableau numpy pour un accès plus rapide
    vecteurs_latents = np.array(vecteurs_latents)
    
    for i, feature in enumerate(features): 
        # Créer des masques booléens
        has_feature = np.array([dico_attributs[img_id][i] == 1 for img_id in images_id])
        not_have_feature = np.array([dico_attributs[img_id][i] == -1 for img_id in images_id])
        
        with_feature = vecteurs_latents[has_feature]
        without_feature = vecteurs_latents[not_have_feature]
        
        if len(with_feature) == 0 or len(without_feature) == 0:
            continue

        moy_with = np.mean(with_feature, axis=0)
        moy_without = np.mean(without_feature, axis=0)

        direction = moy_with - moy_without
        dico_direction[feature] = direction
    
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









#




 






    




