import gradio as gr
from PIL import Image
from source.prepa_dataset import *
import os

max_images = 6 

# -----------------------------
# Chargement des données
# -----------------------------
df_attr = load_attributes("dataset/list_attr_celeba.txt")
df_id = load_identities("dataset/identity_CelebA.txt")
df_merged = merge_attributes_id(df_attr, df_id)


# ---------------------------------------------------------------------------------------
# Etape 1 : affichage d'images en fonction des réponses au questionnaire
# ---------------------------------------------------------------------------------------
def selection_images(sexe, couleur_chev, type_chev, pilosite=None, visage=None, accessoires=None) : 
    '''
    Filtre le dataset selon les réponsdes de l'utilisateur au questionnaire puis affiche jusqu'à 6 images correspondantes
    '''
    print("Réponses utilisateur :", sexe, couleur_chev, type_chev, pilosite, visage, accessoires)
    
    requirem = build_requirements(sexe, couleur_chev, type_chev, pilosite, visage, accessoires)
    print("Requirements :", requirem)

    filtered_df = filtrage_dataset(df_merged, requirem)
    print("Taille filtered_df :", len(filtered_df))
    

    #nombre images : 
    nombre = min(max_images, len(filtered_df)) #on ne peut pas afficher plus que le 6 images et le nb d'images disponibles qui correspondent aux caractéritiques choisies par l'utilisateur
    
    if nombre == 0 : 
        print("Aucune image trouvée")
    
    #Tirage aléatoire des images : 
    selected_images = filtered_df.sample(n=nombre)['image_id']
    liste_images = list(selected_images)
    
    
    images = []
    chemins = []

    for img in liste_images : 
        chemin = f'dataset/img_align_celeba/{img}'
        print("Chemin généré :", chemin)    
        
        image = Image.open(chemin).copy()
        images.append(image)
        chemins.append(chemin)
    
    
    #on complète pour que les 8 blocs alloués aux images dans l'interface soient remplis
    while len(images) < max_images : 
        images.append(None)
        chemins.append(None)
    
    blocs = []
    for i in range(max_images):
        if i < nombre : 
            blocs.append(gr.update(visible=True))
        else : 
            blocs.append(gr.update(visible = False))        
    
    return (*images, *chemins, *blocs, gr.update(visible=False), gr.update(visible=True), f"{nombre} image proposées" ) #cache le questionnaire (2e output) et affiche les images selectionnées (3e output)



# ---------------------------------------------------------------------------------------
# Etape 2 : Récupérer les images choisies par utilisateurs
# ---------------------------------------------------------------------------------------

def recuperer_selection(path1, path2, path3, path4, path5, path6, check1, check2, check3, check4, check5, check6):
    '''
    Cette fonction est appelée lorsque l'utilisateur clique sur 'Continuer vers les actions'
    Elle lit les 6 chemins potentiels des images et les 6 checkboxes, elle récupère ensuite uniquement les chemins des images cochées. 
        
    '''
    selection = []
    if check1 and path1 is not None :
        selection.append(path1)
    if check2 and path2 is not None :
        selection.append(path2)
    if check3 and path3 is not None :
        selection.append(path3)
    if check4 and path4 is not None :
        selection.append(path4)
    if check5 and path5 is not None :
        selection.append(path5)
    if check6 and path6 is not None :
        selection.append(path6) 
    
    return selection



# ---------------------------------------------------------------------------------------
# Etape 3 : Verification en parallèle que le nombre d'images choisies correspond bien à 
# l'action que veut réaliser l'utilisateur
# ---------------------------------------------------------------------------------------

def verif_selection(selection, action):
    '''
    Vérifie que le nombre d'images sélectionnées correspond à l'action
    '''
    n = len(selection)

    if action == 'interpolation':
        if n != 1 : 
            return False, 'Pour la reconstruction, sélectionnez exactement 2 images'
    elif action == 'fusion':
        if n<2 and n > 3:
            return False, 'Pour la fusion, sélectionniez 2 à 3 images'
    elif action == 'mutation':
            return False, 'Pour la mutation, sélectionnez exactement 1 image'

    return True

# ---------------------------------------------------------------------------------------
# Etape 4 : Actions
# ---------------------------------------------------------------------------------------





# ---------------------------------------------------------------------------------------
# Fonction qui permet de retourner au questionnaire si besoin
# ---------------------------------------------------------------------------------------

def retour_selection():
    """
    Revient à la zone de sélection des images sans toucher
    aux cases cochées ni aux images déjà affichées
    """
    return gr.update(visible=True), gr.update(visible=False)


def retour_questionnaire():
    """
    Permet de revenir au questionnaire si besoin.
    """
    return gr.update(visible=True), gr.update(visible=False)


