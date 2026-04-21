import gradio as gr
from PIL import Image
from source.prepa_dataset import *
from source.latent_space import load_directions, build_modifications, modify_image_attributes
import os
import numpy as np
from source.genetic_algorithm import GeneticAlgorithm
from source.latent_space import encode, decode

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
def selection_images(sexe, couleur_chev, type_chev) : 
    '''
    Cette fonction est appelée lorsque l'utilisateur clique sur 'Afficher les images'
    Elle récupère les réponses du questionnaire, construit les contraintes de filtrage, filtre le dataset CelebA et affiche les images correspondantes.

    Paramètres :
        sexe (str) : choix de sexe ("Homme", "Femme", "Aucun changement")
        couleur_chev (str) : choix de couleur de cheveux (ex: "Blond", "Brun", "Noir", "Rouge", "Aucun changement")
        type_chev (str) : choix de type de cheveux (ex: "Lisse", "Bouclés", "Ondulés", "Aucun changement")  
    Retour :
        images (list[Image]) : liste des images à afficher (taille max_images)
        chemins (list[str]) : liste des chemins correspondants aux images affichées
        blocs (list[gr.update]) : liste des mises à jour de visibilité pour les blocs d'affichage des images
        gr.update(visible) : mise à jour de la visibilité du questionnaire (False pour le cacher)
        gr.update(visible) : mise à jour de la visibilité du bouton de validation (True pour l'afficher)
        message (str) : message à afficher indiquant le nombre d'images proposées ou une erreur si aucune image ne correspond aux critères
    '''
    print("Réponses utilisateur :", sexe, couleur_chev, type_chev)
    
    requirem = build_requirements(sexe, couleur_chev, type_chev)
    print("Requirements :", requirem)

    filtered_df = filtrage_dataset(df_merged, requirem)
    print("Taille filtered_df :", len(filtered_df))

    #nombre images : 
    nombre = min(max_images, len(filtered_df)) #on ne peut pas afficher plus que le 6 images et le nb d'images disponibles qui correspondent aux caractéritiques choisies par l'utilisateur
    
    if nombre == 0 : 
        print("Aucune image trouvée")
        images = [None] * max_images
        chemins = [None] * max_images
        blocs = [gr.update(visible=False) for _ in range(max_images)]
        return (*images, *chemins, *blocs, gr.update(visible=True), gr.update(visible=False),"Aucune image proposée")
    
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
    Cette fonction lit les 6 chemins potentiels des images et les 6 checkboxes, elle récupère ensuite uniquement les chemins des images cochées.
    Paramètres :
        path1, path2, path3, path4, path5, path6 (str) : chemins des images affichées dans les 6 blocs d'affichage
        check1, check2, check3, check4, check5, check6 (bool) : valeurs des checkboxes associées à chaque image (True si cochée, False sinon)
    Retour :
        selection (list[str]) : liste des chemins des images sélectionnées par l'utilisateur (c   
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
    Cette fonction vérifie que le nombre d'images sélectionnées par l'utilisateur correspond bien à l'action qu'il souhaite réaliser (reconstruction, interpolation, fusion).
    Paramètres :
        selection (list[str]) : liste des chemins des images sélectionnées par l'utilisateur
        action (str) : l'action que l'utilisateur souhaite réaliser
    Retour :
        valid (bool) : True si la sélection est valide pour l'action, False sinon
        message (str) : message d'erreur si la sélection n'est pas valide, ou message de confirmation si la sélection est valide
    '''
    n = len(selection)

    if action == 'reconstruction':
        if n != 1:
            return False, 'Pour la reconstruction, sélectionnez exactement 1 image'
    elif action == 'interpolation':
        if n != 2 :
            return False, 'Pour une interpolation, sélectionnez exactement 2 images'
    elif action == 'fusion':
        if n != 3 :
            return False, 'Pour une fusion, sélectionnez exactement 3 images'
    return True, ''

# ---------------------------------------------------------------------------------------
# Etape 4 : Questionnaire 2 : récupération des modifications ('mutations') demandées
# ---------------------------------------------------------------------------------------
dico_direction = load_directions("dataset/directions_latentes.npy")


def appliquer_modifications(image_base, couleur_chev_q2, type_chev_q2, pilosite_q2, visage_q2, accessoires_q2):
    '''
    Cette fonction applique les modifications demandées par l'utilisateur dans le questionnaire 2 à l'image sélectionnée, en utilisant les directions latentes calculées à partir du dataset CelebA.
    Paramètres :
        image_base (PIL.Image) : l'image sélectionnée par l'utilisateur à modifier
        couleur_chev_q2 (str) : choix de couleur de cheveux (ex: "Blond", "Brun", "Noir", "Rouge", "Aucun changement")
        type_chev_q2 (str) : choix de type de cheveux (ex: "Lisse", "Bouclés", "Ondulés", "Aucun changement")
        pilosite_q2 (list[str]) : liste des choix de pilosité (ex: ["Barbe", "Moustache"]) 
        visage_q2 (list[str]) : liste des choix de traits du visage (ex: ["Joues rosées", "Nez pointu"]) 
        accessoires_q2 (list[str]) : liste des choix d'accessoires (ex: ["Lunettes", "Maquillage prononcé"]) 
    Retour :
        image_modifiee (PIL.Image) : l'image modifiée selon les choix de l'utilisateur
        message (str) : message indiquant les modifications appliquées ou une erreur si aucune modification n'a été demandée
    '''
    if image_base is None:
        return None, "Erreur : aucune image à modifier."

    # Construction de la liste des modifications
    modifications = build_modifications(couleur_chev=couleur_chev_q2, type_chev=type_chev_q2, pilosite=pilosite_q2, visage=visage_q2, accessoires=accessoires_q2 )

    print("Modifications demandées :", modifications)

    if len(modifications) == 0:
        return image_base, "Aucune modification sélectionnée."

    # Application dans l'espace latent
    image_modifiee = modify_image_attributes(image_base, dico_direction, modifications)

    # Texte résumé
    resume = ["Modifications appliquées :"]
    for attribut, action, alpha in modifications:
        resume.append(f"- {attribut} ({action}, alpha={alpha})")

    message = "\n".join(resume)

    return image_modifiee, message

def create_fitness_function(attribute_name, dico_direction):
    """
    Crée une fonction de fitness pour l'attribut sélectionné.
    
    Cette fonction mesure à quel point un vecteur latent est aligné avec 
    la direction latente correspondant à un attribut donné, en calculant 
    un produit scalaire normalisé.
    
    Paramètres:
        attribute_name (str): Le nom de l'attribut pour lequel on veut créer la fonction.
        dico_direction (dict): Dictionnaire contenant les vecteurs de direction pour chaque attribut.
        
    Retour:
        fitness_function (callable): La fonction de fitness qui évalue un vecteur latent.
    """
    if attribute_name not in dico_direction:
        raise ValueError(f"Attribute '{attribute_name}' not found.")
    
    selected_direction = dico_direction[attribute_name]
    direction_norm = selected_direction / np.linalg.norm(selected_direction)

    def fitness_function(latent_vector):
        latent_vector_norm = latent_vector / np.linalg.norm(latent_vector)
        return np.dot(latent_vector_norm, direction_norm)
        
    return fitness_function

def refine_with_genetic_algorithm(base_image, attribute_to_refine, generations, population_size, mutation_rate, crossover_rate):
    """
    Prend une image de base et utilise un algorithme génétique pour l'affiner 
    et l'optimiser pour un attribut spécifique.
    
    Paramètres:
        base_image (PIL.Image): L'image d'origine depuis laquelle commencer l'optimisation.
        attribute_to_refine (str): Le nom de l'attribut à optimiser.
        generations (int): Le nombre de générations pour lesquelles faire évoluer l'algorithme.
        population_size (int): La taille de la population pour chaque génération.
        mutation_rate (float): La probabilité de mutation.
        crossover_rate (float): La probabilité de croisement.
        
    Retour:
        refined_image (PIL.Image): L'image finale reconstruite à partir de l'algorithme génétique.
        message (str): Un message donnant le résultat de l'optimisation.
    """
    if base_image is None:
        return None, "Error: No base image to refine."
    if attribute_to_refine == "Aucun changement":
        return base_image, "Please select an attribute to refine."

    print(f"Starting GA refinement for attribute: {attribute_to_refine}...")

    dico_direction = load_directions()
    fitness_func = create_fitness_function(attribute_to_refine, dico_direction)
    start_vector = encode(base_image)
    latent_dim = start_vector.shape[1]

    ga = GeneticAlgorithm(
        latent_dim=latent_dim,
        population_size=population_size,
        fitness_func=fitness_func,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate
    )
    
    for i in range(population_size // 2):
        noise = np.random.randn(latent_dim) * 0.1
        ga.population[i] = start_vector.flatten() + noise

    for gen in range(generations):
        best_vector, best_score = ga.evolve()
        print(f"  Gen {gen+1}/{generations}, Best Score: {best_score:.4f}")

    final_vector = ga.population[np.argmax([fitness_func(ind) for ind in ga.population])]
    refined_image = decode(final_vector.reshape(1, -1))

    message = f"Successfully refined image for attribute: **{attribute_to_refine}**."
    print("Refinement finished.")
    
    return refined_image, message

# ---------------------------------------------------------------------------------------
# Fonction qui permet de retourner au questionnaire si besoin
# ---------------------------------------------------------------------------------------

def retour_selection():
    """
    Revient à la zone de sélection des images sans toucher
    aux cases cochées ni aux images déjà affichées.

    Retourne:
        mises à jour de l'interface (gr.update): rend visible la section de 
        sélection d'images et cache la section des résultats.
    """
    return gr.update(visible=True), gr.update(visible=False)

def retour_questionnaire():
    """
    Permet de revenir au questionnaire initial si besoin.

    Retourne:
        mises à jour de l'interface (gr.update): rend visible le 
        questionnaire et cache la sélection d'images.
    """
    return gr.update(visible=True), gr.update(visible=False)


