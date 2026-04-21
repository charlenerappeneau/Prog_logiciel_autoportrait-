import pandas as pd
from source.attributes_mapping import (SEXE_MAP, HAIR_COLOR_MAP, HAIR_TYPE_MAP)
   

# Idée : construit un DataFrame pd qui contient comme lignes les images (leur id) et comme colonne les attributs (Glasses, Male,...). Si l'attribut est présent mettre 1 dans la case
# correspondante, sinon mettre -1

def load_attributes(file_path = 'dataset/list_attr_celeba.txt'):
    '''
    Le fichier 'list_attr_celeba.txt' contient : 
        - Ligne 1 : nombre total d'images
        - Ligne 2 : noms des attributs (Smiling, Blond_Hair, Black_hair,...)
        - Lignes suivantes : une image par ligne avec : id_image attribut1 attribut2 ...
            Exemple : 0000001 -1 1 ...
        1 signifie que l'attribut est présent (exemple : a les cheveux blonds), -1 singnifie que l'attribut est absent (n'a pas les cheveux blonds)

    Cette fonction charge ce fichier texte d'attributs d'images  et le convertit en DataFrame pandas.
    Le DataFrame permettra de filtrer facilement les images selon les attributs choisis par l'utilisateur dans l'interface graphique.
    
    Paramètres : 
        file_path (str)
            Chemin vers le fichier texte contenant les données d'attributs
    
    Retour : 
        df (pandas.DataFrame)
            DataFrame avec : une ligne par image, une colonne 'image_id' contenant le nom de l'image et une colonne par attribut (valeurs : 1 ou -1)
    
    '''
    with open(file_path, 'r', encoding = 'utf-8') as file :
        lignes = file.readlines() #lecture du fichier d'attributs
    
    attributs = lignes[1].strip().split() #la ligne 1 du fichier contient tous les noms d'attributs (découpage de la ligne selon les espaces)
    
    
    data = []
    for ligne in lignes[2:]: # 2 parce que les données des images commencent à la 3e ligne du fichier original donc à l'indice 2 en python 
        donnees = ligne.strip().split() #decoupage de la ligne selon les espaces
        id_image = donnees[0]  #recuparation de l'id de l'image 
        donnees_attr = list(map(int, donnees[1:] ))# données de la ligne qui correspondent aux attributs (mappés en int puis tranformés en liste)
        data.append([id_image] + donnees_attr) #cration d'une nouvelle loste contenant : nom  de l'image + suivi de tous les attributs

    
    #Creation du data frame : 
    df = pd.DataFrame(data, columns = ['image_id'] + attributs)
    return df 


def load_identities(file_path = 'dataset/identity_CelebA.txt') : 
    '''
    Le data set CelebA contient au total 202599 images de 10 177 personnes. Ainsi, plusieurs images sont susceptibles d'être associée à la même personne (la même 'identité').
    Le fichier texte 'identity_CelebA.txt' associe chaque image à un numéro d'identité (exemple : 000010.jpg 612)

    Cette fonction charge ce fichier sous frome de DataFrame pandas.

    Paramètres : 
        file_path (str)
            Chemin vers le fichier texte contenant les données d'idendités
    
    Retour : 
        df (pandas.DataFrame)
            DataFrame avec une ligne par image, une colonne 'image_id' contenant le nom de l'image et une colonne 'identity' contenant le numéro d'identité correspondante à l'image     

    '''
    with open(file_path, 'r', encoding = 'utf-8') as file :
        lignes = file.readlines() #lecture du fichier d'attributs
    
    attributs = ['image_id','identity']
    data = []
    for ligne in lignes: 
        donnees = ligne.strip().split() #decoupage de la ligne selon les espaces
        data.append(donnees)
    
    df_identity = pd.DataFrame(data, columns = attributs)
    return df_identity


def merge_attributes_id(df_attributes, df_identities):
    '''
    Le datasey CelebA fournit les informations sous forme de deux fichiers : 
        - un fichier d'attributs décrivant les caractéritsiques des images (lunettes, sourire,...) 
        - un fichier d'identités associant chaque image à une personne (il y a au total 10 177 idendités )
    Ces informatons étant séparées, il est nécessaire de les fusionner afin d'obtenir à la fin un tableau unique contenant à la fois les attributs et l'identité de chaque image. 

    
    Cette fonction fusionne un DataFrame d'attributs et un DataFrame d'identités à l'aide de la colonne commune 'image_id'

    Parametres : 
        df_attributs (pandas.DataFrame)
            DataFrame contenant une colonne 'image_id' ainsi que des colonnes d'attributs (Eyesglasses,Black_hair,...)
        
        df_identities (pandas.DataFrame)
            DataFrame contenant une colonne 'image_id' ainsi qu'une colonne 'identity_id'
    
    Retour : 
        df_merged (pandas.DataFrame)
            DataFrame fusionné contant pour chaque image son nom (image_id), ses attributs et son identifiant de personne
    '''

    df_merged = df_attributes.merge(df_identities, on = 'image_id')
    return df_merged



def build_requirements(sexe, couleur_chev, type_chev, pilosite=None, visage=None, accessoires=None): 
    
    '''
    Cette fonction construit un dictionnaire de contraintes à partir des choix effectués oar l'utilisateur lors du questionnaire. 
    Elle assure la correspondance entre les r"ponses du questionnaire (exprimées avec des termes compréhensibles par l'utilisateur) et les noms des attributs du dataset CelebA.
    
    Paramètres : 
        Réponse du questionnaire (sexe, cheveux, accessoires,...) provenant de l'interface Gradio.
    
    Retour : 
        requirements (dict) 
            Dictionnaire contenant les contraintes à appliquer pour filtrer les images du dataset CelebA.     
    '''
    requirements = {}

    #-------------------------------------------------------------------------------------
    # Création du dictionnaire requirements basés sur les réponses de l'utilisateur
    #-------------------------------------------------------------------------------------
    # Sexe
    if sexe in SEXE_MAP:
        requirements.update(SEXE_MAP[sexe])

    # Couleur des cheveux
    if couleur_chev in HAIR_COLOR_MAP:
        attr = HAIR_COLOR_MAP[couleur_chev]
        requirements[attr] = 1

    # Type de cheveux
    if type_chev in HAIR_TYPE_MAP:
        attr = HAIR_TYPE_MAP[type_chev]
        requirements[attr] = 1

    print(f'Verif_requirements : {requirements}')
    
    return requirements


def filtrage_dataset(df_celebA, requirements) : 
    '''
    Cette fonction filtre un df d'images selon un ensemble de contraintes définies apr l'utilisateur. Elle applique successivement des conditions sur les
    colonnes du df afin de conserver que les images correspondants aux attributs demandés.

    Parametres : 
        df_celebA (pandas.DataFrame)
            DataFrame contenant les images et leurs attributs
        
        requirements (dict)
            Dictionnaire de contraintes définies par l'utilisateur (choix des attributs)
    
    Retour : 
        filtered_df (pandas.DataFrame)
            Sous-ensemble du DataFrame initial contenant uniquement les images respectant toutes les contraintes. 
    '''
    filtered_df = df_celebA.copy()

    for attribut, value in requirements.items():
        if attribut in filtered_df.columns : 
            filtered_df = filtered_df[filtered_df[attribut] == value] #garde seulement les lignes où la colonne de l'attribut  = valeur 
        
    return filtered_df


#-----------------------------------------------
# Tests de fonctions : 
#-----------------------------------------------
if __name__ == "__main__":
    
    print('Test fonction load_attributes')
    df_attr = load_attributes(file_path = 'dataset/list_attr_celeba.txt')
    print(f'Taille tableau : {df_attr.shape}') #devrait donner 202599 lignes et 41 colonnes (image_id + 40 attributs)
    print(df_attr.head())
    

    print('Test fonction load_identities')
    df_id = load_identities(file_path = 'dataset/identity_CelebA.txt') 
    print(f'Taille tableau : {df_id.shape}') #devrait donner 202599 lignes et 2 colonnes (image_id + identity)
    print(df_id.head())


    print('Test fonction merge_attributes_id')
    df_merge = merge_attributes_id(df_attr, df_id)
    print(f'Taille tableau : {df_merge.shape}') #devrait donner 202599 lignes et 42 colonnes (image_id + 40 attributs + identity)
    print(df_merge.head())






