'''
Ce fichier centralise toutes les correspondances entre :
- les termes affichés à l'utilisateur dans l'interface
- les noms d'attributs du dataset CelebA
'''


#-------------------------------------------------------------------------------------
    # Correspondance entre les choix dans le questionnaire et attributs du dataset 
#-------------------------------------------------------------------------------------

# Sexe
SEXE_MAP = {"Homme": {"Male": 1}, "Femme": {"Male": -1}}

# Couleur des cheveux
HAIR_COLOR_MAP = {"Noir": "Black_Hair", "Blond": "Blond_Hair", "Brun": "Brown_Hair", "Chauve": "Bald", "Gris": "Gray_Hair"}

# Type de cheveux
HAIR_TYPE_MAP = {"Raides": "Straight_Hair", "Ondulés": "Wavy_Hair"}

# Pilosité
PILOSITY_MAP = {"Barbe": "5_o_Clock_Shadow", "Moustache": "Mustache", "Bouc": "Goatee", "Frange": "Bangs", "Sideburns": "Sideburns", "Calvitie frontale": "Receding_Hairline"}

# Traits du visage
FACIAL_FEATURES_MAP = {"Joues rosées": "Rosy_Cheeks", "Nez pointu": "Pointy_Nose", "Peau pâle": "Pale_Skin", "Visage ovale": "Oval_Face", "Yeux étroits": "Narrow_Eyes", "Pommettes hautes": "High_Cheekbones",
                        "Bouche entrouverte": "Mouth_Slightly_Open", "Double menton": "Double_Chin", "Sourcils épais": "Bushy_Eyebrows", "Gros nez": "Big_Nose", "Lèvres pulpeuses": "Big_Lips", "Cernes": "Bags_Under_Eyes"}

# Accessoires
ACCESSORIES_MAP = {"Lunettes": "Eyeglasses", "Maquillage prononcé": "Heavy_Makeup", "Boucles d'oreilles": "Wearing_Earrings", "Chapeau": "Wearing_Hat", "Rouge à lèvres": "Wearing_Lipstick", "Collier": "Wearing_Necklace", "Cravate": "Wearing_Necktie"}