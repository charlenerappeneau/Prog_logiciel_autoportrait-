# Projet Robot Generator

Application de génération et modification de portraits basée sur un **Variational Autoencoder (VAE)** entraîné sur le dataset **CelebA**.

## L’application permet :

- de proposer des visages selon des caractéristiques choisies
- de sélectionner des images
- d’effectuer des transformations dans l’espace latent
- de générer de nouveaux portraits via une interface **Gradio**


# Dataset utilisé : CelebA

L’application repose sur le dataset **CelebFaces Attributes Dataset (CelebA)**, une base de données largement utilisée en vision par ordinateur pour les tâches liées aux visages.

## CelebA contient :

- plus de **202 599 images** de visages
- plus de **10 000 identités différentes**
- **40 attributs binaires** annotés manuellement pour chaque visage

## Exemples d’attributs disponibles :

- sexe (`Male`)
- cheveux blonds / bruns / noirs / gris
- barbe
- moustache
- lunettes
- visage ovale
- ...

## Codage des attributs :

- `1` = attribut présent
- `-1` = attribut absent

## Pourquoi CelebA ?

CelebA a été choisi car il permet :

- de filtrer les images selon les caractéristiques choisies par l’utilisateur
- d’entraîner un modèle génératif sur un grand nombre de visages réels
- de modifier certaines caractéristiques via l’espace latent du VAE
- de produire des portraits variés et cohérents

## Principaux fichiers exploités :

- `img_align_celeba/` : dossier contenant les images
- `list_attr_celeba.txt` : attributs des visages
- `identity_CelebA.txt` : identifiants des personnes

> **Remarque :**  
> Les images du dataset proviennent de célébrités.  
> Le projet les utilise uniquement dans un cadre pédagogique et expérimental.

# A. INSTALLATION COMPLÈTE DU PROJET

## 0. PRÉREQUIS

- Docker Desktop installé (Windows / Mac) ou Docker Engine (Linux)
- Docker lancé et fonctionnel
- Prévoir plusieurs Go d’espace libre pour le dataset CelebA

Le projet s’exécute dans un conteneur Docker.
Il n’est pas nécessaire d’installer Python, pip ou de créer un environnement virtuel.

Suivre les étapes ci-dessous dans l’ordre.


## 1. CLONER LE PROJET

Ouvrir un terminal puis exécuter (selon SSH ou HTTPS au choix) :

### Version HTTPS

```bash
git clone https://github.com/charlenerappeneau/Prog_logiciel_autoportrait-.git
```

### Version HTTPS

```bash 
git clone git@github.com:charlenerappeneau/Prog_logiciel_autoportrait-.git
```

Puis aller dans le dossier du projet en exécutant : 

```bash 
cd Prog_logiciel_autoportrait-
cd Prog_logiciel_autoportrait-
```

## 2.  CRÉER LE DOSSIER DATASET

À la racine du projet, exécuter : 

```bash 
mkdir dataset
```

## 3. TÉLÉCHARGER LE DATASET CELEBA

Télécharger et placer dans le dossier dataset/ les fichiers suivants : 

a. Dossier d'images 'img_align_celeba.zip' sur le lien : https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ
(Dézipper le dossier avec : clic droit > Extraire tout sous Windows ou `unzip img_align_celeba.zip` sous Linux / macOS)

b. Le fichier 'identity_CelebA.txt' sur le lien : https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs?resourcekey=0-pEjrQoTrlbjZJO2UL8K_WQ 

c. Le fichier 'list_attr_celeba.txt' sur le lien : https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs?resourcekey=0-pEjrQoTrlbjZJO2UL8K_WQ 

L'arborescence doit ressembler à ceci : 

``` text
dataset/
├── identity_CelebA.txt
├── list_attr_celeba.txt
└── img_align_celeba/
    ├── 000001.jpg
    ├── 000002.jpg
    ├── ...
    └── ...
```

## 4. TÉLÉCHARGER LES FICHIERS PRÉCALCULÉS

Télécharger les fichiers suivants depuis ce lien : https://drive.google.com/drive/folders/1wsWlB6VD_bAZ-7T06WZo6hsnNJNX-X9y?usp=sharing

- df_latents.pkl
- directions_latentes.npy

Puis les placer dans :

dataset/

Le dossier dataset/ doit alors contenir :

```text
dataset/
├── identity_CelebA.txt
├── list_attr_celeba.txt
├── img_align_celeba/
├── df_latents.pkl
└── directions_latentes.npy
```

## 5. CONSTRUIRE L’IMAGE DOCKER

Depuis la racine du projet :

```bash
docker build -t portrait-app .
```

Cette étape installe automatiquement toutes les dépendances dans l’image Docker.

## 6. LANCER L’APPLICATION

Depuis la racine du projet :

Linux / macOS / WSL :

```bash
docker run --rm -p 7860:7860 -v $(pwd)/dataset:/app/dataset portrait-app
```

Windows PowerShell :

```bash
docker run --rm -p 7860:7860 -v ${PWD}/dataset:/app/dataset portrait-app
```

Le terminal affichera une adresse locale du type : http://127.0.0.1:7860
Ouvrir ce lien dans un navigateur.



>**Remarques**
>Les fichiers :
>- df_latents.pkl
>- directions_latentes.npy
>sont déjà fournis. Il n’est donc pas nécessaire de lancer :
>- compute_latent_vectors.py
>- compute_latent_directions.py



## UTILISATIONS SUIVANTES

Il suffit de relancer :

Linux / macOS / WSL :

```bash
docker run --rm -p 7860:7860 -v $(pwd)/dataset:/app/dataset portrait-app
```

Windows PowerShell :

```bash
docker run --rm -p 7860:7860 -v ${PWD}/dataset:/app/dataset portrait-app
```


## EN CAS D’ERREUR

Vérifier que :

- Docker est lancé
- le dossier dataset/ existe
- identity_CelebA.txt est présent
- list_attr_celeba.txt est présent
- img_align_celeba/ contient les images
- df_latents.pkl est présent
- directions_latentes.npy est présent

Vérifier également que la commande est lancée depuis la racine du projet.

# B. LANCEMENT DE L'INTERFACE : GENERATION DE PORTRAITS

Une fois l’application lancée, une interface web s’ouvre automatiquement dans votre navigateur.
Adresse locale habituelle : http://127.0.0.1:7860

Cette interface permet de générer un portrait robot à partir du dataset CelebA, en combinant :

- un questionnaire utilisateur
- une sélection d’images proposées
- des transformations dans l’espace latent du VAE


## PRINCIPE GÉNÉRAL

L’application fonctionne en 4 grandes étapes :

1. Décrire un visage avec un premier questionnaire
2. Sélectionner une ou plusieurs images proposées
3. Générer un portrait robot (reconstruction / interpolation / fusion)
4. Modifier et affiner le résultat avec un second questionnaire


## ÉTAPE 1 : QUESTIONNAIRE INITIAL

Au lancement, l’utilisateur arrive sur le **Questionnaire 1**.

### Objectif

Décrire les caractéristiques générales du visage recherché.

### Champs disponibles

1. Sexe : Homme / Femme / Non précisé  
2. Couleur des cheveux : Blond / Brun / Noir / Gris / Chauve / Non précisé  
3. Type de cheveux : Raides / Ondulés / Non précisé

> Si **Non précisé** est choisi, ce critère n’est pas utilisé pour filtrer les images.

Une fois les choix effectués, cliquer sur :

`Voir les propositions`


## ÉTAPE 2 : PROPOSITIONS D’IMAGES

L’application filtre automatiquement le dataset CelebA selon les réponses du questionnaire.

Elle affiche ensuite jusqu’à **6 visages** correspondant aux critères.

Chaque image possède une case à cocher permettant la sélection.

L’utilisateur doit ensuite choisir l’action souhaitée.

## 1. Reconstruction

### Principe

Sélectionner exactement **1 image**.

Le modèle encode puis reconstruit l’image à travers le VAE.

### Objectif

Observer ce que le modèle a appris et générer une version reconstruite du visage.

### Bouton

`Reconstruction`

## 2. Fusion2

### Principe

Sélectionner exactement **2 images**.

Le modèle encode les deux visages dans l’espace latent puis crée un visage intermédiaire.

### Objectif

Obtenir un portrait mélangeant progressivement les deux visages.

### Bouton

`Fusion 3 images`

Si le nombre d’images sélectionnées n’est pas égal à 2, un message d’erreur apparaît.

## 3. Fusion3

### Principe

Sélectionner exactement **3 images**.

Le modèle combine les représentations latentes de trois visages.

### Objectif

Générer un portrait synthétique inspiré des trois images choisies.

### Bouton

`Fusion 3 images`

Si le nombre d’images sélectionnées n’est pas égal à 3, un message d’erreur apparaît.


## BOUTON RETOUR

`Retour au questionnaire`

Permet de revenir à l’étape 1 pour modifier les critères.


## ÉTAPE 3 : RÉSULTAT

Après une action valide, la zone **Résultat** s’affiche.

Elle contient :

- un message de confirmation
- l’image générée
- un bouton : `Retour à la sélection`

Ce bouton permet de revenir aux images proposées pour refaire un autre test.


## ÉTAPE 4 : QUESTIONNAIRE D’AJUSTEMENT

Après génération du portrait, un second questionnaire apparaît, organisé en deux onglets.

### Objectif

Modifier le visage obtenu sans recommencer tout le processus.

### Onglet 1 : Ajustement Standard

Cette méthode applique une modification simple et directe. Elle "pousse" le visage dans la direction d'un ou plusieurs attributs.

#### MODIFICATIONS DISPONIBLES

##### 1. Cheveux

Changer la couleur :

- Blond, Brun, Noir, Gris, Chauve

Changer le type :

- Raides, Ondulés

##### 2. Pilosité

- Barbe, Moustache, Bouc, Frange, Sideburns, etc.

##### 3. Forme et traits du visage

- Joues rosées, Nez pointu, Peau pâle, etc.

##### 4. Accessoires

- Lunettes, Maquillage prononcé, Chapeau, etc.

#### VALIDATION

Cliquer sur :

`Appliquer les modifications`

Le portrait est recalculé et remplacé par la nouvelle version.

### Onglet 2 : Affinage par Algorithme Génétique (Expérimental)

Cette méthode avancée utilise un algorithme évolutionniste pour rechercher de manière intensive la meilleure version d'un visage selon **un seul attribut** à la fois. C'est une recherche plus fine et plus puissante que l'ajustement standard.

#### PRINCIPE

1.  **Sélectionner un attribut** : Choisir dans la liste l'unique caractéristique à optimiser (par exemple, `Smiling` ou `Big_Lips`).
2.  **Régler les paramètres** : Ajuster les sliders pour contrôler le processus de recherche (nombre de générations, taille de la population, etc.).
3.  **Lancer l'affinage** : Cliquer sur le bouton `Affiner avec l'Algorithme Génétique`.

L'algorithme va alors créer une population de "candidats" autour de votre image actuelle et les faire "évoluer" sur plusieurs générations pour trouver le visage qui maximise le score de l'attribut choisi.

#### VALIDATION

Cliquer sur :

`Affiner avec l'Algorithme Génétique`

Le portrait est remplacé par le meilleur résultat trouvé par l'algorithme.

## EXEMPLES D’UTILISATION

### Exemple 1

- Femme, Blond, Ondulés
- Sélectionner 2 images → **Interpolation**
- Dans l'onglet "Ajustement Standard", ajouter : `Lunettes`, `Rouge à lèvres`
- Cliquer sur `Appliquer les modifications`.

### Exemple 2

- Homme, Brun, Raides
- Sélectionner 1 image → **Reconstruction**
- Dans l'onglet "Affinage par Algorithme Génétique", sélectionner `Smiling`
- Cliquer sur `Affiner avec l'Algorithme Génétique` pour obtenir une version plus souriante du visage.


## CONSEILS

- Tester plusieurs sélections pour obtenir des visages variés.
- Utiliser **Non précisé** pour avoir davantage de résultats.
- Les modifications de l'ajustement standard peuvent être cumulées.
- L'affinage génétique est plus puissant mais ne traite qu'un attribut à la fois.


## FERMETURE DE L’APPLICATION

Dans le terminal, appuyer sur :

`CTRL + C`


# AUTEURS

Projet réalisé dans le cadre d’un projet de développement logiciel par :

- Inass Ouadrhiri
- Miranda Masini
- Charlène Rappeneau
- Salomé Sonnallier

**INSA LYON - 4A - GROUPE BIM B**