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

- Python 3.10 ou supérieur recommandé
- pip installé
- Prévoir plusieurs Go d’espace libre pour le dataset CelebA

Le projet fonctionne sur CPU. Un GPU accélère les calculs mais n’est pas obligatoire.

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
cd Prog_logiciel_autoportrait
```


## 2. CRÉER UN ENVIRONNEMENT VIRTUEL PYTHON
Windows :

```bash 
python -m venv .venv
.venv\Scripts\activate
```
Linux / macOS :

```bash 
python3 -m venv .venv
source .venv/bin/activate
```

Une fois activé, le terminal affiche généralement : (.venv)


## 3.  INSTALLER LES DÉPENDANCES

```bash 
pip install -r requirements.txt
```


## 4.  CRÉER LE DOSSIER DATASET


À la racine du projet, exécuter : 

```bash 
mkdir dataset
```

## 5. TÉLÉCHARGER LE DATASET CELEBA

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

## 6. GÉNÉRER LES FICHIERS NÉCESSAIRES AU LANCEMENT DE  L'APPLICATION

Deux scripts doivent être exécutés une première fois avant de lancer l'interface graphique : 

a. Calcul des vecteurs latents. En partant de la racine du projet (donc en étant seulement dans le dossier Prog_logiciel_autoportrait/), exécuter la commande : 

```bash
python -m scripts.compute_latent_vectors
```

Ce script génère : dataset/df_latents.pkl

b. Calcul des directions latentes (pour modification d'attributs). En partant de la racine du projet (donc en étant seulement dans le Prog_logiciel_autoportrait/), exécuter la commande : 

```bash
python -m scripts.compute_latent_directions
```

Ce script génère : dataset/directions_latentes.npy

Après exécution des deux scripts, le dossier dataset/ doit contenir :

```text
dataset/
├── identity_CelebA.txt
├── list_attr_celeba.txt
├── img_align_celeba/
├── df_latents.pkl
└── directions_latentes.npy
```

## 7. LANCER L’APPLICATION

En partant de la racine du projet (donc en étant seulement dans le dossier Prog_logiciel_autoportrait/)
exécuter la commande : 

```bash
python -m scripts.launching_app
```

Le terminal affichera une adresse locale du type : http://127.0.0.1:7860
Ouvrir ce lien dans un navigateur (appuyer sur ctrl + lien)


> **Remarque :**  
> Une fois les fichiers générés (df_latents.pkl et directions_latentes.npy),
> il n’est pas nécessaire de relancer les deux scripts.

Il suffit de refaire :
Windows : 

```bash
.venv\Scripts\activate
```

Linux / macOS :

```bash
source .venv/bin/activate
```

Puis depuis la racine du projet : 

```bash
python -m scripts.launching_app
```

En cas d'erreur pendant le lancement, vérifier que :
- dataset/ existe
- identity_CelebA.txt est présent
- list_attr_celeba.txt est présent
- img_align_celeba/ contient les images
- df_latents.pkl a bien été généré
- directions_latentes.npy a bien été généré
Si une bibliothèque manque : pip install -r requirements.txt


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
4. Modifier le résultat avec un second questionnaire


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

Si plusieurs images ou aucune image sont sélectionnées, un message d’erreur apparaît.

## 2. Interpolation

### Principe

Sélectionner exactement **2 images**.

Le modèle encode les deux visages dans l’espace latent puis crée un visage intermédiaire.

### Objectif

Obtenir un portrait mélangeant progressivement les deux visages.

### Bouton

`Interpolation`

Si le nombre d’images sélectionnées n’est pas égal à 2, un message d’erreur apparaît.

## 3. Fusion

### Principe

Sélectionner exactement **3 images**.

Le modèle combine les représentations latentes de trois visages.

### Objectif

Générer un portrait synthétique inspiré des trois images choisies.

### Bouton

`Fusion`

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

Après génération du portrait, un second questionnaire apparaît.

### Objectif

Modifier le visage obtenu sans recommencer tout le processus.

Les modifications sont appliquées grâce aux directions latentes calculées à partir du dataset.

## MODIFICATIONS DISPONIBLES

### 1. Cheveux

Changer la couleur :

- Blond
- Brun
- Noir
- Gris
- Chauve
- Aucun changement

Changer le type :

- Raides
- Ondulés
- Aucun changement

### 2. Pilosité

Cases multiples :

- Barbe
- Moustache
- Bouc
- Frange
- Sideburns
- Calvitie frontale

### 3. Forme et traits du visage

Cases multiples :

- Joues rosées
- Nez pointu
- Peau pâle
- Visage ovale
- Yeux étroits
- Pommettes hautes
- Double menton
- Sourcils épais
- Gros nez
- Lèvres pulpeuses
- Cernes

### 4. Accessoires

Cases multiples :

- Lunettes
- Maquillage prononcé
- Boucles d’oreilles
- Chapeau
- Rouge à lèvres
- Collier
- Cravate


## VALIDATION

Cliquer sur :

`Appliquer les modifications`

Le portrait est recalculé automatiquement et remplacé par la nouvelle version.


## EXEMPLES D’UTILISATION

### Exemple 1

- Femme
- Blond
- Ondulés

Puis sélectionner 2 images :

**Interpolation**

Ensuite ajouter :

- Lunettes
- Rouge à lèvres

### Exemple 2

- Homme
- Brun
- Raides

Puis sélectionner 1 image :

**Reconstruction**

Ensuite ajouter :

- Barbe
- Sourcils épais


## CONSEILS

- Tester plusieurs sélections pour obtenir des visages variés
- Utiliser **Non précisé** pour avoir davantage de résultats
- Les modifications peuvent être cumulées
- Certaines transformations sont plus visibles que d’autres selon l’image de départ


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

