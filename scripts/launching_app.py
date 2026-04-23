# Ce script lance l'application Gradio pour la génération de portraits robots à partir du dataset CelebA.
# Il utilise les fonctions définies dans source/app.py pour gérer les différentes étapes de l'application
# notamment la sélection d'images, la reconstruction, l'interpolation, la fusion et l'application de modifications basées sur les directions latentes calculées précédemment
# L'interface guide l utilisateur à travers un questionnaire initial, la sélection d'images, et un second questionnaire pour affiner les modifications souhaitées sur le portrait généré

import sys
import os

# Ajoute le dossier parent au chemin de recherche de Python
# Cela permet de trouver le dossier 'source' depuis le dossier 'scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
from PIL import Image
from torch import le
from source.app import selection_images, retour_selection, retour_questionnaire, appliquer_modifications, refine_with_genetic_algorithm
from source.latent_space import reconstruct_image, fusion_2_images, fusion_3_images, load_directions

dico_direction = load_directions()
refinable_attributes = ["Aucun changement"] + sorted(list(dico_direction.keys()))

#======================================================================
# INTERFACE
#======================================================================

with gr.Blocks() as demo:
    gr.Markdown("# <span style='color: purple; font-size: 2em;'>Robot Portrait Generator</span>")
    gr.Markdown("Bienvenue dans le générateur de robot portrait ! Suivez les étapes pour créer votre portrait robot personnalisé à partir du dataset CelebA. " \
    "Commencez par répondre au questionnaire pour décrire les caractéristiques générales du visage que vous souhaitez générer, puis sélectionnez une ou plusieurs " \
    "images proposées en fonction de vos réponses pour la reconstruction, l'interpolation ou la fusion. Enfin, affinez votre portrait robot en choisissant les modifications " \
    "que vous souhaitez apporter dans le second questionnaire.")

    #------------------------------------------------------------------------------------------------------------
    #  Etape 1  : questionnaire 1 : seules les caractéristiques principales sont mises en avant 
    #  cette étape contient le premier questionnaire qui permet à l'utilisateur de décrire les caractéristiques générales du visage qu'il souhaite générer (sexe, couleur et type de cheveux). 
    #  En fonction des réponses, des propositions d'images seront affichées dans la colonne suivante pour que l'utilisateur puisse sélectionner une ou plusieurs images 
    #  comme point de départ pour la reconstruction ou l'interpolation.
    #------------------------------------------------------------------------------------------------------------ 
    
    with gr.Column(visible=True) as questionnaire:
    
        gr.Markdown("## <span style='color: blue; font-size: 1.5em;'>Questionnaire 1</span>")
        gr.Markdown("Décrivez les caractéristiques générales du visage") 

        # Sexe
        gr.Markdown("### Sexe")
        sexe = gr.Dropdown(['Homme','Femme', 'Non précisé'], label="Sexe")

        # Cheveux
        gr.Markdown("### Cheveux")
        couleur_chev = gr.Dropdown(['Blond','Brun','Noir','Gris','Chauve', 'Non précisé'], label="Couleur des cheveux")
        type_chev = gr.Dropdown(['Raides','Ondulés', 'Non précisé'], label="Type de cheveux")
        
        button1 = gr.Button('Voir les propositions ') # ce bouton permet de passer à l'étape suivante : la sélection d'images proposées en fonction des réponses au questionnaire 1 (sexe, couleur et type de cheveux)

    #--------------------------------------------------------------------------------------------------------------------------------
    # Etape 2 : Slection d'images initiales
    # cette étape affiche les propositions d'images en fonction des réponses au questionnaire 1
    # et permet à l'utilisateur de sélectionner une ou plusieurs images pour la reconstruction, l'interpolation ou la fusion
    #--------------------------------------------------------------------------------------------------------------------------------
    with gr.Column(visible=False) as images_select:
        
        gr.Markdown("## Sélectionnez une ou plusieurs images")
        
        info = gr.Textbox(label="Informations", interactive=False)
        gr.Markdown(""" ## Que voulez-vous faire ?
                    - **Reconstruction** : sélectionnez exactement **1 image**
                    - **Fusion 2 images ** : sélectionnez exactement **2 images**
                    - **Fusion 3 images** : sélectionnez exactement **3 images**""")

        #--------------------------------------------------------------------------------------------
        # States qui stockent les chemins : 
        # les states permettent de stocker les chemins des images sélectionnées pour les utiliser 
        # dans les fonctions de reconstruction, d'interpolation et de fusion
        #--------------------------------------------------------------------------------------------
        path1 = gr.State()
        path2 = gr.State()
        path3 = gr.State()
        path4 = gr.State()
        path5 = gr.State()
        path6 = gr.State()

    # Les gr.Row permettent d'organiser les éléments de l'interface en lignes, ici on a deux lignes de trois images chacune pour afficher les propositions d'images à sélectionner par l'utilisateur
    # Chaque image est accompagnée d'une checkbox pour permettre la sélection        

        with gr.Row():
            with gr.Column(visible=False) as bloc1:
                img1 = gr.Image(type="pil", label="Image 1")
                check1 = gr.Checkbox(label="Sélectionner cette image")

            with gr.Column(visible=False) as bloc2:
                img2 = gr.Image(type="pil", label="Image 2")
                check2 = gr.Checkbox(label="Sélectionner cette image")

            with gr.Column(visible=False) as bloc3:
                img3 = gr.Image(type="pil", label="Image 3")
                check3 = gr.Checkbox(label="Sélectionner cette image") 

        
        with gr.Row():
            with gr.Column(visible=False) as bloc4:
                img4 = gr.Image(type="pil", label="Image 4")
                check4 = gr.Checkbox(label="Sélectionner cette image")

            with gr.Column(visible=False) as bloc5:
                img5 = gr.Image(type="pil", label="Image 5")
                check5 = gr.Checkbox(label="Sélectionner cette image")

            with gr.Column(visible=False) as bloc6:
                img6 = gr.Image(type="pil", label="Image 6")
                check6 = gr.Checkbox(label="Sélectionner cette image") 


        with gr.Row():
            reconstruction_button = gr.Button("Reconstruction")
            fusion2_button = gr.Button("Fusion 2 images")
            fusion3_button = gr.Button("Fusion 3 images")

        button_retour_quest = gr.Button("Retour au questionnaire") ## ce bouton permet à l'utilisateur de revenir au questionnaire 1 pour modifier ses réponses (sexe, couleur et type de cheveux) 
        
            
    
    #----------------------------------------------------------------
    # Etape 3 : Resultats
    # zone de résultats qui s'affiche après que l'utilisateur ait sélectionné 
    # les images et choisi une action (reconstruction ou interpolation ou fusion)
    #---------------------------------------------------------------- 

    with gr.Column(visible=False) as result_zone:
        gr.Markdown("## Résultat")

        result_text = gr.Textbox(label="Message",interactive=False)

        result_image = gr.Image(type="pil", label="Image résultat")

        button_retour_selection = gr.Button("Retour à la selection")

    
    #--------------------------------
    # Fonctions de traitement Gradio
    #--------------------------------
    # le code ci-dessous fait le lien entre les éléments de l'interface et les fonctions de traitement définies dans source/app.py et source/latent_space.py
    # notamment la fonction selection_images qui propose des images en fonction des réponses au questionnaire 1, la fonction recuperer_selection qui récupère les images sélectionnées par l'utilisateur, et les fonctions eval_reconstruction et eval_interpolation 
    # qui vérifient que le nombre d'images sélectionnées correspond à l'action choisie par l'utilisateur (reconstruction/ interpolation/fusion) et qui appellent ensuite les fonctions de reconstruction et d'interpolation définies dans source/latent_space.py pour
    # générer le portrait robot final à partir des images sélectionnées
    
    def eval_reconstruction(i1, i2, i3, i4, i5, i6, c1, c2, c3, c4, c5, c6):
        '''
        Cette fonction est appelée lorsque l'utilisateur clique sur le bouton "Reconstruction"
        elle récupère les images sélectionnées par l'utilisateur et vérifie que le nombre d'images sélectionnées est bien égal à 1, sinon elle affiche un message d'erreur
        si une seule image est sélectionnée, elle appelle la fonction reconstruct_image de source/latent_space.py pour générer le portrait robot à partir de l'image sélectionnée, elle affiche ensuite le résultat et un message de succès
        et elle cache la zone de sélection d'images pour afficher uniquement le résultat et le questionnaire 2
        
        Paramètres : les 6 chemins potentiels des images et les 6 checkboxes associés
        Retour : l'image résultat de la reconstruction, un message de succès ou d'erreur, la visibilité de la zone de résultat, la visibilité de la zone de sélection d'images et la visibilité du questionnaire 2
        
        '''

        images = [i1, i2, i3, i4, i5, i6]
        checks = [c1, c2, c3, c4, c5, c6]
        selection = [img for img, c in zip(images, checks) if c and img is not None]
        
        if len(selection) != 1:
            # Reste sur la sélection, met un message d'erreur
            return None, "Erreur : Veuillez sélectionner exactement 1 image pour la reconstruction.", gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            
        res_img = reconstruct_image(selection[0])
        # Cache la sélection, affiche le résultat
        return res_img, "Reconstruction réussie !", gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)

    def eval_fusion2(i1, i2, i3, i4, i5, i6, c1, c2, c3, c4, c5, c6):

        '''
        Cette fonction est appelée lorsque l'utilisateur clique sur le bouton "Fusion 2 images"
        elle récupère les images sélectionnées par l'utilisateur et vérifie que le nombre d'images sélectionnées est bien égal à 2, sinon elle affiche un message d'erreur
        si deux images sont sélectionnées, elle appelle la fonction fusion_2_images de source/latent_space.py pour générer le portrait robot à partir des images sélectionnées, elle affiche ensuite le résultat et un message de succès
        et elle cache la zone de sélection d'images pour afficher uniquement le résultat et le questionnaire 2
        
        Paramètres : les 6 chemins potentiels des images et les 6 checkboxes associés
        Retour : l'image résultat de la fusion, un message de succès ou d'erreur, la visibilité de la zone de résultat, la visibilité de la zone de sélection d'images et la visibilité du questionnaire 2
        
        '''

        images = [i1, i2, i3, i4, i5, i6]
        checks = [c1, c2, c3, c4, c5, c6]
        selection = [img for img, c in zip(images, checks) if c and img is not None]
        
        if len(selection) != 2:
            return None, "Erreur : Veuillez sélectionner exactement 2 images pour la fusion.", gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            
        res_img = fusion_2_images(selection[0], selection[1])
        return res_img, "Fusion réussie !", gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)

    
    def eval_fusion3(i1, i2, i3, i4, i5, i6, c1, c2, c3, c4, c5, c6):
        '''
        Cette fonction est appelée lorsque l'utilisateur clique sur le bouton "Fusion 3 images"
        elle récupère les images sélectionnées par l'utilisateur et vérifie que le nombre d'images sélectionnées est bien égal à 3, sinon elle affiche un message d'erreur
        si trois images sont sélectionnées, elle appelle la fonction de fusion (à définir) pour générer le portrait robot à partir des images sélectionnées, elle affiche ensuite le résultat et un message de succès
        et elle cache la zone de sélection d'images pour afficher uniquement le résultat et le questionnaire 2
        
        Paramètres : les 6 chemins potentiels des images et les 6 checkboxes associés
        Retour : l'image résultat de la fusion, un message de succès ou d'erreur, la visibilité de la zone de résultat, la visibilité de la zone de sélection d'images et la visibilité du questionnaire 2
        
        '''

        images = [i1, i2, i3, i4, i5, i6]
        checks = [c1, c2, c3, c4, c5, c6]
        selection = [img for img, c in zip(images, checks) if c and img is not None]
        
        if len(selection) != 3:
            return None, "Erreur : Veuillez sélectionner exactement 3 images pour la fusion.", gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            
        res_img = fusion_3_images(selection[0], selection[1], selection[2]) 
        return res_img, "Fusion réussie !", gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
    
    #----------------------------------------------------------------
    # Etape 4 : Questionnaire 2 
    # (propositions de changements : ajout/suppression caractéristiques)
    #----------------------------------------------------------------
    with gr.Column(visible=False) as questionnaire2:
        gr.Markdown("## <span style='color: blue; font-size: 1.5em;'>Questionnaire 2 : Ajustements</span>")
        
        with gr.Tabs():
            with gr.TabItem("Ajustement Standard"):
                gr.Markdown("Modifiez le portrait en ajoutant ou changeant des caractéristiques.")
                couleur_chev_q2 = gr.Dropdown(['Aucun changement', 'Blond', 'Brun', 'Noir', 'Gris', 'Chauve'], label="Changer la couleur des cheveux")
                type_chev_q2 = gr.Dropdown(['Aucun changement', 'Raides', 'Ondulés'], label="Changer le type de cheveux")
                pilosite_q2 = gr.CheckboxGroup(["Barbe", "Moustache", "Bouc", "Frange", "Sideburns", "Calvitie frontale"], label="Modifications de pilosité")
                visage_q2 = gr.CheckboxGroup(["Joues rosées", "Nez pointu", "Peau pâle", "Visage ovale", "Yeux étroits", "Pommettes hautes", "Double menton","Sourcils épais", "Gros nez", "Lèvres pulpeuses", "Cernes"], label="Forme et traits du visage")
                accessoires_q2 = gr.CheckboxGroup(["Lunettes", "Maquillage prononcé", "Boucles d'oreilles", "Chapeau", "Rouge à lèvres", "Collier", "Cravate"], label="Modifications d'accessoires")
                appliquer_modifs_button = gr.Button("Appliquer les modifications")

            with gr.TabItem("Affinage par Algorithme Génétique (Expérimental)"):
                gr.Markdown("Sélectionnez un attribut et ajustez les paramètres pour optimiser le portrait.")
                
                attribute_to_refine_dropdown = gr.Dropdown(
                    refinable_attributes, value="Aucun changement", label="Attribut à affiner"
                )
                
                with gr.Accordion("Paramètres de l'Algorithme Génétique", open=False):
                    ga_gens = gr.Slider(2, 30, value=10, step=1, label="Générations", info="Nombre de générations pour l'évolution génétique")
                    ga_pop_size = gr.Slider(5, 30, value=10, step=2, label="Taille de la Population", info="Nombre d'individus dans chaque génération")
                    ga_mut_rate = gr.Slider(0.001, 0.5, value=0.03, label="Taux de Mutation", info="Taux de mutation pour les individus")
                    ga_cross_rate = gr.Slider(0.1, 1.0, value=0.8, label="Taux de Crossover", info="Taux de crossover pour la reproduction. Le crossover combine les caractéristiques de deux parents pour créer un enfant. Un taux plus élevé favorise la diversité génétique, tandis qu'un taux plus bas favorise la préservation des traits existants.")

                refine_button = gr.Button("Affiner avec l'Algorithme Génétique", variant="primary")
    #--------------------------------
    # Boutons 
    #-------------------------------- 
    
    button1.click(selection_images, inputs=[sexe, couleur_chev, type_chev], outputs=[img1, img2, img3, img4, img5, img6, path1, path2, path3, path4, path5, path6, bloc1, bloc2, bloc3, bloc4, bloc5, bloc6, questionnaire, images_select, info]) #bouton qui affiche les propositions d'images en fonction des réponses au questionnaire 1, il rend visible la zone de sélection d'images et cache le questionnaire 1
    button_retour_quest.click(retour_questionnaire, inputs=[], outputs=[questionnaire, images_select]) #bouton qui permet de revenir au questionnaire 1, il rend visible le questionnaire 1 et cache la zone de sélection d'images
    button_retour_selection.click(retour_selection, inputs = [], outputs = [images_select, result_zone]) #bouton qui permet de revenir à la sélection d'images après avoir vu le résultat, il rend visible la zone de sélection d'images et cache la zone de résultat et le questionnaire 2
    all_imgs_and_checks = [img1, img2, img3, img4, img5, img6, check1, check2, check3, check4, check5, check6] #bouton qui permet de récupérer les images sélectionnées et les checkboxes associés pour les fonctions de reconstruction et d'interpolation
    reconstruction_button.click(eval_reconstruction, inputs=all_imgs_and_checks, outputs=[result_image, result_text, result_zone, images_select, questionnaire2]) #bouton qui permet de lancer la fonction de reconstruction, il vérifie que le nombre d'images sélectionnées est bien égal à 1, sinon il affiche un message d'erreur, si une seule image est sélectionnée, il affiche le résultat de la reconstruction et le questionnaire 2 pour les modifications basées sur les directions latentes
    fusion2_button.click(eval_fusion2, inputs=all_imgs_and_checks, outputs=[result_image, result_text, result_zone, images_select, questionnaire2]) #bouton qui permet de lancer la fonction d'interpolation, il vérifie que le nombre d'images sélectionnées est bien égal à 2, sinon il affiche un message d'erreur, si deux images sont sélectionnées, il affiche le résultat de l'interpolation et le questionnaire 2 pour les modifications basées sur les directions latentes
    fusion3_button.click(eval_fusion3, inputs=all_imgs_and_checks, outputs=[result_image, result_text, result_zone, images_select, questionnaire2]) #bouton qui permet de lancer la fonction de fusion, il vérifie que le nombre d'images sélectionnées est bien égal à 3, sinon il affiche un message d'erreur, si trois images sont sélectionnées, il affiche le résultat de la fusion et le questionnaire 2 pour les modifications basées sur les directions latentes
    appliquer_modifs_button.click(appliquer_modifications, inputs=[result_image, couleur_chev_q2, type_chev_q2, pilosite_q2, visage_q2, accessoires_q2], outputs=[result_image, result_text]) #bouton qui permet d'appliquer les modifications choisies dans le questionnaire 2, il utilise les directions latentes calculées précédemment pour modifier le portrait généré en fonction des changements demandés par l'utilisateur, il affiche ensuite le résultat modifié et un message de succès
    refine_button.click(
        fn=refine_with_genetic_algorithm,
        inputs=[result_image, attribute_to_refine_dropdown, ga_gens, ga_pop_size, ga_mut_rate, ga_cross_rate],
        outputs=[result_image, result_text]
    )


def main(): #sert à lancer l'application Gradio, elle est appelée à la fin du script pour démarrer l'interface et rendre le serveur accessible à l'adresse http://localhost:7860 depuis un autre appareil
    demo.queue().launch(server_name="0.0.0.0", server_port=7860) # le serveur est accessible à l'adresse http://localhost:7860  depuis un autre appareil

if __name__ == "__main__": #sert a vérifier que le script est exécuté directement et non importé en tant que module, si c'est le cas, il appelle la fonction main() pour lancer l'application Gradio
    main()

