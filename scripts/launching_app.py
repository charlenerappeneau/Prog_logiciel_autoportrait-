import gradio as gr
from PIL import Image
from source.app import selection_images, retour_selection, retour_questionnaire, appliquer_modifications
from source.latent_space import reconstruct_image, interpolate_images


#======================================================================
# INTERFACE
#======================================================================

with gr.Blocks() as demo:

    gr.Markdown("# Portrait Robot Generator")
    gr.Markdown("Projet VAE basé sur CelebA")

    #------------------------------------------------------------------------------------------------------------
    #  Etape 1 questionnaire - seules les caractéristiques principales sont mises en avant (pas les 40 attributs)
    #------------------------------------------------------------------------------------------------------------ 

    with gr.Column(visible=True) as questionnaire:
        gr.Markdown("## Questionnaire 1")
        gr.Markdown("Décrivez les caractéristiques générales du visage")

        # Sexe
        gr.Markdown("### Sexe")
        sexe = gr.Dropdown(['Homme','Femme', 'Non précisé'], label="Sexe")

        # Cheveux
        gr.Markdown("### Cheveux")
        couleur_chev = gr.Dropdown(['Blond','Brun','Noir','Gris','Chauve', 'Non précisé'], label="Couleur des cheveux")
        type_chev = gr.Dropdown(['Raides','Ondulés', 'Non précisé'], label="Type de cheveux")
        
        button1 = gr.Button('Voir les propositions ')


    #----------------------------------------------------------------
    # Etape 2 : Slection d'images initiales
    #---------------------------------------------------------------- 
    with gr.Column(visible=False) as images_select:
        gr.Markdown("## Sélectionnez une ou plusieurs images")
        
        info = gr.Textbox(label="Informations", interactive=False)
        gr.Markdown(""" ## Que voulez-vous faire ?
                    - **Reconstruction** : sélectionnez exactement **1 image**
                    - **Interpolation** : sélectionnez exactement **2 images** """)

        #----------------------------
        # States qui stockent les chemins
        #----------------------------
        path1 = gr.State()
        path2 = gr.State()
        path3 = gr.State()
        path4 = gr.State()
        path5 = gr.State()
        path6 = gr.State()

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
            interpolation_button = gr.Button("Interpolation")

        button_retour_quest = gr.Button("Retour au questionnaire")

    
    #----------------------------------------------------------------
    # Etape 3 : Resultats
    #---------------------------------------------------------------- 
    with gr.Column(visible=False) as result_zone:
        gr.Markdown("## Résultat")

        result_text = gr.Textbox(label="Message",interactive=False)

        result_image = gr.Image(type="pil", label="Image résultat")

        button_retour_selection = gr.Button("Retour à la selection")

    
    
    #--------------------------------
    # Fonctions de traitement Gradio
    #--------------------------------
    def eval_reconstruction(i1, i2, i3, i4, i5, i6, c1, c2, c3, c4, c5, c6):
        images = [i1, i2, i3, i4, i5, i6]
        checks = [c1, c2, c3, c4, c5, c6]
        selection = [img for img, c in zip(images, checks) if c and img is not None]
        
        if len(selection) != 1:
            # Reste sur la sélection, met un message d'erreur
            return None, "Erreur : Veuillez sélectionner exactement 1 image pour la reconstruction.", gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            
        res_img = reconstruct_image(selection[0])
        # Cache la sélection, affiche le résultat
        return res_img, "Reconstruction réussie !", gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)

    def eval_interpolation(i1, i2, i3, i4, i5, i6, c1, c2, c3, c4, c5, c6):
        images = [i1, i2, i3, i4, i5, i6]
        checks = [c1, c2, c3, c4, c5, c6]
        selection = [img for img, c in zip(images, checks) if c and img is not None]
        
        if len(selection) != 2:
            return None, "Erreur : Veuillez sélectionner exactement 2 images pour l'interpolation.", gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            
        res_img = interpolate_images(selection[0], selection[1])
        return res_img, "Interpolation réussie !", gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)



    #----------------------------------------------------------------
    # Etape 4 : Questionnaire 2 
    # (propositions de changements : ajout/suppression caractéristiques)
    #----------------------------------------------------------------
    with gr.Column(visible=False) as questionnaire2:
        gr.Markdown("## Questionnaire 2")
        gr.Markdown("Choisissez les changements que vous souhaitez apporter au portrait généré")

        gr.Markdown("### Cheveux")
        couleur_chev_q2 = gr.Dropdown(['Aucun changement', 'Blond', 'Brun', 'Noir', 'Gris', 'Chauve'], label="Changer la couleur des cheveux")
        type_chev_q2 = gr.Dropdown(['Aucun changement', 'Raides', 'Ondulés'], label="Changer le type de cheveux")

        gr.Markdown("### Pilosité")
        pilosite_q2 = gr.CheckboxGroup(["Barbe", "Moustache", "Bouc", "Frange", "Sideburns", "Calvitie frontale"], label="Modifications de pilosité")

        gr.Markdown("### Forme et traits du visage")
        visage_q2 = gr.CheckboxGroup(["Joues rosées", "Nez pointu", "Peau pâle", "Visage ovale", "Yeux étroits", "Pommettes hautes", "Double menton","Sourcils épais", "Gros nez", "Lèvres pulpeuses", "Cernes"], label="Modifications du visage")

        gr.Markdown("### Accessoires")
        accessoires_q2 = gr.CheckboxGroup(["Lunettes", "Maquillage prononcé", "Boucles d'oreilles", "Chapeau", "Rouge à lèvres", "Collier", "Cravate"], label="Modifications d'accessoires")
    
        appliquer_modifs_button = gr.Button("Appliquer les modifications")



    #--------------------------------
    # Boutons 
    #-------------------------------- 
    
    #Affiche propositions
    button1.click(selection_images, inputs=[sexe, couleur_chev, type_chev], outputs=[img1, img2, img3, img4, img5, img6, path1, path2, path3, path4, path5, path6, bloc1, bloc2, bloc3, bloc4, bloc5, bloc6, questionnaire, images_select, info]) #gradio appelle la fonction selection_images qui aura en entrée l'input
    
    
    button_retour_quest.click(retour_questionnaire, inputs=[], outputs=[questionnaire, images_select])
    button_retour_selection.click(retour_selection, inputs = [], outputs = [images_select, result_zone])
    
    # Boutons d'action
    all_imgs_and_checks = [img1, img2, img3, img4, img5, img6, check1, check2, check3, check4, check5, check6]
    
    reconstruction_button.click(eval_reconstruction, inputs=all_imgs_and_checks, outputs=[result_image, result_text, result_zone, images_select, questionnaire2])
    interpolation_button.click(eval_interpolation, inputs=all_imgs_and_checks, outputs=[result_image, result_text, result_zone, images_select, questionnaire2])
    
    appliquer_modifs_button.click(appliquer_modifications, inputs=[result_image, couleur_chev_q2, type_chev_q2, pilosite_q2, visage_q2, accessoires_q2], outputs=[result_image, result_text])
    

demo.queue().launch()




