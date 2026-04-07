import gradio as gr
from PIL import Image
from source.app import selection_images, retour_selection, retour_questionnaire
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
        gr.Markdown("## Portrait robot")
        gr.Markdown("Décrivez les caractéristiques du visage")

        # Sexe
        gr.Markdown("### Sexe")
        sexe = gr.Dropdown(['Homme','Femme', 'Non précisé'], label="Sexe")

        # Cheveux
        gr.Markdown("### Cheveux")
        couleur_chev = gr.Dropdown(['Blond','Brun','Noir','Gris','Chauve', 'Non précisé'], label="Couleur des cheveux")
        type_chev = gr.Dropdown(['Raides','Ondulés', 'Non précisé'], label="Type de cheveux")

        # Pilosité
        gr.Markdown("### Pilosité")
        pilosite = gr.CheckboxGroup(["Barbe", "Moustache", "Bouc", "Frange", "Sideburns", "Calvitie frontale"], label="Choix multiples")

        # Visage
        gr.Markdown("### Forme et traits du visage")
        visage = gr.CheckboxGroup(["Joues rosées","Nez pointu","Peau pâle","Visage ovale","Yeux étroits","Pommettes hautes","Double menton","Sourcils épais","Gros nez","Lèvres pulpeuses","Cernes"], label="Choix multiples")
        #j ai enlevé : "Bouche entrouverte", souriant

        # Accessoires
        gr.Markdown("### Accessoires")
        accessoires = gr.CheckboxGroup(["Lunettes","Maquillage prononcé","Boucles d'oreilles","Chapeau","Rouge à lèvres","Collier","Cravate"], label="Choix multiples")
        
        button1 = gr.Button('Voir les propositions ')


    #----------------------------------------------------------------
    # Etape 2 : Slection d'images initiales
    #---------------------------------------------------------------- 
    with gr.Column(visible=False) as images_select:
        gr.Markdown("## Sélectionnez une ou plusieurs images")
        
        info = gr.Textbox(label="Informations", interactive=False)

        gr.Markdown(""" ## Que voulez-vous faire ?
                    - **Reconstruction** : sélectionnez exactement **1 image**
                    - **Interpolation** : sélectionnez exactement **2 images**
                    - **Fusion** : sélectionnez **au moins 2 images** """)



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
            fusion_button = gr.Button("Fusion")

            
        button_retour_quest = gr.Button("Retour au questionnaire")

    
    #----------------------------------------------------------------
    # Etape 3 : Resultats
    #---------------------------------------------------------------- 
    with gr.Column(visible=False) as result_zone:
        gr.Markdown("## Résultat")

        result_text = gr.Textbox(label="Message",interactive=False)

        result_image = gr.Image(label="Image résultat")

        button_retour_selection = gr.Button("Retour à la selection")






    #--------------------------------
    # Boutons 
    #-------------------------------- 
    
    #Affiche propositions
    button1.click(selection_images, inputs=[sexe, couleur_chev, type_chev], outputs=[img1, img2, img3, img4, img5, img6, path1, path2, path3, path4, path5, path6, bloc1, bloc2, bloc3, bloc4, bloc5, bloc6, questionnaire, images_select, info]) #gradio appelle la fonction selection_images qui aura en entrée l'input
    

    
    button_retour_quest.click(retour_questionnaire, inputs=[], outputs=[questionnaire, images_select])
    button_retour_selection.click(retour_selection, inputs = [], outputs = [images_select,result_zone ])
    
demo.launch()




