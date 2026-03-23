import gradio as gr #bibliotheque pour interface web en python
from latent_space import *
from dataset import selection_images 




#======================================================================
# FONCTIONS UTILES :
#====================================================================== 

def images_to_actions():
    '''
    Sert à cacher le questionnaire et rendre visible les choix d'imgaes
    '''
    return(gr.update(visible=False),gr.update(visible=True))




#======================================================================
# INTERFACES :
#====================================================================== 


with gr.Blocks() as demo: #pour organiser l'interface en Blocks
    gr.Markdown("# Portrait Robot Generator") #titre en markdown
    gr.Markdown("Projet VAE basé sur CelebA")

    #----------------------------------------------------------------------------------
    # QUESTIONNAIRE :
    #---------------------------------------------------------------------------------- 
    with gr.Column(visible=True) as questionnaire : 
        gr.Markdown("## Afin de permettre une meilleur identification du suspect, veuillez répondre au questionnaire ci-dessous")
        sexe = gr.Radio(["Homme","Femme"], label = 'Sexe')
        couleur_chev = gr.Dropdown(['Blond','Brun', 'Noir', 'Roux'], label = 'Couleur Cheveux')
        type_chev = gr.Dropdown(['Lisse', 'Bouclé', 'Ondulé', 'Frisé'], label = 'Type Cheveux')
        yeux = gr.Dropdown(['Marron','Bleus', 'Verts'], label = 'Couleur yeux')
        lunettes = gr.Checkbox(label = 'Lunettes')
        barbe = gr.Checkbox(label = 'Barbe')
        #autres ? 

        button1 = gr.Button('Voir les propositions')

        


    #----------------------------------------------------------------------------------
    # PROPOSITION D'IMAGES :
    #---------------------------------------------------------------------------------- 
    with gr.Column(visible=False) as images_select : 
        gr.Markdown("## Veuillez sélectionner une ou plusieurs images parmi celles proposées ci-dessous")
        gallery = gr.Gallery(label = "Selection d'images")
        #code
        selectionnes = gr.Image(type='pil',label='Images sélectionnées')
        button2 = gr.Button("Continuer vers les actions") #ce bouton mène aux différentes actions possibles de l'algorithme génétique



    #----------------------------------------------------------------------------------
    # ONGLETS DES ACTIONS :
    #---------------------------------------------------------------------------------- 
    with gr.Column(visible = False) as actions: 
        
        gr.Markdown("##Choisissez une action")
        
        # Creation d'un onglet 'Reconstruction'
        with gr.Tab("Reconstruction"): 
            input_img = gr.Image(type="pil", label="Image entrée") #gradio donnera l'image à Python sous frome d'objet PIL Image
            output_img = gr.Image(label="Image reconstruite") #zone ou l'image résultat sera affichée 

            reconstruct_btn = gr.Button("Reconstruire") #creation d'un boutton avec le texte 'Reconstruire'
            reconstruct_btn.click( reconstruct_image, inputs=input_img, outputs=output_img) #liaison du bouton à la fonction 'reconstruct_img' qui prendra en entree l'inputs et donnera en retour outputs

        
        
        # Creation d'un onglet interpolation :
        with gr.Tab("Interpolation"): 
            imgA = gr.Image(type="pil", label="Image A")
            imgB = gr.Image(type="pil", label="Image B")

            interp_img = gr.Image(label="Interpolation")
            interp_btn = gr.Button("Interpoler")
            interp_btn.click(interpolate_images, inputs=[imgA, imgB], outputs=interp_img) #liaison du bouton à la fonction 'interpolate_images' qui prendra en entree l'inputs et donnera en retour outputs




    # ACTIVATION BOUTONS : 

    button1.click(selection_images, inputs = [sexe, couleur_chev, type_chev, yeux, lunettes, barbe], outputs = [gallery, questionnaire, images_select])
    button2.click(images_to_actions, inputs= [], outputs = [images_select, actions])


demo.launch()
