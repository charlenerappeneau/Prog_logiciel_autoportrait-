import gradio as gr #bibliotheque pour interface web en python
from latent_space import *

#Creation interface Gradio :
with gr.Blocks() as demo: #pour organiser l'interface en Blocks

    gr.Markdown("# Portrait Robot Generator") #titre en markdown
    gr.Markdown("Projet VAE basé sur CelebA")

    with gr.Tab("Reconstruction"): #creation d'un onglet 'Reconstruction'

        input_img = gr.Image(type="pil", label="Image entrée") #gradio donnera l'image à Python sous frome d'objet PIL Image
        output_img = gr.Image(label="Image reconstruite") #zone ou l'image résultat sera affichée 

        reconstruct_btn = gr.Button("Reconstruire") #creation d'un boutton avec le texte 'Reconstruire'

        reconstruct_btn.click( reconstruct_image, inputs=input_img, outputs=output_img) #liaison du bouton à la fonction 'reconstruct_img' qui prendra en entree l'inputs et donnera en retour outputs

    with gr.Tab("Interpolation"): #creation d'un onglet interpolation

        imgA = gr.Image(type="pil", label="Image A")
        imgB = gr.Image(type="pil", label="Image B")

        interp_img = gr.Image(label="Interpolation")

        interp_btn = gr.Button("Interpoler")

        interp_btn.click(interpolate_images, inputs=[imgA, imgB], outputs=interp_img) #liaison du bouton à la fonction 'interpolate_images' qui prendra en entree l'inputs et donnera en retour outputs

demo.launch()
