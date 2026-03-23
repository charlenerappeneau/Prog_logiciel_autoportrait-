import gradio as gr
from PIL import Image
from source.latent_space import *
from source.dataset import selection_images

#======================================================================
# FONCTIONS
#======================================================================

def selection_images(sexe, couleur_chev, type_chev, pilosite, accessoires, visage):
    # Ici, tu génères ou récupères les images correspondant aux choix
    images = [Image.new("RGB", (128, 128), color="gray") for _ in range(3)]
    # Retour : images de la galerie, masquer questionnaire, afficher images_select
    return images, gr.update(visible=False), gr.update(visible=True)

def images_to_actions():
    # Masquer sélection d'images et afficher actions
    return gr.update(visible=False), gr.update(visible=True)

def reconstruct_image(input_img):
    # Ici, place ton code de reconstruction VAE
    return input_img  # placeholder

def interpolate_images(imgA, imgB):
    # Ici, place ton code d'interpolation VAE
    return imgA  # placeholder

#======================================================================
# INTERFACE
#======================================================================

with gr.Blocks() as demo:

    gr.Markdown("# Portrait Robot Generator")
    gr.Markdown("Projet VAE basé sur CelebA")

    # ---------------- QUESTIONNAIRE ----------------
    with gr.Column(visible=True) as questionnaire:
        gr.Markdown("## 🕵️ Portrait robot")
        gr.Markdown("Décrivez les caractéristiques du visage")

        # Sexe
        gr.Markdown("### Sexe")
        sexe = gr.Dropdown(
            ['Homme','Femme'],
            label="Sexe"
        )

        # Cheveux
        gr.Markdown("### Cheveux")
        couleur_chev = gr.Dropdown(
            ['Blond','Brun','Noir','Roux','Gris','Chauve'],
            label="Couleur des cheveux"
        )
        type_chev = gr.Dropdown(
            ['Raides','Ondulés','Bouclés','Frisés'],
            label="Type de cheveux"
        )

        # Pilosité
        gr.Markdown("### Pilosité")
        pilosite = gr.CheckboxGroup(
            ["Barbe", "Moustache", "Bouc", "Frange", "Calvitie frontale"],
            label="Choix multiples"
        )

        # Accessoires
        gr.Markdown("### Accessoires")
        accessoires = gr.CheckboxGroup(
            ["Lunettes","Maquillage prononcé","Boucles d’oreilles","Chapeau",
             "Rouge à lèvres","Collier","Cravate"],
            label="Choix multiples"
        )

        # Visage
        gr.Markdown("### Forme du visage et traits")
        visage = gr.CheckboxGroup(
            ["Joues rosées","Nez pointu","Peau pâle","Visage ovale","Yeux étroits",
             "Pommettes hautes","Bouche entrouverte","Double menton","Joues rondes",
             "Sourcils épais","Gros nez","Lèvres pulpeuses","Cernes","Souriant","Attirant"],
            label="Choix multiples"
        )

        button1 = gr.Button('Voir les propositions')

    # ---------------- IMAGES ----------------
    with gr.Column(visible=False) as images_select:
        gr.Markdown("## Sélectionnez une ou plusieurs images")
        gallery = gr.Gallery(label="Selection d'images", show_label=True, elem_id="gallery", interactive=True)
        selectionnes = gr.Image(type='pil', label='Images sélectionnées')
        button2 = gr.Button("Continuer vers les actions")

    # ---------------- ACTIONS ----------------
    with gr.Column(visible=False) as actions:
        gr.Markdown("## Choisissez une action")

        with gr.Tab("Reconstruction"):
            input_img = gr.Image(type="pil", label="Image entrée")
            output_img = gr.Image(label="Image reconstruite")
            reconstruct_btn = gr.Button("Reconstruire")
            reconstruct_btn.click(reconstruct_image, inputs=input_img, outputs=output_img)

        with gr.Tab("Interpolation"):
            imgA = gr.Image(type="pil", label="Image A")
            imgB = gr.Image(type="pil", label="Image B")
            interp_img = gr.Image(label="Interpolation")
            interp_btn = gr.Button("Interpoler")
            interp_btn.click(interpolate_images, inputs=[imgA, imgB], outputs=interp_img)

    # ---------------- BOUTONS ----------------
    button1.click(
        selection_images,
        inputs=[sexe, couleur_chev, type_chev, pilosite, accessoires, visage],
        outputs=[gallery, questionnaire, images_select]
    )

    button2.click(
        images_to_actions,
        inputs=[],
        outputs=[images_select, actions]
    )

demo.launch()
