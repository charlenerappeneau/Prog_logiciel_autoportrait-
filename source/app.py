import gradio as gr
from PIL import Image
import numpy as np


def reconstruct_image(image):
    """
    Placeholder pour reconstruction avec VAE
    """
    if image is None:
        return None

    img = np.array(image)

    # simulation reconstruction (placeholder)
    reconstructed = img

    return Image.fromarray(reconstructed)


def interpolate_images(img1, img2):
    """
    Placeholder interpolation espace latent
    """
    if img1 is None or img2 is None:
        return None

    img1 = np.array(img1)
    img2 = np.array(img2)

    blend = (0.5 * img1 + 0.5 * img2).astype(np.uint8)

    return Image.fromarray(blend)


with gr.Blocks() as demo:

    gr.Markdown("# Portrait Robot Generator")
    gr.Markdown("Projet VAE basé sur CelebA")

    with gr.Tab("Reconstruction"):

        input_img = gr.Image(type="pil", label="Image entrée")
        output_img = gr.Image(label="Image reconstruite")

        reconstruct_btn = gr.Button("Reconstruire")

        reconstruct_btn.click(
            reconstruct_image,
            inputs=input_img,
            outputs=output_img
        )

    with gr.Tab("Interpolation"):

        imgA = gr.Image(type="pil", label="Image A")
        imgB = gr.Image(type="pil", label="Image B")

        interp_img = gr.Image(label="Interpolation")

        interp_btn = gr.Button("Interpoler")

        interp_btn.click(
            interpolate_images,
            inputs=[imgA, imgB],
            outputs=interp_img
        )

demo.launch()
