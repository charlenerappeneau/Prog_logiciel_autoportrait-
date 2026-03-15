import numpy as np #bibliotheque pour manipuler tableaux numeriques
from PIL import Image #bibliotheque pour manipulation d'images
import numpy as np #bibliotheque pour manipuler tableaux numeriques

def reconstruct_image(image):
    """
    Cette fonction prend une image en entrée
    Placeholder pour reconstruction avec VAE
    """
    if image is None:
        return None

    img = np.array(image) #conversion image en tableau numpy

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
