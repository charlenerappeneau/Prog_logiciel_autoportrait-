# Portrait Robot Generator

This project is a web application based on a Variational Autoencoder (VAE) trained on the CelebA dataset. It allows users to generate portrait robots (facial composites) based on specific physical characteristics like gender, hair color, facial features, and accessories.

## Features
- **Semantic Search**: Find images that match specific facial attribute criteria.
- **Latent Space Manipulation**: 
  - **Reconstruction**: Reconstruct a given face using the trained VAE.
  - **Interpolation**: Interpolate between two images to generate intermediate faces.
  - **Fusion / Mutation**: Add or remove specific physical traits of an image without altering identity via vector arithmetic in the latent space.

## Project Structure
- `dataset/`: Contains the CelebA dataset metadata (`dataset/list_attr_celeba.txt`, `dataset/identity_CelebA.txt`) and aligned images (`dataset/img_align_celeba/`). THIS IS NOT PROVIDED IN THE REPO.
You can download the dataset from the internet directly. Just make sure that the Two files and the directory structure above is fulfilled.
- `source/`: Contains the core ML algorithms, PyTorch VAE model definition, and data preprocessing logic. Pre-trained weights should reside in `source/vaemodels-igimu/`.
- `scripts/`: Contains the scripts to launch the app and train the model.

## Installation

1. Clone this repository.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Start the Gradio web interface by executing the launch script from the root directory:

```bash
python -m scripts.launching_app
```
