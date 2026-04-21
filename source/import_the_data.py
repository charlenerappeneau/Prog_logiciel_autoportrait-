import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

img_size=128

class ImageDataset(Dataset) : 
	"""
	Dataset personnalisé pour charger des images depuis une liste de fichiers.

	Paramètres
	----------
	file_list : list
		Liste des chemins vers les fichiers images.
	img_size : int
		Taille à laquelle redimensionner les images (img_size x img_size).

	Renvoie
	-------
	(tensor, int)
		- L'image transformée en tenseur PyTorch.
		- La taille d'image (img_size), ici utilisé comme label placeholder.
    """
	def __init__(self, file_list, img_size):
		self.file_list=file_list
		# Transformations appliquées à chaque image
		self.transform=transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])

	def __len__(self):
		"""Retourne le nombre total d'images dans le dataset."""
		return len(self.file_list)

	def __getitem__(self, idx):
		"""
		Charge et transforme une image à l'index donné.

		Paramètres
		----------
		idx : int
		Index de l'image à charger.
		Renvoie
		-------
		img : torch.Tensor
			Image transformée.
		img_size : int
			Taille de l'image (placeholder pour un label).
		"""
		img_path=self.file_list[idx]
		# Ouverture de l'image et conversion en RGB
		img=Image.open(img_path).convert("RGB")
		# Application des transformations
		img=self.transform(img)
		return img, img_size


def import_data(path, batch_size, img_size):
	"""
	Importe les données d'un dossier, crée les datasets et dataloaders.

	Paramètres
  	----------
	path : str
		Chemin du dossier contenant les images .jpg.
			batch_size : int
				Taille des batchs pour les DataLoaders.
	img_size : int
		Taille de redimensionnement des images.

	Renvoie
	-------
	train_loader : DataLoader
		DataLoader pour l'ensemble d'entraînement.
	test_loader : DataLoader
		DataLoader pour l'ensemble de test.
	"""
	# Liste tous les fichiers .jpg dans le dossier
	files=[os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
	# Séparation train/test
	train_files, test_files=train_test_split(files, test_size=0.3, random_state=30)
	# Création des datasets
	train_dataset=ImageDataset(train_files, img_size)
	test_dataset=ImageDataset(test_files, img_size)
	# Création des DataLoaders
	train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
	return train_loader, test_loader

