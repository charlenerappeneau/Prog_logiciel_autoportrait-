import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

img_size=128

class ImageDataset(Dataset) : 
	def __init__(self, file_list, img_size):
		self.file_list=file_list
		self.transform=transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, idx):
		img_path=self.file_list[idx]
		img=Image.open(img_path).convert("RGB")
		img=self.transform(img)
		return img, img_size


def import_data(path, batch_size, img_size):
	files=[os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
	train_files, test_files=train_test_split(files, test_size=0.3, random_state=30)
	train_dataset=ImageDataset(train_files, img_size)
	test_dataset=ImageDataset(test_files, img_size)
	train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
	return train_loader, test_loader

