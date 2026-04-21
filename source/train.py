import torch
import torch.utils.data
from os import mkdir
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from vae_pytorch import VAE, img_size, path, latent_dim, img_dim, data_ready_to_use_tensor
from import_the_data import import_data
import torch.optim as optim
import random
from string import ascii_lowercase

#parametres généraux 
epochs=30 #nombre d'époques d'entrainement
batch_size=16 #taille des batchs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#Pour utiliser le GPU

#permet de sauvegarder les résultats 
directory=f"vaemodels-{''.join(random.choices(ascii_lowercase, k=5))}" # Génère un nom aléatoire pour éviter d'écraser d'anciens modèles
mkdir(directory)
print(directory)

#on charge les données
train_loader, test_loader = import_data(
    path="img_align_celeba",
    batch_size=batch_size,
    img_size=img_size
)

#initialisation du modele et optimizer
model=VAE().to(device)
optimizer=optim.Adam(model.parameters(), lr=1e-3)

#fonction de perte du VAE
def loss_function(recon_x, x, mu, log_var):
	"""
	Calcule la loss totale du VAE : reconstruction + divergence KL.

	Paramètres
	----------
	recon_x : torch.Tensor
		Image reconstruite par le VAE.
	x : torch.Tensor
		Image originale.
	mu : torch.Tensor
		log_var : torch.Tensor
		Log-variance de la distribution latente.

	Retour
	------
	torch.Tensor
		Valeur de la loss totale.
	"""
	# Utilise L1_loss pour plus de netteté que MSE (utilisé auparavant)
	recon_loss = F.l1_loss(recon_x, x.view(-1, img_dim), reduction='mean')  
	# KLD avec un beta plus équilibré
	kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
	return recon_loss + 0.0002 * kld


#Fonction d'entrainement
def train(epoch):
	"""
	Entraîne le modèle sur une epoch complète.

	Paramètres
	----------
	epoch : int
		Numéro de l'epoch en cours.
	"""
	model.train() #entrainement
	train_loss=0
	for batch_idx, (data,_) in enumerate(train_loader):
		torch.cuda.empty_cache()#enleve le cache
		data = data.to(device)
		optimizer.zero_grad()#reinitialisation des gradients
		recon_batch, mu, log_var = model(data)
		log_var = torch.clamp_(log_var, -10, 10) # evite les valeurs extrêmes
		loss = loss_function(recon_batch, data, mu, log_var) #calcul du loss
		loss.backward()#calcul des gradients
		train_loss += loss.item()
		optimizer.step()#mise à jour des poids
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader),loss.item() / len(data)))
	print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


#fonction de test
def test(epoch):
	"""
	Évalue le modèle sur l'ensemble de test et sauvegarde des reconstructions.

	Paramètres
	----------
	epoch : int
		Numéro de l'époque en cours.
	"""
	model.eval() #mode test
	test_loss=0
	with torch.no_grad(): #pas de calcul de grad
		for i, (data, _) in enumerate(test_loader):
			data=data.to(device)
			recon_batch, mu, log_var=model(data)
			test_loss+=loss_function(recon_batch, data, mu, log_var).item() #acumulation du loss
			# Sauvegarde des reconstructions du premier batch
			if i==0: #sauvegarde original vs reconstruction du premier batch
				n=min(data.size(0),8)
				comparison=torch.cat([data[:n], recon_batch.view(batch_size, 3, img_size, img_size)[:n]])
				save_image(comparison.cpu(), f'{directory}/reconstruction_{str(epoch)}.png', nrow=n)
	test_loss/=len(test_loader.dataset)
	print('{:.4f}'.format(test_loss))


if __name__=="__main__":
	print(f'epochs: {epochs}')
	for epoch in range(1, epochs+1):
		train(epoch) #entrainement
		torch.save(model, f'{directory}/vae_model_{epoch}.pth')#sauvegarde les poids et l'architecture
		test(epoch) #test
		# Génération d'images aléatoires
		with torch.no_grad():#génération d'image à partir de vecteur aléatoire
			sample=torch.randn(64, latent_dim).to(device)
			sample=model.decode(sample).cpu()
			save_image(sample.view(64, 3, img_size, img_size),f'{directory}/sample_{str(epoch)}.png')

