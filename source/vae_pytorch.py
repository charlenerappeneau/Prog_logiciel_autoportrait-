import torch
from torchvision import transforms
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision.utils import save_image

#paramètres
path = 'img_align_celeba/' #attention le dataset a par la suite été changé de dossier donc le chemin ne marche que sur mon bureau local
img_size=128 #nouvelle taille des images
latent_dim=256 #dimension de l'espace latent
img_dim =3*img_size*img_size #nombre de pixels

#Preprocessing, On transforme les images et on les normalize avec ToTensor
data_ready_to_use_tensor= transforms.Compose([transforms.Resize(img_size, antialias=True), transforms.CenterCrop(img_size), transforms.ToTensor()])

#transformation de la sortie du VAE
data_output=transforms.Compose([transforms.Resize(img_size, antialias=True), transforms.CenterCrop(img_size)])

#Definition du VAE
class VAE(nn.Module):
	"""
	VAE basé sur un encodeur convolutionnel
	et un décodeur symétrique utilisant des Upsample + Conv2D.

	L'encodeur réduit progressivement la résolution tout en augmentant
	la profondeur des features. Le décodeur reconstruit l'image à partir
	d'un vecteur latent de dimension `latent_dim`.
	"""
	def __init__(self):
		super(VAE, self).__init__()
		channels=3
		dims=[32,64,128,256,512] #profondeur >
		self.final_dim = dims[-1]
		modules=[]
		#Encodeur, réduction de res spatiale et augmentation profondeur
		for dim in dims:
			modules.append(nn.Sequential(nn.Conv2d(channels, out_channels=dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(dim), nn.LeakyReLU()))
			channels=dim
		self.encoder=nn.Sequential(*modules)
		#calcul taille spatiale finale
		out=self.encoder(torch.rand(1,3,img_size,img_size))
		self.size=out.shape[2]
		# Projections vers l'espace latent
		self.fc_mu=nn.Linear(dims[-1] * self.size * self.size, latent_dim) #Couche moy pour projeter les données espace latent
		self.fc_var=nn.Linear(dims[-1] * self.size * self.size, latent_dim) #Couche var pour projeter les données espace latent

		#Decodeur, projection du vecteur latent vers dim traitable
		modules=[]
		self.decoder_input=nn.Linear(latent_dim, dims[-1]*self.size*self.size)
		dims.reverse() #inversion car symétrie avec encodeur

		for i in range(len(dims)-1):
			modules.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),nn.Conv2d(dims[i], dims[i+1], kernel_size=3, padding=1),nn.BatchNorm2d(dims[i+1]),nn.LeakyReLU()))
		self.decoder=nn.Sequential(*modules)
		#Couche finale pour retrouver 3canaux RGB
		self.final_layer=nn.Sequential(nn.ConvTranspose2d(dims[-1],dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(dims[-1]), nn.LeakyReLU(), nn.Conv2d(dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Sigmoid())

	def encode(self, x):
		"""
		Encode une image en paramètres (mu, log_var) de la distribution latente.
		"""
		result=self.encoder(x)
		result=torch.flatten(result, start_dim=1)
		mu=self.fc_mu(result)
		log_var=self.fc_var(result)
		return mu, log_var

	def reparameterize(self, mu, log_var):
		"""
		reparamétrisation :
		z = mu + sigma * epsilon
		"""
		std=torch.exp(0.5*log_var)
		eps=torch.randn_like(std)
		return eps*std+mu

	def decode(self, z):
		"""
		Reconstruit une image à partir d'un vecteur latent z.
		"""
		output=self.decoder_input(z)
		output=output.view(-1, self.final_dim, self.size, self.size)
		output=self.decoder(output)
		output=self.final_layer(output)
		output=data_output(output)
		output=torch.flatten(output, start_dim=1)
		output=torch.nan_to_num(output)
		return output

	def forward(self, x):#enchaine les trois fonctions
		"""
		Pipeline complet du VAE :
		encode → reparameterize → decode
		"""
		mu, log_var=self.encode(x)
		z=self.reparameterize(mu, log_var)
		return self.decode(z), mu, log_var



