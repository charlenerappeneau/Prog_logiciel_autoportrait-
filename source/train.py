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

epochs=30
batch_size=16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

directory=f"vaemodels-{''.join(random.choices(ascii_lowercase, k=5))}"
mkdir(directory)
print(directory)

train_loader, test_loader = import_data(
    path="img_align_celeba",
    batch_size=batch_size,
    img_size=img_size
)

model=VAE().to(device)
optimizer=optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, log_var):
	mse=F.mse_loss(recon_x, x.view(-1, img_dim))
	kld=-0.5*torch.mean(1+log_var-mu.pow(2)-log_var.exp())
	kld_poids=0.0005
	loss=mse+kld_poids*kld
	return loss

def train(epoch):
	model.train()
	train_loss=0
	for batch_idx, (data,_) in enumerate(test_loader):
		torch.cuda.empty_cache()
		data = data.to(device)
		optimizer.zero_grad()
		recon_batch, mu, log_var = model(data)
		log_var = torch.clamp_(log_var, -10, 10)
		loss = loss_function(recon_batch, data, mu, log_var)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader),loss.item() / len(data)))
	print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
	model.eval()
	test_loss=0
	with torch.no_grad():
		for i, (data, _) in enumerate(test_loader):
			data=data.to(device)
			recon_batch, mu, log_var=model(data)
			test_loss+=loss_function(recon_batch, data, mu, log_var).item()
			if i==0:
				n=min(data.size(0),8)
				comparison=torch.cat([data[:n], recon_batch.view(batch_size, 3, img_size, img_size)[:n]])
				save_image(comparison.cpu(), f'{directory}/reconstruction_{str(epoch)}.png', nrow=n)
	test_loss/=len(test_loader.dataset)
	print('{:.4f}'.format(test_loss))

if __name__=="__main__":
	print(f'epochs: {epochs}')
	for epoch in range(1, epochs+1):
		train(epoch)
		torch.save(model, f'{directory}/vae_model_{epoch}.pth')
		test(epoch)
		with torch.no_grad():
			sample=torch.randn(64, latent_dim).to(device)
			sample=model.decode(sample).cpu()
			save_image(sample.view(64, 3, img_size, img_size),f'{directory}/sample_{str(epoch)}.png')

