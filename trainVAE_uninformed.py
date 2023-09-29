import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import sys
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import tensor as tt
import torch.nn.functional as F
import torchvision.transforms as tforms
from torchvision.transforms import v2
import math
import time

ts = time.time()
os.chdir("/n/home02/agagliano/galaxyAutoencoder")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ZouGam = np.load("FullSample_26k_photoz_wOrientation.npz")
images_rescaled = ZouGam['x'].astype(np.float32)
labels = ZouGam['y'].astype(np.float32)

d = int(sys.argv[1])

nepochs = 1000
learning_rate = 5.e-6
beta0 = [10.]*1000
beta1 = [1.]*1000

f = open("TrainingOutput_ZouGama_26k_specz_uninformed_d%i_Final.txt"%d, 'w')
print("learning rate: %.1e"%learning_rate, file=f)
print("number of epochs: %i"%nepochs, file=f)
print("using device: %s"% device, file=f)
transform = tforms.ToTensor()

class GalaxyDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


# random split
N = len(images_rescaled)

np.random.seed(12345)
indices = np.random.permutation(N)
train_idxs = indices[:int(0.8*N)]
valid_idxs = indices[int(0.8*N):]

# transforms and train/valid split
#now, labels should be the sets [redshift, log(stellar mass), log(SFR)]
#transform from 128x128 to 69x69 to work within the pre-existing framework and run quickly
images_train = [F.interpolate(transform(x).unsqueeze(0), size=(69, 69), mode='bicubic').squeeze() for x in images_rescaled[train_idxs]]
images_valid = [F.interpolate(transform(x).unsqueeze(0), size=(69, 69), mode='bicubic').squeeze() for x in images_rescaled[valid_idxs]]

train_ds = GalaxyDataset(images_train, labels[train_idxs], None)
valid_ds = GalaxyDataset(images_valid, labels[valid_idxs], None)

train_loader = DataLoader(
    train_ds,
    batch_size=128,
    num_workers=0,
    shuffle=True,
    pin_memory=torch.cuda.is_available()
)

valid_loader = DataLoader(
    valid_ds,
    batch_size=128,
    num_workers=0,
    shuffle=True,
    pin_memory=torch.cuda.is_available()
)

class ConvVAE(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.d = d

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, padding=1),    # 35x35
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, padding=1),   # 18x18
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*18*18, 2*d)
        )

        self.dec1 = nn.Sequential(
            nn.Linear(d, 32*18*18),
            nn.Unflatten(1, (32, 18, 18)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 1, padding=1, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, 3, 1, padding=1, output_padding=0),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn(mu.shape).to(device)*torch.exp(0.5*logvar)
        else:
            return mu

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        return x.view(-1, 2, self.d)


    def decode(self, z):
        x_hat = self.dec1(z)
        x_hat = F.interpolate(x_hat, size=(35, 35), mode='bilinear')
        x_hat = self.dec2(x_hat)
        x_hat = F.interpolate(x_hat, size=(69, 69), mode='bilinear')
        x_hat = self.dec3(x_hat)
        return x_hat

    def forward(self, x):
        mu_logvar = self.encode(x)

        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]

        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)

        return x_hat, mu, logvar


def display_images(in_, out, nrows=1, ncols=4, label=None, count=False):
    out_pic = np.array(out.data.cpu() * 255, dtype=int).transpose((0, 2, 3, 1))
    plotCounter = 1
    for row in range(nrows):
        if in_ is not None:
            in_pic = np.array(in_.data.cpu() * 255, dtype=int).transpose((0, 2, 3, 1))
            plt.figure(figsize=(18, 4))
            plt.suptitle(label, color='w', fontsize=16)
            for col in range(ncols):
                plt.subplot(1, ncols, plotCounter)
                plt.imshow(in_pic[col + ncols * row])
                plt.axis('off')
                plt.title("Real Image")

                plt.subplot(2, ncols, plotCounter+1)
                plt.imshow(out_pic[col + ncols * row])
                plt.axis('off')
                plt.title("Reconstruction")
                if count: plt.title(str(col + ncols * row), color='w')
                plotCounter += 1

#set the number of hyperparameters
model = ConvVAE(d=d).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)

# Adding additional terms? Try to re-derive the KL divergence from
# https://ai.stackexchange.com/questions/26366/how-is-this-pytorch-expression-equivalent-to-the-kl-divergence
def ELBO(x_hat, x, mu, logvar, y, beta0, beta1, beta2):
    MSE = torch.nn.MSELoss(reduction='sum')(x_hat, x)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return beta0*MSE + beta1*KLD

validation_losses = []
training_losses = []

best_vloss = 100000.

for epoch in range(0, nepochs):
    # train
    print("Beginning training...")
    model.train()
    train_loss = 0
    print("beta0: %.1f"%beta0[epoch], file=f)
    print("beta1: %.1f"%beta1[epoch], file=f)
    print("beta2: %.1f"%beta2[epoch], file=f)
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        x_hat, mu, logvar = model(x)
        loss = ELBO(x_hat, x, mu, logvar, y, beta0[epoch], beta1[epoch], beta2[epoch])
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # validate
    with torch.no_grad():
        model.eval()
        valid_loss = 0
        for x, y in valid_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            valid_loss += ELBO(x_hat, x, mu, logvar, y, beta0[epoch], beta1[epoch], beta2[epoch])

    validation_losses.append(valid_loss / len(valid_loader.dataset))
    training_losses.append(train_loss / len(train_loader.dataset))

    print(f' Epoch {epoch} | Train: {train_loss / len(train_loader.dataset):.5f} | Valid: {validation_losses[-1]:.5f}', file=f)

    # display
    if epoch % 50 == 0:
        display_images(x, x_hat, 1, label=f'Epoch {epoch}')
        plt.savefig("plots/Reconstructions_GAMAZou26k_d%i_uninformed_epoch%05d_Final.png"%(d, epoch), dpi=300, bbox_inches='tight')

        #see how well it's learning rotation:
        fig, ax =plt.subplots(nrows=5, ncols=10, figsize=(50, 20))
        axs = ax.ravel()

        mu = mu.cpu()
        for j in np.arange(d):
            for i in np.arange(10):
                if j == 1:
                    paramMin = 0
                else:
                    paramMin = np.nanmin(mu[:, j])
                paramMax = np.nanmax(mu[:, j])
                ParamRange = np.linspace(paramMin, paramMax, 10)

                tempSet = np.nanmedian(mu, axis=0)
                tempSet[j] = ParamRange[i]
                x_reconstructed = model.decode(torch.FloatTensor(tempSet).to(device).unsqueeze(0))
                x_img = np.array(x_reconstructed.data.cpu() * 255, dtype=int).transpose((0, 2, 3, 1)).squeeze()
                ax[j, i].imshow(x_img)
                ax[j, i].axis('off')
                ax[j, i].set_title('Latent %i = %.2f'%(j, ParamRange[i]))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        plt.savefig("plots/LatentGrid_GAMAZou26k_d%i_uninformed_epoch%05d_Final.png"%(d, epoch),dpi=300, bbox_inches='tight')

    # Track best performance, and save the model's state
    avg_vloss = valid_loss / len(valid_loader.dataset)
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        torch.save(model.state_dict(), './cVAE_ZouGama26k_d%i_uninformed_Final.h5' % d)
        print("Saving model at epoch %i." %epoch, file=f)

#check that the loss converged and that the entries for mu are near what we expect them to be near...
mu_list = []
y_list = []
with torch.no_grad():
    model.eval()

    for x, y in valid_loader:
        x = x.to(device)
        x_hat, mu, logvar = model(x)

        mu_list.append(mu)
        y_list.append(y)

mu_list = torch.cat(mu_list).cpu()
y_list = torch.cat(y_list).cpu()


# using the variable axs for multiple Axes
sns.set_context("talk")

try:
    validation_losses = [x.cpu() for x in validation_losses]
    training_losses = [x.cpu() for x in training_losses]
except:
    print("Losses already on cpu!")

#evaluate the losses
plt.figure(figsize=(10,7))
plt.plot(validation_losses, label='Validation Loss')
plt.plot(training_losses, label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Batch-Wide Loss")
plt.legend()
plt.savefig("plots/Losses_ZouGama26k_d%i_uninformed_Final.png"%d,dpi=200, bbox_inches='tight')
f.close()
