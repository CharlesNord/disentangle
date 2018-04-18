import torch
from torch.utils import data as Data
import torchvision
from torchvision.transforms import ToTensor
from linear_cvae import VAE
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torchvision.utils import save_image
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
from linear_ae import AE

train_data = torchvision.datasets.MNIST(root='../mnist', train=True, transform=ToTensor(), download=False)
test_data = torchvision.datasets.MNIST(root='../mnist', train=False, transform=ToTensor(), download=False)

train_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=2)
test_loader = Data.DataLoader(dataset=test_data, batch_size=128, shuffle=True, num_workers=2)

feature_model = VAE().cuda()
#feature_model = AE().cuda()
feature_dict = torch.load('pretrain_vae.pkl')
#feature_dict = torch.load('pretrain_ae.pkl')
feature_model.load_state_dict(feature_dict)
feature_model.eval()

print 'FUCKFUCKFUCK'


class Disentangle(torch.nn.Module):
    def __init__(self):
        super(Disentangle, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 500),
            torch.nn.PReLU(),
            torch.nn.Linear(500, 500),
            torch.nn.PReLU(),
            torch.nn.Linear(500, 200),
            torch.nn.PReLU(),
            torch.nn.Linear(200, 10),
            torch.nn.PReLU(),
            torch.nn.Linear(10, 2)
        )
        self.decoder = torch.nn.Sequential(
            # torch.nn.Linear(12, 10),
            # torch.nn.PReLU(),
            # torch.nn.Linear(10, 200),
            torch.nn.Linear(12, 200),
            torch.nn.PReLU(),
            torch.nn.Linear(200, 500),
            torch.nn.PReLU(),
            torch.nn.Linear(500, 784),
            torch.nn.Sigmoid()
        )

    def encode(self, x):
        z = self.encoder(x.view(-1, 784))
        return z

    def decode(self, z, f):
        z_f = torch.cat([z, f], dim=1)
        recon_x = self.decoder(z_f)
        return recon_x

    def forward(self, x, f):
        z = self.encode(x)
        recon_x = self.decode(z, f)
        return recon_x, z


def scatter(feat, label, epoch):
    plt.ion()
    plt.clf()
    if feat.shape[1] > 2:
        if feat.shape[0] > 5000:
            feat = feat[:5000, :]
            label = label[:5000]
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        feat = tsne.fit_transform(feat)

    palette = np.array(sns.color_palette('hls', 10))
    ax = plt.subplot(aspect='equal')
    for i in range(10):
        plt.plot(feat[label == i, 0], feat[label == i, 1], '.', c=palette[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    ax.axis('tight')
    for i in range(10):
        xtext, ytext = np.median(feat[label == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=18)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])

    plt.draw()
    plt.savefig('./disentangled/scatter_{}.png'.format(epoch))
    plt.pause(0.001)


disentangle = Disentangle().cuda()
loss_func = torch.nn.BCELoss()
optimizer = torch.optim.Adam(disentangle.parameters(), lr=1e-3)


def train(epoch):
    disentangle.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data.view(-1, 784)).cuda()
        feat, _ = feature_model.encode(data)
        # feat = feature_model.encode(data)
        recon_x, _ = disentangle(data, feat)
        loss = loss_func(recon_x, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        if batch_idx % 50 == 0:
            print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch, loss.data[0] / len(data)))

    avg_loss = train_loss / len(train_loader.dataset)
    print('======> Epoch: {} \t Average loss: {:.4f}'.format(epoch, avg_loss))
    return avg_loss



def test(epoch):
    disentangle.eval()
    test_loss = 0
    feat_total = []
    target_total = []
    for i, (data, target) in enumerate(test_loader):
        data = Variable(data.view(-1, 784), volatile=True).cuda()
        feat, _ = feature_model.encode(data)
        # feat = feature_model.encode(data)
        recon_batch, z = disentangle(data, feat)
        test_loss += loss_func(recon_batch, data).data[0]
        feat_total.append(z.data.cpu())
        target_total.append(target)
        if i == 0:
            n = min(data.size(0), 10)
            comparison = torch.cat([data.view(-1, 1, 28, 28)[:n], recon_batch.view(-1, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(), './disentangled/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('=======> Test set loss: {:.4f}'.format(test_loss))
    feat_total = torch.cat(feat_total, dim=0)
    target_total = torch.cat(target_total, dim=0)
    scatter(feat_total.numpy(), target_total.numpy(), epoch)
    return test_loss


test_loss_log = []
train_loss_log = []

for epoch in range(1, 20):
    train_loss = train(epoch)
    test_loss = test(epoch)
    train_loss_log.append(train_loss)
    test_loss_log.append(test_loss)

plt.plot(train_loss_log, 'r--')
plt.plot(test_loss_log, 'g-')
plt.show()
