import torch
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

from linear_cvae import VAE

train_data = torchvision.datasets.MNIST(root='../mnist', train=True, transform=torchvision.transforms.ToTensor(),
                                        download=False)
test_data = torchvision.datasets.MNIST(root='../mnist', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=False)
train_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=2)
test_loader = Data.DataLoader(dataset=test_data, batch_size=128, shuffle=True, num_workers=2)

model = VAE().cuda()


def loss_func(recon_x, x, mu, logvar):
    # MSE = F.mse_loss(recon_x, x)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD / (recon_x.data.shape[0] * 784)
    return KLD + BCE, KLD


def one_hot(data, labels):
    data_shape = data.data.shape
    label = torch.zeros(data_shape[0], 10)
    for i in range(data_shape[0]):
        label[i, labels[i]] = 1
    return label


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
KL = []


def train(epoch):
    model.train()
    train_loss = 0
    for step, (batch_x, batch_labels) in enumerate(train_loader):
        batch_x = Variable(batch_x).cuda()
        batch_labels = Variable(one_hot(batch_x, batch_labels)).cuda()
        recon_x, mu, logvar = model(batch_x, batch_labels)
        loss, KL_loss = loss_func(recon_x, batch_x, mu, logvar)
        KL.append(KL_loss.data[0])
        train_loss += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print("Epoch: {}\tStep: {}\tLoss: {:.6f}".format(epoch, step, loss.data[0] / len(batch_x)))

    avg_loss = train_loss / len(train_loader.dataset)
    print("=============> Epoch: {}\t Average loss: {:.4f}".format(epoch, avg_loss))
    return avg_loss


def test(epoch):
    model.eval()
    test_loss = 0
    for step, (batch_x, batch_labels) in enumerate(test_loader):
        batch_x = Variable(batch_x).cuda()
        batch_labels = Variable(one_hot(batch_x, batch_labels)).cuda()
        recon_x, mu, logvar = model(batch_x, batch_labels)
        loss, _ = loss_func(recon_x, batch_x, mu, logvar)
        test_loss += loss.data[0]

    save_image(torch.cat([batch_x[0:10], recon_x[0:10].view(-1, 1, 28, 28)]).data.cpu(),
               './pretrain/epoch_{}.png'.format(epoch), nrow=10)
    print('=======> Epoch: {}\t Average loss: {:.4f}'.format(epoch, test_loss / len(test_loader.dataset)))

    sample_label = torch.LongTensor(range(10)).repeat(10)
    sample_data = Variable(torch.randn(100,10)).cuda()
    one_hot_label = Variable(one_hot(sample_data, sample_label)).cuda()
    sample = model.decode(sample_data, one_hot_label)
    save_image(sample.data.view(100, 1, 28, 28),
               './pretrain/conditional_vae_sample_' + str(epoch) + '.png', nrow=10)
    return test_loss / len(test_loader.dataset)



train_log = []
test_log = []
for epoch in range(20):
    train_loss = train(epoch)
    test_loss = test(epoch)
    train_log.append(train_loss)
    test_log.append(test_loss)
    scheduler.step(test_loss)

plt.plot(train_log, 'r-')
plt.plot(test_log, 'b-')
plt.show()
plt.figure()
plt.plot(KL, 'k--')
plt.show()

torch.save(model.state_dict(), 'pretrain_vae.pkl')
