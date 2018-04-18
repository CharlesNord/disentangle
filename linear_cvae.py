import torch
from torch.autograd import Variable


class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 200),
            torch.nn.ReLU()
        )
        self.fc1 = torch.nn.Linear(200, 10)
        self.fc2 = torch.nn.Linear(200, 10)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(20, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 784),
            torch.nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x.view(-1, 784))
        mu = self.fc1(h)
        logvar = self.fc2(h)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + eps.mul(std)

    def decode(self, z, label):
        z = torch.cat([z, label], dim=1)
        return self.decoder(z)

    def forward(self, x, label):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z, label), mu, logvar
