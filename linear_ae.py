import torch


class AE(torch.nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 10)
        )
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
        h = self.encoder(x)
        return h

    def decode(self, z, label):
        z = torch.cat([z, label], dim=1)
        return self.decoder(z)

    def forward(self, x, label):
        z = self.encode(x)
        return self.decode(z, label), z
