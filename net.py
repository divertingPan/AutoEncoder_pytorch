import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(28 * 28, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, 8),
            nn.LeakyReLU(inplace=True),
            nn.Linear(8, 4),
        )

        # decoder
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.Linear_0 = nn.Linear(4, 8)
        self.Linear_1 = nn.Linear(8, 16)
        self.Linear_2 = nn.Linear(16, 32)
        self.Linear_3 = nn.Linear(32, 64)
        self.Linear_4 = nn.Linear(64, 128)
        self.Linear_5 = nn.Linear(128, 28 * 28)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = x.view(-1, 28*28)
        x_encoded = self.encoder(x)
        x_decoded = self.Linear_0(x_encoded)
        x_decoded = self.LeakyReLU(x_decoded)
        x_decoded = self.Linear_1(x_decoded)
        x_decoded = self.LeakyReLU(x_decoded)
        x_decoded = self.Linear_2(x_decoded)
        x_decoded = self.LeakyReLU(x_decoded)
        x_decoded = self.Linear_3(x_decoded)
        x_decoded = self.LeakyReLU(x_decoded)
        x_decoded = self.Linear_4(x_decoded)
        x_decoded = self.LeakyReLU(x_decoded)
        x_decoded = self.Linear_5(x_decoded)
        x_decoded = self.LeakyReLU(x_decoded)
        x_decoded = self.Sigmoid(x_decoded)
        x_decoded = x_decoded.view(-1, 28, 28)
        return x_encoded, x_decoded


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.Linear_0 = nn.Linear(4, 8)
        self.Linear_1 = nn.Linear(8, 16)
        self.Linear_2 = nn.Linear(16, 32)
        self.Linear_3 = nn.Linear(32, 64)
        self.Linear_4 = nn.Linear(64, 128)
        self.Linear_5 = nn.Linear(128, 28 * 28)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_decoded = self.Linear_0(x)
        x_decoded = self.LeakyReLU(x_decoded)
        x_decoded = self.Linear_1(x_decoded)
        x_decoded = self.LeakyReLU(x_decoded)
        x_decoded = self.Linear_2(x_decoded)
        x_decoded = self.LeakyReLU(x_decoded)
        x_decoded = self.Linear_3(x_decoded)
        x_decoded = self.LeakyReLU(x_decoded)
        x_decoded = self.Linear_4(x_decoded)
        x_decoded = self.LeakyReLU(x_decoded)
        x_decoded = self.Linear_5(x_decoded)
        x_decoded = self.LeakyReLU(x_decoded)
        x_decoded = self.Sigmoid(x_decoded)
        x_decoded = x_decoded.view(-1, 28, 28)
        return x_decoded


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(True),
            nn.Conv2d(3, 3, 2, 2, 0),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(True),
            nn.Conv2d(3, 5, 2, 2, 0),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(True),
            nn.Conv2d(5, 3, 3, 3, 1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(True),
            nn.Conv2d(3, 1, 3, 2, 1),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 3, 3, 2, 1),
            nn.BatchNorm2d(3),
            nn.Tanh(),
            nn.ConvTranspose2d(3, 5, 3, 3, 1),
            nn.BatchNorm2d(5),
            nn.Tanh(),
            nn.ConvTranspose2d(5, 3, 2, 2, 0),
            nn.BatchNorm2d(3),
            nn.Tanh(),
            nn.ConvTranspose2d(3, 3, 2, 2, 0),
            nn.BatchNorm2d(3),
            nn.Tanh(),
            nn.ConvTranspose2d(3, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class DoubleConvNet(nn.Module):
    def __init__(self):
        super(DoubleConvNet, self).__init__()
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(1, 5, 8, 4, 2),  # 7x7x5
            nn.BatchNorm2d(5),
            nn.LeakyReLU(True),
            nn.Conv2d(5, 7, 5, 1, 0),  # 3x3x7
            nn.BatchNorm2d(7),
            nn.LeakyReLU(True),
            nn.Conv2d(7, 9, 2, 1, 0),  # 2x2x19
            nn.BatchNorm2d(9),
            nn.LeakyReLU(True),
            nn.Conv2d(9, 10, 2, 1, 0),  # 1x1x11
            nn.Sigmoid(),
        )

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(1, 5, 14, 1, 0),  # 15x15x5
            nn.BatchNorm2d(5),
            nn.LeakyReLU(True),
            nn.Conv2d(5, 7, 8, 1, 0),  # 8x8x7
            nn.BatchNorm2d(7),
            nn.LeakyReLU(True),
            nn.Conv2d(7, 9, 4, 4, 0),  # 2x2x9
            nn.BatchNorm2d(9),
            nn.LeakyReLU(True),
            nn.Conv2d(9, 10, 2, 1, 0),  # 1x1x11
            nn.Sigmoid(),
        )

        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(20, 9, 2, 1, 0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
            nn.ConvTranspose2d(9, 7, 2, 1, 0),
            nn.BatchNorm2d(7),
            nn.ReLU(True),
            nn.ConvTranspose2d(7, 5, 5, 1, 0),
            nn.BatchNorm2d(5),
            nn.ReLU(True),
            nn.ConvTranspose2d(5, 1, 8, 4, 2),
            nn.Sigmoid()
        )

        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(20, 9, 2, 1, 0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
            nn.ConvTranspose2d(9, 7, 4, 4, 0),
            nn.BatchNorm2d(7),
            nn.ReLU(True),
            nn.ConvTranspose2d(7, 5, 8, 1, 0),
            nn.BatchNorm2d(5),
            nn.ReLU(True),
            nn.ConvTranspose2d(5, 1, 14, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded_1 = self.encoder_1(x)
        encoded_2 = self.encoder_2(x)
        encoded = torch.cat((encoded_1, encoded_2), 1)
        decoded_1 = self.decoder_1(encoded)
        decoded_2 = self.decoder_2(encoded)
        decoded = decoded_1 + decoded_2
        return encoded, decoded
