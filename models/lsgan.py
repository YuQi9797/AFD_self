import torch
import torch.nn as nn


class Discriminiator(nn.Module):
    def __init__(self, in_filters=128, model_type='wrn'):
        super(Discriminiator, self).__init__()
        if model_type == 'wrn':
            self.out_filters = 64  # WRN[B, 128, 8 ,8]
        elif model_type == 'res':  # # 由于ResNet的layer4=>[Batch_size, 2048, 4, 4]
            self.out_filters = 256

        self.model = nn.Sequential(
            nn.Conv2d(in_filters, self.out_filters, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(self.out_filters),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.out_filters, 1, kernel_size=1, padding=0, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


def test():
    discriminator = Discriminiator(in_filters=128)
    y = discriminator(torch.randn(1, 128, 8, 8))
    # y = discriminator(torch.randn(1, 3, 32, 32))
    print(y.size())  # 1, 1, 7, 7

# test()
