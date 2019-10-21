import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

class ConvAE(nn.Module):
    def __init__(self, image_channels=1, input_size=28, nz=40):
        super(ConvAE, self).__init__()

        if torch.cuda.is_available():
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

        self.have_cuda = True
        self.nz = nz
        self.input_size = input_size
        self.image_channels = image_channels
        self.conv = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.encode_fc = nn.Sequential(
            nn.Linear(128 * (input_size // 4) * (input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )

        self.q_z_mean = nn.Linear(1024, self.nz)
        self.q_z_var = nn.Linear(1024, self.nz)

        self.decode_fc = nn.Sequential(
            nn.Linear(nz, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 128 * (input_size // 4) * (input_size // 4)),
            nn.BatchNorm1d(128 * (input_size // 4) * (input_size // 4)),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, self.image_channels, 4, 2, 1),
            nn.Sigmoid()
        )
        initialize_weights(self)
    
    def encode(self, x):
        conv = self.conv(x)
        h = self.encode_fc(conv.view(-1, 128*(self.input_size//4) * (self.input_size // 4)))
        #h = self.encode_fc(x.view(-1, 64*(self.input_size//4) * (self.input_size // 4)))
        return self.q_z_mean(h), self.q_z_var(h)
    
    def decode(self, z):
        deconv_input= self.decode_fc(z)
        deconv_input = deconv_input.view(-1, 128, self.input_size//4, self.input_size//4)
        return self.deconv(deconv_input)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.rand_like(std)
        z = eps.mul(std).add_(mu)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar