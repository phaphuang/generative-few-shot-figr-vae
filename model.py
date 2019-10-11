import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, image_channels=1, input_size=28, nz=100):
        super(ConvAE, self).__init__()

        self.have_cuda = True
        self.nz = nz
        self.input_size = input_size
        self.conv = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.encode_fc = nn.Sequential(
            nn.Linear(128 * (input_size // 4) * (input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, nz)
        )
        self.decode_fc = nn.Sequential(
            nn.Linear(nz, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 128 * (input_size // 4) * (input_size // 4)),
            nn.BatchNorm1d(128 * (input_size // 4) * (input_size // 4)),
            nn.ReLU(True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, image_channels, 4, 2, 1),
            nn.Sigmoid()
        )
        initialize_weights(self)
    
    def encode(self, x):
        conv = self.conv(x)
        h1 = self.encode_fc(conv.view(-1, 128*(self.input_size//4) * (self.input_size // 4)))
        return h1
    
    def decode(self, z):
        deconv_input= self.decode_fc(z)
        deconv_input = deconv_input.view(-1, 128, self.input_size//4, self.input_size//4)
        return self.deconv(deconv_input)

    def forward(self, x):
        z = self.encode(x)
        decoded = self.decode(z)
        return decoded, z


class ResNetGenerator(nn.Module):
    def __init__(self, input_size,  image_channels=1, height=32, length=32, hidden_size=64, blocks=4):
        super(ResNetGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.blocks = blocks
        self.height = height
        self.length = length
        self.mult = 2**blocks

        self.initial_linear = nn.Linear(input_size, hidden_size * self.mult * height//self.mult * length//self.mult)
        self.initial_norm = nn.LayerNorm(hidden_size * self.mult * height//self.mult * length//self.mult)
        self.initial_activ = nn.PReLU(hidden_size * self.mult * height//self.mult * length//self.mult)

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** (blocks - i), hidden_size * 2 ** (blocks - i), (3, 3), padding=(1, 1)) for i
             in range(blocks)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(
            [hidden_size * 2 ** (blocks - i), height // (2 ** (blocks - i)), length // (2 ** (blocks - i))]) for i in
                                    range(blocks)])
        self.activ1 = nn.ModuleList([nn.PReLU(hidden_size * 2 ** (blocks - i)) for i in range(blocks)])

        self.convs2 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** (blocks - i), hidden_size * 2 ** (blocks - i), (3, 3), padding=(1, 1)) for i
             in range(blocks)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(
            [hidden_size * 2 ** (blocks - i), height // (2 ** (blocks - i)), length // (2 ** (blocks - i))]) for i in
                                    range(blocks)])
        self.activ2 = nn.ModuleList([nn.PReLU(hidden_size * 2 ** (blocks - i)) for i in range(blocks)])

        self.convs3 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** (blocks - i), hidden_size * 2 ** (blocks - i), (3, 3), padding=(1, 1)) for i
             in range(blocks)])
        self.norm3 = nn.ModuleList([nn.LayerNorm(
            [hidden_size * 2 ** (blocks - i), height // (2 ** (blocks - i)), length // (2 ** (blocks - i))]) for i in
                                    range(blocks)])
        self.activ3 = nn.ModuleList([nn.PReLU(hidden_size * 2 ** (blocks - i)) for i in range(blocks)])

        self.convs4 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** (blocks - i), hidden_size * 2 ** (blocks - i), (3, 3), padding=(1, 1)) for i
             in range(blocks)])
        self.norm4 = nn.ModuleList([nn.LayerNorm(
            [hidden_size * 2 ** (blocks - i), height // (2 ** (blocks - i)), length // (2 ** (blocks - i))]) for i in
                                    range(blocks)])
        self.activ4 = nn.ModuleList([nn.PReLU(hidden_size * 2 ** (blocks - i)) for i in range(blocks)])

        self.transitions_conv = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** (blocks - i), hidden_size * 2 ** (blocks - i - 1), (3, 3), padding=(1, 1)) for
             i in range(blocks)])
        self.transitions_norm = nn.ModuleList([nn.LayerNorm(
            [hidden_size * 2 ** (blocks - i - 1), height // (2 ** (blocks - i)), length // (2 ** (blocks - i))]) for i in
                                    range(blocks)])
        self.transitions_activ = nn.ModuleList([nn.PReLU(hidden_size * 2 ** (blocks - i - 1)) for i in range(blocks)])

        self.final_conv = nn.Conv2d(hidden_size, image_channels, (5, 5), padding=(2, 2))
        self.final_activ = nn.Tanh()

    def forward(self, inputs):
        x = self.initial_linear(inputs)
        x = self.initial_activ(x)
        x = self.initial_norm(x)

        x = x.view(x.shape[0], self.hidden_size * self.mult, self.height//self.mult, self.length//self.mult)

        for i in range(self.blocks):
            fx = self.convs1[i](x)
            fx = self.activ1[i](fx)
            fx = self.norm1[i](fx)
            fx = self.convs2[i](fx)
            fx = self.activ2[i](fx)
            fx = self.norm2[i](fx)

            x = x + fx

            fx = self.convs3[i](x)
            fx = self.activ3[i](fx)
            fx = self.norm3[i](fx)
            fx = self.convs4[i](fx)
            fx = self.activ4[i](fx)
            fx = self.norm4[i](fx)

            x = x + fx

            x = self.transitions_conv[i](x)
            x = self.transitions_activ[i](x)
            x = self.transitions_norm[i](x)
            x = F.upsample(x, scale_factor=2)

        x = self.final_conv(x)
        x = self.final_activ(x)

        return x


class ResNetDiscriminator(nn.Module):
    def __init__(self, image_channels=1, height=32, length=32, hidden_size=64, blocks=4):
        super(ResNetDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.blocks = blocks

        self.initial_conv = nn.Conv2d(image_channels, hidden_size, (7, 7), padding=(3, 3))
        self.initial_norm = nn.LayerNorm([hidden_size, height, length])
        self.initial_activ = nn.PReLU(hidden_size)

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** i, (3, 3), padding=(1, 1)) for
             i in range(blocks)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(
            [hidden_size * (2 ** i), height // (2 ** i), length // (2 ** i)]) for i
            in range(blocks)])
        self.activ1 = nn.ModuleList([nn.PReLU(hidden_size * (2 ** i)) for i in range(blocks)])

        self.convs2 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** i, (3, 3), padding=(1, 1)) for
             i in range(blocks)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(
            [hidden_size * (2 ** i), height // (2 ** i), length // (2 ** i)]) for i
            in range(blocks)])
        self.activ2 = nn.ModuleList([nn.PReLU(hidden_size * (2 ** i)) for i in range(blocks)])

        self.convs3 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** i, (3, 3), padding=(1, 1)) for
             i in range(blocks)])
        self.norm3 = nn.ModuleList([nn.LayerNorm(
            [hidden_size * (2 ** i), height // (2 ** i), length // (2 ** i)]) for i
            in range(blocks)])
        self.activ3 = nn.ModuleList([nn.PReLU(hidden_size * (2 ** i)) for i in range(blocks)])

        self.convs4 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** i, (3, 3), padding=(1, 1)) for
             i in range(blocks)])
        self.norm4 = nn.ModuleList([nn.LayerNorm(
            [hidden_size * (2 ** i), height // (2 ** i), length // (2 ** i)]) for i
            in range(blocks)])
        self.activ4 = nn.ModuleList([nn.PReLU(hidden_size * (2 ** i)) for i in range(blocks)])

        self.transitions_conv = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** (i+1), (3, 3), padding=(1, 1)) for
             i in range(blocks)])
        self.transitions_norm = nn.ModuleList([nn.LayerNorm(
            [hidden_size * 2 ** (i + 1), height // (2 ** i), length // (2 ** i)]) for i in
                                    range(blocks)])
        self.transitions_activ = nn.ModuleList([nn.PReLU(hidden_size * 2 ** (i + 1)) for i in range(blocks)])
        self.final_linear = nn.Linear(hidden_size * 2 ** blocks, 1)

    def forward(self, inputs):
        x = self.initial_conv(inputs)
        x = self.initial_activ(x)
        x = self.initial_norm(x)

        for i in range(self.blocks):
            fx = self.convs1[i](x)
            fx = self.activ1[i](fx)
            fx = self.norm1[i](fx)
            fx = self.convs2[i](fx)
            fx = self.activ2[i](fx)
            fx = self.norm2[i](fx)

            x = x + fx

            fx = self.convs3[i](x)
            fx = self.activ3[i](fx)
            fx = self.norm3[i](fx)
            fx = self.convs4[i](fx)
            fx = self.activ4[i](fx)
            fx = self.norm4[i](fx)

            x = x + fx
            x = self.transitions_conv[i](x)
            x = self.transitions_activ[i](x)
            x = self.transitions_norm[i](x)
            x = F.avg_pool2d(x, kernel_size=(2, 2))

        x = F.avg_pool2d(x, kernel_size=(x.shape[-2], x.shape[-1]))
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)

        return x



class DCGANGenerator(nn.Module):
    def __init__(self, input_size,  image_channels=1, height=32, length=32, hidden_size=64, blocks=4):
        super(DCGANGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.blocks = blocks
        self.height = height
        self.length = length
        self.mult = 2**blocks

        self.initial_linear = nn.Linear(input_size, hidden_size * self.mult * height//self.mult * length//self.mult)
        self.initial_activ = nn.PReLU(hidden_size * self.mult * height//self.mult * length//self.mult)
        self.initial_norm = nn.LayerNorm(hidden_size * self.mult * height//self.mult * length//self.mult)

        self.convs = nn.ModuleList([nn.Conv2d(hidden_size * 2 **(blocks - i), hidden_size * 2**(blocks - i - 1), (5, 5), padding=(2, 2)) for i in range(blocks)])
        self.activ = nn.ModuleList([nn.PReLU(hidden_size * 2**(blocks - i - 1)) for i in range(blocks)])
        self.norm = nn.ModuleList([nn.LayerNorm(
            [hidden_size * 2 ** (blocks - i - 1), height // (2 ** (blocks - i)), length // (2 ** (blocks - i))]) for i in
                       range(blocks)])

        self.final_conv = nn.Conv2d(hidden_size, image_channels, (5, 5), padding=(2, 2))
        self.final_activ = nn.Tanh()

    def forward(self, inputs):
        x = self.initial_linear(inputs)
        x = self.initial_activ(x)
        x = self.initial_norm(x)
        x = x.view(x.shape[0], self.hidden_size * self.mult, self.height//self.mult, self.length//self.mult)

        for i in range(self.blocks):
            x = self.convs[i](x)
            x = self.activ[i](x)
            x = self.norm[i](x)
            x = F.upsample(x, scale_factor=2)
        x = self.final_conv(x)
        x = self.final_activ(x)
        return x


class DCGANDiscriminator(nn.Module):
    def __init__(self, image_channels=1, height=32, length=32, hidden_size=64, blocks=4):
        super(DCGANDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.blocks = blocks



        self.initial_conv = nn.Conv2d(image_channels, hidden_size, (5, 5), padding=(2, 2))
        self.initial_norm = nn.LayerNorm([hidden_size, height, length])
        self.initial_activ = nn.PReLU(hidden_size)

        self.convs = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** (i + 1), (5, 5), padding=(2, 2)) for
             i in range(blocks)])
        self.norm = nn.ModuleList([nn.LayerNorm(
            [hidden_size * 2 ** (i + 1), height // (2 ** i), length // (2 ** i)]) for i
                                   in range(blocks)])
        self.activ = nn.ModuleList([nn.PReLU(hidden_size * 2 ** (i + 1)) for i in range(blocks)])

        self.final_linear = nn.Linear(hidden_size * 2 ** blocks * height//(2**blocks) * length//(2**blocks), 1)

    def forward(self, inputs):
        x = self.initial_conv(inputs)
        x = self.initial_norm(x)
        x = self.initial_activ(x)

        for i in range(self.blocks):
            x = self.convs[i](x)
            x = self.norm[i](x)
            x = self.activ[i](x)
            x = F.avg_pool2d(x, kernel_size=(2, 2))

        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)
        return x