""" Training script.
usage: train.py [options]

options:
    --inner_learning_rate=ilr   Learning rate of inner loop [default: 1e-3]
    --outer_learning_rate=olr   Learning rate of outer loop [default: 1e-4]
    --batch_size=bs             Size of task to train with [default: 4]
    --inner_epochs=ie           Amount of meta epochs in the inner loop [default: 10]
    --height=h                  Height of image [default: 28]
    --length=l                  Length of image [default: 28]
    --dataset=ds                Dataset name (Mnist, Omniglot, FIGR8) [default: Mnist]
    --network=net               Either ConvAE or ? [default: ConvAE]
    -h, --help                  Show this help message and exit
"""
from docopt import docopt

import torch
import torch.optim as optim
import torch.autograd as autograd

from tensorboardX import SummaryWriter
import numpy as np
import os
from environments import MnistMetaEnv, OmniglotMetaEnv, FIGR8MetaEnv
from model import ConvAE

import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def wassertein_loss(inputs, targets):
    return torch.mean(inputs * targets)

def normalize_data(data):
    data *= 2
    data -= 1
    return data

def unnormalize_data(data):
    data += 1
    data /= 2
    return data

class FIGR:
    def __init__(self, args):
        self.load_args(args)
        self.id_string = self.get_id_string()
        self.z_shape = 100
        self.writer = SummaryWriter('logs/' + self.id_string)
        self.env = eval(self.dataset + 'MetaEnv(height=self.height, length=self.length)')
        self.initialize_vae()
        self.load_checkpoint()
    
    def load_args(self, args):
        self.outer_learning_rate = float(args['--outer_learning_rate'])
        self.inner_learning_rate = float(args['--inner_learning_rate'])
        self.batch_size = int(args['--batch_size'])
        self.inner_epochs = int(args['--inner_epochs'])
        self.height = int(args['--height'])
        self.length = int(args['--length'])
        self.dataset = args['--dataset']
        self.network = args['--network']
        self.x_dim = int(args['--height']) * int(args['--length'])
    
    def get_id_string(self):
        return '{}_{}_olr{}_ilr{}_bsize{}_ie{}_h{}_l{}'.format(
            self.network,
            self.dataset,
            str(self.outer_learning_rate),
            str(self.inner_learning_rate),
            str(self.batch_size),
            str(self.inner_epochs),
            str(self.height),
            str(self.length)
        )
    
    def load_checkpoint(self):
        if os.path.isfile('logs/' + self.id_string + '/checkpoint'):
            checkpoint = torch.load('logs/' + self.id_string + '/checkpoint')
            self.netAE.load_state_dict(checkpoint['convae'])
            self.eps = checkpoint['episode']
        else:
            self.eps = 0
    
    def initialize_vae(self):
        # D and G on CPU since they never to a feed forward operation
        self.netAE = eval(self.network + '(self.env.channels, self.env.height, self.z_shape)')
        self.meta_netAE = eval(self.network + '(self.env.channels, self.env.height, self.z_shape)').to(device)
        self.netAE_optim = optim.Adam(params=self.netAE.parameters(), lr=self.outer_learning_rate)
        self.meta_netAE_optim = optim.SGD(params=self.meta_netAE.parameters(), lr=self.inner_learning_rate)
    
    def reset_meta_model(self):
        self.meta_netAE.train()
        self.meta_netAE.load_state_dict(self.netAE.state_dict())
    
    def inner_loop(self, real_batch):
        self.meta_netAE.train()
        self.meta_netAE_optim.zero_grad()
        
        recon_batch, z = self.meta_netAE(real_batch)
        
        reconstruction_loss = F.binary_cross_entropy(recon_batch.squeeze().view(-1, self.x_dim), real_batch.view(-1, self.x_dim), reduction='sum')
        
        reconstruction_loss.backward()
        self.meta_netAE_optim.step()

        return reconstruction_loss.item()

    
    def meta_training_loop(self):
        data, task = self.env.sample_training_task(self.batch_size)
        #data = normalize_data(data)
        real_batch = data.to(device)

        convae_total_loss = 0

        for _ in range(self.inner_epochs):
            recon_loss = self.inner_loop(real_batch)
            convae_total_loss += recon_loss
        
        self.writer.add_scalar('Training_convae_loss', convae_total_loss, self.eps)

        #print("Meta Loss:", convae_total_loss)
         
         # Updating both generator and discriminator
        for p, meta_p in zip(self.netAE.parameters(), self.meta_netAE.parameters()):
             diff = p - meta_p.cpu()
             p.grad = diff
        self.netAE_optim.step()
    
    def validation_run(self):
        data, task = self.env.sample_validation_task(self.batch_size)
        training_images = data.cpu().numpy()
        training_images = np.concatenate([training_images[i] for i in range(self.batch_size)], axis=-1)
        #data = normalize_data(data)
        real_batch = data.to(device)

        convae_total_loss = 0

        for _ in range(self.inner_epochs):
            recon_loss = self.inner_loop(real_batch)
            convae_total_loss += recon_loss
        
        self.meta_netAE.eval()
        with torch.no_grad():
            img = self.meta_netAE.decode(torch.tensor(np.random.normal(size=(self.batch_size * 3, self.z_shape)), dtype=torch.float, device=device))
            img = img.cpu()
            img = np.concatenate([np.concatenate([img[i * 3 + j] for j in range(3)], axis=-2) for i in range(self.batch_size)], axis=-1)
            #img = unnormalize_data(img)
            img = np.concatenate([training_images, img], axis=-2)
            self.writer.add_image('Validation_generated', img, self.eps)
            self.writer.add_scalar('Validation_convae_loss', convae_total_loss, self.eps)
            
            plt.imshow(img[0], cmap='Greys_r')
            plt.show()
            print("Loss:", convae_total_loss)
            

    def training(self):
        while self.eps <= 1000000:
            self.reset_meta_model()
            self.meta_training_loop()

            # Validation run every 10000 training loop
            if self.eps % 100 == 0:
                self.validation_run()
                self.checkpoint_model()
            self.eps += 1
    
    def checkpoint_model(self):
        checkpoint = {'convae': self.netAE.state_dict(),
                      'episode': self.eps}
        torch.save(checkpoint, 'logs/' + self.id_string + '/checkpoint')

if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    env = FIGR(args)
    env.training()
