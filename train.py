import os
import shutil
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import VAE
from utils import return_MVTecAD_loader

def loss_function(recon_x, x, mu, logvar):
    recon = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kld, recon, kld

def training(epoch, model, dataloader, optimizer):
    model.train()
    for idx, (inputs, _) in enumerate(dataloader):
        inputs = inputs.cuda()
        output = model(inputs)
        loss, rec_loss, kl_loss = loss_function(output, inputs, model.mu, model.logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if idx % 100 == 0:
            print('%d epoch [%d/%d] | loss: %.4f | rec loss: %.4f | KL: %.4f' % (epoch, idx, len(dataloader), loss.item(), rec_loss.item(), kl_loss.item()))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--category', type=str, default='')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed_torch', type=int, default=np.random.randint(10000000))
    parser.add_argument('--seed_numpy', type=int, default=np.random.randint(10000000))
    args = parser.parse_args()
    
    np.random.seed(args.seed_numpy)
    torch.manual_seed(args.seed_torch)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    save_dir = os.path.join(args.checkpoint, args.category)
    os.makedirs(save_dir, exist_ok=True)
    
    print('Get training dataset.')
    data_dir = os.path.join(args.data_root, args.category, 'train/good')
    training_dataloader = return_MVTecAD_loader(data_dir, 
                                                argmentation=True, 
                                                rs=512, 
                                                cs=128, 
                                                num_data=10000, 
                                                batch_size=args.batch_size, 
                                                shuffle=True)
    
    print('Build training network.')
    channel = next(iter(training_dataloader))[0].size(1)
    model = VAE(z_dim=args.z_dim, input_c=channel).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    for epoch in range(args.epochs):
        training(epoch, model, training_dataloader, optimizer)
        
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'numpy_seed': args.seed_numpy,
                    'torch_seed': args.seed_torch},
                   os.path.join(save_dir, 'model'))
    
    