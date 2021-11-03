import os
import time
import shutil
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import roc_curve, roc_auc_score


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms as T

from model import VAE
from utils import *

def optimize(model, index, save_dir, x_org, rec_x, th, alpha, lam, max_iters):
    rec_path = os.path.join(save_dir, 'reconstructed_images', 'sample_%d' % index)
    opt_path = os.path.join(save_dir, 'optimized_images', 'sample_%d' % index)
    os.makedirs(rec_path, exist_ok=True)
    os.makedirs(opt_path, exist_ok=True)
    save_image(x_org, index, 0, opt_path)
    
    loss = F.binary_cross_entropy(x_org, rec_x, reduction='sum')
    loss.backward()
    grads = x_org.grad.data
    x_t = x_org - alpha*grads*(x_org - rec_x)**2
    for i in range(max_iters):
        x_t = Variable(x_t.clamp(min=0, max=1), requires_grad=True)
        rec_x = model(x_t).detach()
        rec_loss = F.binary_cross_entropy(x_t, rec_x, reduction='sum')
        if rec_loss <= th:    break
        l1 = torch.abs(x_t - x_org).sum()
        loss = rec_loss + lam*l1
        loss.backward()
        grads = x_t.grad.data
        
        energy = grads * (x_t - rec_x)**2
        x_t = x_t - alpha*energy
        save_image(rec_x, index, i+1, rec_path)
        save_image(x_t, index, i+1, opt_path)
    
    make_gif(image_dir_path=opt_path, sample_idx=index)
    return rec_x
    
def anomaly_detection(model, dataloader, threshold_rec, save_dir, args=None):
    pred_list, label_list = [], []
    for index, (x, y, mask) in tqdm(enumerate(dataloader)):
        mask_path = os.path.join(save_dir, 'mask_images')
        os.makedirs(mask_path, exist_ok=True)
        save_image(mask, index, 0, mask_path, mask=True)
        
        x = x.cuda()
        x.requires_grad_(True)
        reconstructed_x = model(x).detach()
        if args.use_grad:
            final_x = optimize(model, index, save_dir, x, reconstructed_x, threshold_rec, 
                               alpha=args.alpha, lam=args.lam, max_iters=args.max_iters)
        else:
            final_x = reconstructed_x
        
        dssim_map = get_residual_map(final_x, x, x.size(1))
        pred_list.append(dssim_map)
        label_list.append(mask.data.numpy())
        
    label_np = np.array(label_list).reshape(-1)
    pred_np = np.array(pred_list).reshape(-1)
    fpr, tpr, thresholds = roc_curve(label_np, pred_np)
    fig = plt.figure()
    plt.plot(fpr, tpr, marker='None')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    fig.savefig(save_dir + '/roc_curve.pdf')
    fig.savefig(save_dir + '/roc_curve.png')
    
    auroc_score = roc_auc_score(label_np, pred_np)
    print("auroc_score :", auroc_score, "\n")
    
    with open(save_dir + 'auroc_score.txt', 'a') as f:
        print("Reconstruction threshold :", threshold_rec, file=f)
        print("auroc_score :", auroc_score, "\n", file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='',
                        help='Pre-trained model path')
    parser.add_argument('--data_root', type=str, default='',
                        help='MVTec dataset directory path')
    parser.add_argument('--category', type=str, default='',
                        help='Select a category in MVTec dataset.')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lam', type=float, default=0.05)
    parser.add_argument('--max_iters', type=int, default=25)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--use_grad', action='store_true',
                        help='If this argument is true, one detects anomaly samples with the iterative update using the energy function.')
    parser.add_argument('--save_dir', type=str, default='Outputs')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    save_path = os.path.join(args.save_dir, args.category)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    
    data_dir = os.path.join(args.data_root, args.category, 'train/good')
    training_dataloader = return_MVTecAD_loader(data_dir,
                                                args.category,
                                                argmentation=True, 
                                                rs=512, 
                                                cs=128, 
                                                num_data=10000,
                                                batch_size=1,
                                                shuffle=True)
    test_dataset = MVTecDataset(root_path=args.data_root, class_name=args.category, resize=512, cropsize=128)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    print("test dataset size : ", len(test_dataloader.dataset))
    
    channel = next(iter(training_dataloader))[0].size(1)
    checkpoint = torch.load(args.model_path)
    state_dict = checkpoint['model_state_dict']
    model = VAE(z_dim=100, input_c=channel).cuda()
    model.load_state_dict(state_dict)
    model.eval()
    
    print('Start to record reconstruction loss for calculating threshold.')
    training_losses = save_rec_loss(model, training_dataloader)
    
    print("----- start Anomaly detection -----------")
    result_path = os.path.join(save_path, 'results')
    os.makedirs(result_path, exist_ok=True)
    threshold_rec = np.percentile(training_losses, 0)
    print("Reconstruction threshold: ", threshold_rec)
    anomaly_detection(model, test_dataloader, threshold_rec, result_path, args=args)
    print("----- end Anomaly detection -------------")
        
        