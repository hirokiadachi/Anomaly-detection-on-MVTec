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

from model import VAE, AE
from utils import *

def optimize(model, index, save_dir, x_org, rec_x, th, alpha, lam, max_iters, loss_func):
    rec_path = os.path.join(save_dir, 'reconstructed_images', 'sample_%d' % index)
    opt_path = os.path.join(save_dir, 'optimized_images', 'sample_%d' % index)
    grad_path = os.path.join(save_dir, 'gradient_images', 'sample_%d' % index)
    enegy_path = os.path.join(save_dir, 'enegy_images', 'sample_%d' % index)
    os.makedirs(rec_path, exist_ok=True)
    os.makedirs(opt_path, exist_ok=True)
    os.makedirs(grad_path, exist_ok=True)
    os.makedirs(enegy_path, exist_ok=True)
    save_image(x_org, index, 0, opt_path)
    
    loss = loss_func(x_org, rec_x, reduction='sum')
    loss.backward()
    grads = x_org.grad.data
    x_t = x_org - alpha*grads*(x_org - rec_x)**2
    save_image(normalize(grads), index, 0, grad_path)
    save_image(normalize((x_org - rec_x)**2), index, 0, enegy_path)
    
    losses = torch.zeros(max_iters)
    decay_rate = 0.1
    minimum = 1e12
    for i in range(max_iters):
        x_t = Variable(x_t.clamp(min=0, max=1), requires_grad=True)
        rec_x = model(x_t).detach()
        rec_loss = loss_func(x_t, rec_x, reduction='sum')
        losses[i] = rec_loss.item()
        
        ## learning rate decay
        ## If you'd like to use fixed learning rate, you should comment out bellow 4 lines.
        if minimum >= rec_loss:
            minimum = min(minimum, rec_loss)
        else:
            alpha *= decay_rate
        
        if rec_loss <= th:    break
        l1 = torch.abs(x_t - x_org).sum()
        loss = rec_loss + lam*l1
        loss.backward()
        grads = x_t.grad.data
        
        energy = grads * (x_t - rec_x)**2
        ## If you'd like to use binary mask, you comment out above line and then activate bellow three lines, please.
        #mask_th = torch.flatten((x_t - rec_x)**2, start_dim=1).median(dim=1)[0]
        #binary_mask = torch.where(torch.pow(x_t - rec_x, 2)>mask_th, 1, 0)
        #energy = grads * binary_mask
        
        x_t = x_t - alpha*energy
        save_image(rec_x, index, i+1, rec_path)
        save_image(x_t, index, i+1, opt_path)
        save_image(normalize(grads), index, i+1, grad_path)
        save_image(normalize((x_t - rec_x)**2), index, i+1, enegy_path)
    
    make_gif(image_dir_path=opt_path, sample_idx=index)
    return rec_x, losses
    
def anomaly_detection(model, dataloader, threshold_rec, save_dir, loss_func, args=None):
    pred_list, label_list = [], []
    total_losses = torch.zeros(len(dataloader.dataset), args.max_iters)
    for index, (x, y, mask) in tqdm(enumerate(dataloader)):
        mask_path = os.path.join(save_dir, 'mask_images')
        os.makedirs(mask_path, exist_ok=True)
        save_image(mask, index, 0, mask_path, mask=True)
        
        x = x.cuda()
        x.requires_grad_(True)
        reconstructed_x = model(x).detach()
        if args.use_grad:
            final_x, losses = optimize(model, index, save_dir, x, reconstructed_x, threshold_rec, 
                                       alpha=args.alpha, lam=args.lam, max_iters=args.max_iters, loss_func=loss_func)
            total_losses[index] = losses
        else:
            final_x = reconstructed_x
            total_losses = None
        
        dssim_map = get_residual_map(final_x, x, x.size(1))
        pred_list.append(dssim_map)
        label_list.append(mask.data.numpy())
        
        pred_map = np.where(dssim_map>0.2, 1, 0) # (H, W)
        pred_map = torch.from_numpy(pred_map)    # (H, W)
        pred_map = pred_map.unsqueeze(0)      # (C, H, W)
        pred_map = pred_map.unsqueeze(0)   # (B, C, H, W)
        pred_path = os.path.join(save_dir, 'pred_images')
        os.makedirs(pred_path, exist_ok=True)
        save_image(pred_map, index, 0, pred_path, mask=True)
        
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
        
    #if total_losses is not None:
    #    avg_losses = torch.mean(total_losses, dim=1)

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
    parser.add_argument('--model_type', type=str, choices=['ae', 'vae'], default='ae')
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
    
    if args.model_type == 'vae':
        loss_func = F.binary_cross_entropy
        model = VAE(z_dim=100, input_c=channel).cuda()
    elif args.model_type == 'ae':
        loss_func = l2_squared
        model = AE(z_dim=100, input_c=channel).cuda()
    model.load_state_dict(state_dict)
    model.eval()
    
    print('Start to record reconstruction loss for calculating threshold.')
    training_losses = save_rec_loss(model, training_dataloader, loss_func)
    
    print("----- start Anomaly detection -----------")
    result_path = os.path.join(save_path, 'results')
    os.makedirs(result_path, exist_ok=True)
    threshold_rec = np.percentile(training_losses, 0)
    print("Reconstruction threshold: ", threshold_rec)
    anomaly_detection(model, test_dataloader, threshold_rec, result_path, loss_func, args=args)
    print("----- end Anomaly detection -------------")
        
        
