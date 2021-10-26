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
from utils import return_MVTecAD_loader, MVTecDataset, torch_img_to_numpy, get_residual_map

def save_dssim_rec(model, dataloader):
    cnt = 0
    training_losses = np.zeros(len(dataloader.dataset))
    total_rec_ssim = np.zeros((len(dataloader.dataset), 128, 128))
    for (inputs, _) in tqdm(dataloader):
        inputs = inputs.cuda()
        batch = inputs.size(0)
        with torch.no_grad():
            rec_x = model(inputs)
        loss = F.binary_cross_entropy(rec_x, inputs, reduction='sum').cpu().data.numpy()
        ssim_res_map = get_residual_map(rec_x, inputs, inputs.size(1))
        training_losses[cnt:cnt+batch] = loss
        total_rec_ssim[cnt:cnt+batch]  = ssim_res_map
        cnt += batch
        
    print("train loss size :", len(training_losses))
    print("train ssim size :", len(total_rec_ssim))
    return training_losses, total_rec_ssim

def save_image(x, sample_idx, i, dirpath, mask=None, gif=False):
    save_dir_name = os.path.join(dirpath, 'sample_%d' % sample_idx)
    os.makedirs(save_dir_name, exist_ok=True)
    if x.size(1) > 1:
        img = Image.fromarray((x * 255).clamp(min=0, max=255).squeeze().permute(1,2,0).data.cpu().to(torch.uint8).numpy())
    else:
        img = Image.fromarray((x * 255).clamp(min=0, max=255).squeeze().data.cpu().to(torch.uint8).numpy())
    
    if mask:
        img.save(os.path.join(save_dir_name, 'mask.png'))
    else:
        img.save(os.path.join(save_dir_name, '{:0>6}.png'.format(i)))
    
    if gif:
        img_list = sorted(os.listdir(save_dir_name))[:-2]
        PATH_LIST = [Image.open(os.path.join(save_dir_name, img_list[i])) for i in range(len(img_list))]
        PATH_LIST[0].save(os.path.join(save_dir_name, 'sample_%d.gif' % (sample_idx)), save_all=True, append_images=PATH_LIST[1:])
    
def anomaly_detection(model, dataloader, threshold_rec, threshold_cls, save_dir, lower=0, upper=1, args=None):
    pred_list, label_list = [], []
    for index, (x, y, mask) in tqdm(enumerate(dataloader), desc='Quantile %d' % quantile):
        x = x.cuda()
        save_image(x, index, 0, save_dir, gif=False)
        save_image(mask, index, 0, save_dir, mask=True, gif=False)
        label = mask.max()
        x.requires_grad_(True)
        rec_x = model(x).detach()
        loss = F.binary_cross_entropy(x, rec_x, reduction='sum')
        loss.backward()
        
        grads = x.grad.data
        x_t = x - args.alpha*grads*(x - rec_x)**2
        
        for i in range(args.max_iters):
            x_t = torch.clamp(x_t, min=0, max=1)
            x_t = Variable(x_t, requires_grad=True)
            rec_x = model(x_t).detach()
            rec_loss = F.binary_cross_entropy(x_t, rec_x, reduction='sum')
            if rec_loss <= threshold_rec:    break
            l1 = torch.abs(x_t - x).sum()
            loss = rec_loss + args.lam*l1
            loss.backward()
            grads = x.grad.data
            
            enagy = grads * (x_t - rec_x)**2
            x_t = x_t - args.alpha*enagy
            save_image(x_t, index, i+1, save_dir, gif=True)
        
        ssim_res_map = get_residual_map(x_t, x, x.size(1))
        mask_pred = np.zeros((128, 128))
        mask_pred[ssim_res_map > threshold_cls] = 1
        
        pred_list.append(mask_pred)
        label_list.append(mask.data.numpy())
    
    label_np = np.array(label_list).reshape(-1)
    pred_np = np.array(pred_list).reshape(-1)
    fpr, tpr, thresholds = roc_curve(label_np, pred_np)
    fig = plt.figure()
    plt.plot(fpr, tpr, marker='None')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    fig.savefig(save_dir + '/roc_curve_quantile_'+str(quantile)+'.pdf')
    fig.savefig(save_dir + '/roc_curve_quantile_'+str(quantile)+'.png')
    
    auroc_score = roc_auc_score(label_np, pred_np)
    print("auroc_score :", auroc_score, "\n")
    
    with open(save_dir + 'auroc_score.txt', 'a') as f:
        print("quantile :", quantile, file=f)
        print("Reconstruction threshold :", threshold_rec, file=f)
        print("Classification threshold :", threshold_cls, file=f)
        print("auroc_score :", auroc_score, "\n", file=f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--category', type=str, default='')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lam', type=float, default=0.05)
    parser.add_argument('--max_iters', type=int, default=25)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_dir', type=str, default='Outputs')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    save_path = os.path.join(args.save_dir, args.category)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    
    data_dir = os.path.join(args.data_root, args.category, 'train/good')
    training_dataloader = return_MVTecAD_loader(data_dir, 
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
    
    print('Start to record reconstruction loss and DSSIM score for calculating threshold.')
    training_losses, total_rec_ssim = save_dssim_rec(model, training_dataloader)
    
    print("----- start Anomaly detection -----------")
    quantiles = np.arange(start=0, stop=110, step=10)
    for quantile in quantiles:
        result_path = os.path.join(save_path, 'quantiles%d'%quantile)
        os.makedirs(result_path, exist_ok=True)
        threshold_rec = np.percentile(training_losses, quantile)
        print("Reconstruction threshold: ", threshold_rec)
        threshold_cls = float(np.percentile(total_rec_ssim, quantile))
        print("Classification threshold :", threshold_cls)
        anomaly_detection(model, test_dataloader, threshold_rec, threshold_cls, result_path, lower=0, upper=1, args=args)
    print("----- end Anomaly detection -------------")
        
        