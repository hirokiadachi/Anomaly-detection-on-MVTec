import os
import numpy as np
import multiprocessing
from tqdm import tqdm
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms as T

class MVTecAD(Dataset):
    def __init__(self, image_dir, transform):
        self.transform = transform
        self.image_dir = image_dir
        
    def __len__(self):
        return len(os.listdir(self.image_dir))
    
    def __getitem__(self, i):
        filename = '{:0>3}.png'.format(i)
        image = Image.open(os.path.join(self.image_dir, filename))
        if self.transform:
            image = self.transform(image)
        return image, torch.zeros(1)
    
def return_MVTecAD_loader(image_dir, argmentation=True, rs=512, cs=128, num_data=10000, batch_size=128, shuffle=True):
    transform = T.Compose([
        T.Resize((rs, rs)),
        T.RandomCrop((cs, cs)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ToTensor(),
    ])
    
    dataset = MVTecAD(image_dir=image_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=shuffle, num_workers=multiprocessing.cpu_count())
    
    if argmentation:
        cnt = 0
        channel = iter(data_loader).next()[0].size(1)
        stacked_data = torch.zeros(num_data, channel, cs, cs)
        stacked_labels = torch.zeros(num_data)
        with tqdm(total=num_data, desc='Augmentation') as pbar:
            while cnt < num_data:
                for (img, _) in data_loader:
                    batch = img.size(0)
                    if cnt+batch > num_data: batch=num_data - cnt
                    stacked_data[cnt:cnt+batch] = img[:batch]
                    cnt += batch
                    pbar.update(batch)
        cstm_dataset = TensorDataset(stacked_data, stacked_labels)
        cstm_dataloader = DataLoader(dataset=cstm_dataset, batch_size=batch_size, shuffle=True)
    else:
        cstm_dataloader = data_loader
    
    print("train data size :", len(cstm_dataloader.dataset))
    
    return cstm_dataloader


CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
class MVTecDataset(Dataset):
    def __init__(self, root_path='./mvtec_anomaly_detection', class_name='bottle', resize=512, cropsize=128):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)

        self.class_name = class_name
        self.resize     = resize
        self.cropsize   = cropsize
        self.mvtec_folder_path = root_path

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        if class_name in ['carpet','grid','leather','tile','wood']:
            print("Textures category")
            self.transform_x    = T.Compose([ T.Resize(resize),  # T.Resize(resize, Image.ANTIALIAS)
                                              T.CenterCrop(cropsize),
                                              T.ToTensor(), ])
            self.transform_mask = T.Compose([ T.Resize(resize, Image.NEAREST),
                                              T.CenterCrop(cropsize),
                                              T.ToTensor()])
        else:
            print("Objects category")
            self.transform_x    = T.Compose([ T.Resize((128,128)),  # T.Resize(resize),
                                              T.ToTensor(), ])
            self.transform_mask = T.Compose([ T.Resize((128,128),Image.NEAREST),  # T.Resize(resize, Image.NEAREST),
                                              T.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x) #.convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        x, y, mask = [], [], []

        img_dir = os.path.join(self.mvtec_folder_path, self.class_name, 'test')
        gt_dir = os.path.join(self.mvtec_folder_path, self.class_name, 'ground_truth')
        print("img_dir :", img_dir)
        print("gt_dir  :", gt_dir)

        img_types = sorted(os.listdir(img_dir))
        print("img_types :", img_types)

        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
    
def torch_img_to_numpy(img):
    # 値の変換
    img = img.detach().cpu().numpy()
    img *= 255.  # 値の範囲を[0～1]から[0～255]に変更
    img = np.transpose(img, (0,2,3,1))  # [データ数, c, h, w]から[データ数, h, w, c]に変更
    img = np.squeeze(img)  # (カラー画像)[c, h, w] or (グレースケール画像)[h, w]に変更
    return img

def get_residual_map(recon_img, input_img, img_c):
    recon_img = torch_img_to_numpy(recon_img)
    input_img = torch_img_to_numpy(input_img)

    # DSSIMを計算
    if img_c == 1:
        ssim_residual_map = 1 - ssim(input_img, recon_img, win_size=11, full=True)[1]
    else:
        ssim_residual_map = ssim(input_img, recon_img, win_size=11, full=True, multichannel=True)[1]
        ssim_residual_map = 1 - np.mean(ssim_residual_map, axis=2)

    return ssim_residual_map / 2