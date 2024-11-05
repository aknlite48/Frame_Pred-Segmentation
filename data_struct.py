import os
import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SegVid(Dataset):
    def __init__(self,fpath,num):
        self.vid = []
        self.masks = []
        self.imgs = []
        dirs = sorted(os.listdir(fpath))
        dirs = [d for d in dirs if d[0]=='v']
        for path in tqdm.tqdm(dirs[:num]):
            if path[0]!='v':
                continue
            video = []
            for i in range(22):
                frame_path=os.path.join(fpath,path,'image_'+str(i)+'.png')
                frame = (transforms.ToTensor())(Image.open(frame_path).convert('RGB'))
                video.append(frame)
            video=torch.stack(video)
            self.vid.append(video)
            frame_path=os.path.join(fpath,path,'image_'+str(21)+'.png')
            frame = (transforms.ToTensor())(Image.open(frame_path).convert('RGB'))
            self.imgs.append(frame)
            mask_path = os.path.join(fpath,path,'mask.npy')
            mask = torch.Tensor(np.load(mask_path))
            self.masks.append(mask[21])

    def __len__(self):
        return len(self.vid)

    def __getitem__(self,idx):
        return self.vid[idx][:11],self.masks[idx]

    def get_image(self,idx):
        return self.imgs[idx],self.masks[idx]
    def get_vid(self,idx):
        return self.vid[idx][11:]