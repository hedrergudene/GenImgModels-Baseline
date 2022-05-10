# Requirements
from typing import List, Dict, Iterable, Tuple
import json
import os
import numpy as np
import torch
import torchvision
import cv2


# Albumentations transform
transforms = torchvision.transforms.Compose([
                 torchvision.transforms.RandomRotation(10),
                 torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2),
                 torchvision.transforms.RandomHorizontalFlip(p=.5),
                 torchvision.transforms.GaussianBlur(kernel_size=(3,3)),
             ])


# Torch dataset
class GenImgModelsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_config_path:str,
                 data_flag:str,
                 folder:str,
                 transforms:torchvision.transforms=transforms,
                 elems:List[Tuple[str]]=None,
                 ):
        # Parameters
        with open(data_config_path, 'r') as f:
            self.data_dct = json.load(f)[data_flag]
        self.path = os.path.join('input',self.data_dct['python_class'],folder)
        self.label2idx = {v:int(k) for k,v in self.data_dct['label'].items()}
        if elems is not None:
            self.elems = elems
        else:
            self.elems = [(os.path.join(self.path, label, img_name), self.label2idx[label]) for label in os.listdir(self.path) for img_name in os.listdir(os.path.join(self.path,label))]
        self.transforms = transforms

    def __len__(self):
        return len(self.elems)

    def __getitem__(self, idx):
        img_path, label = self.elems[idx]
        if self.data_dct['n_channels']==1:
            image = torch.from_numpy(np.expand_dims(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), axis=0).astype(float)/255.).type(torch.FloatTensor)
        else:
            image = torch.from_numpy(np.transpose(cv2.imread(img_path, cv2.IMREAD_COLOR), (2,0,1)).astype(float)/255.).type(torch.FloatTensor)
        if self.transforms is not None:
            image = self.transforms(image)
        label = torch.LongTensor([label])
        return {"x":image, "y":label}
