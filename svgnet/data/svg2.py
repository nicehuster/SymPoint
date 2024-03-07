import numpy as np
import torch
from torch.utils.data import Dataset

import os.path as osp
from glob import glob
import json
import math
import random
from .aug_utils2 import *
SVG_CATEGORIES = [
    #1-6 doors
    {"color": [224, 62, 155], "isthing": 1, "id": 1, "name": "single door"},
    {"color": [157, 34, 101], "isthing": 1, "id": 2, "name": "double door"},
    {"color": [232, 116, 91], "isthing": 1, "id": 3, "name": "sliding door"},
    {"color": [101, 54, 72], "isthing": 1, "id": 4, "name": "folding door"},
    {"color": [172, 107, 133], "isthing": 1, "id": 5, "name": "revolving door"},
    {"color": [142, 76, 101], "isthing": 1, "id": 6, "name": "rolling door"},
    #7-10 window
    {"color": [96, 78, 245], "isthing": 1, "id": 7, "name": "window"},
    {"color": [26, 2, 219], "isthing": 1, "id": 8, "name": "bay window"},
    {"color": [63, 140, 221], "isthing": 1, "id": 9, "name": "blind window"},
    {"color": [233, 59, 217], "isthing": 1, "id": 10, "name": "opening symbol"},
    #11-27: furniture
    {"color": [122, 181, 145], "isthing": 1, "id": 11, "name": "sofa"},
    {"color": [94, 150, 113], "isthing": 1, "id": 12, "name": "bed"},
    {"color": [66, 107, 81], "isthing": 1, "id": 13, "name": "chair"},
    {"color": [123, 181, 114], "isthing": 1, "id": 14, "name": "table"},
    {"color": [94, 150, 83], "isthing": 1, "id": 15, "name": "TV cabinet"},
    {"color": [66, 107, 59], "isthing": 1, "id": 16, "name": "Wardrobe"},
    {"color": [145, 182, 112], "isthing": 1, "id": 17, "name": "cabinet"},
    {"color": [152, 147, 200], "isthing": 1, "id": 18, "name": "gas stove"},
    {"color": [113, 151, 82], "isthing": 1, "id": 19, "name": "sink"},
    {"color": [112, 103, 178], "isthing": 1, "id": 20, "name": "refrigerator"},
    {"color": [81, 107, 58], "isthing": 1, "id": 21, "name": "airconditioner"},
    {"color": [172, 183, 113], "isthing": 1, "id": 22, "name": "bath"},
    {"color": [141, 152, 83], "isthing": 1, "id": 23, "name": "bath tub"},
    {"color": [80, 72, 147], "isthing": 1, "id": 24, "name": "washing machine"},
    {"color": [100, 108, 59], "isthing": 1, "id": 25, "name": "squat toilet"},
    {"color": [182, 170, 112], "isthing": 1, "id": 26, "name": "urinal"},
    {"color": [238, 124, 162], "isthing": 1, "id": 27, "name": "toilet"},
    #28:stairs
    {"color": [247, 206, 75], "isthing": 1, "id": 28, "name": "stairs"},
    #29-30: equipment
    {"color": [237, 112, 45], "isthing": 1, "id": 29, "name": "elevator"},
    {"color": [233, 59, 46], "isthing": 1, "id": 30, "name": "escalator"},

    #31-35: uncountable
    {"color": [172, 107, 151], "isthing": 0, "id": 31, "name": "row chairs"},
    {"color": [102, 67, 62], "isthing": 0, "id": 32, "name": "parking spot"},
    {"color": [167, 92, 32], "isthing": 0, "id": 33, "name": "wall"},
    {"color": [121, 104, 178], "isthing": 0, "id": 34, "name": "curtain wall"},
    {"color": [64, 52, 105], "isthing": 0, "id": 35, "name": "railing"},
    {"color": [0, 0, 0], "isthing": 0, "id": 36, "name": "bg"},
]

class SVGDataset(Dataset):


    def __init__(self, data_root, split,data_norm, aug, repeat=1, logger=None):
        
        self.split = split
        self.data_norm = data_norm
        self.aug = aug
        self.repeat = repeat
        
        self.data_list = glob(osp.join(data_root,split,"*.json"))
        logger.info(f"Load {split} dataset: {len(self.data_list)} svg")
        self.data_idx = np.arange(len(self.data_list))
        
        
        
    def __len__(self):
        return len(self.data_list)*self.repeat
    
    
    def load(self, json_file, idx ):
        
        json_file = json_file.replace(".svg",".json")
        data = json.load(open(json_file))
        width, height = data["width"], data["height"]
        coords = data["args"]
        coords = np.array(coords).reshape(-1,4,2)
        neighbors = np.array(data["neighbors"]).reshape(-1,16)
        
        lens = np.array(data["lengths"])
        ctypes = np.array(data["commands"])
        
        seg = np.array(data["semanticIds"])
        ins = np.array(data["instanceIds"])
        
        valid_pos = ins != -1
        ins[valid_pos] += idx*2048  # avoid repeat id
        
        labels = np.stack([seg, ins]).swapaxes(0, 1) # n,2
        
        return coords, lens, ctypes, neighbors, labels, width, height
        
             
    def train_aug(self, coords, lens, labels, ctypes, neighbors, width, height):
        
        # flip
        if self.aug.hflip and np.random.rand() < self.aug.aug_prob:
            coords = hfilp(coords, width)
        if self.aug.vflip and np.random.rand() < self.aug.aug_prob:
            coords = vflip(coords, height)
        
        # rotate
        if self.aug.rotate and np.random.rand() < self.aug.aug_prob:
            coords = random_rotate(coords, width,height)
         
        # scale
        if self.aug.scale and np.random.rand() < self.aug.aug_prob:
            coords, lens = random_scale(coords,lens)
            
        # shift
        if self.aug.shift and np.random.rand() < self.aug.aug_prob:
            coords = random_shift(coords, width, height)
            
        # delete
        if self.aug.delete and np.random.rand() < self.aug.aug_prob:
            coords, lens, labels, ctypes, neighbors = \
                random_delete(coords, lens, labels, ctypes, neighbors)

      
        return  coords, lens, ctypes, neighbors, labels

    def angles_with_horizontal(self, coords):
        
        
        start_points, end_points = coords[:,0,:], coords[:,-1,:]
        lines = np.concatenate([start_points[:,None,:],
                        end_points[:,None,:]],axis=1)
        x1, y1 = lines[:, 0, 0], lines[:, 0, 1]
        x2, y2 = lines[:, 1, 0], lines[:, 1, 1]
        slopes = (y2 - y1) / (x2 - x1 + 1e-8)
        slopes[x2 - x1 == 0] = np.inf
        theta_radians = np.arctan(slopes)
        return theta_radians
    
    def extract_feat(self,coords, lens, labels, ctypes, neighbors, width, height, k=4):
        
        min_points = 2048
        num = coords.shape[0]
        max_num = max(num+1,min_points) # pad one
        coords[:, :, 0] = coords[:, :, 0] / width
        coords[:, :, 1] = coords[:, :, 1] / height
        
        _neighbors = neighbors[:,:k]
        neighbors = np.full_like(np.zeros((max_num,k)),-1)
        neighbors[:num] = _neighbors
        neighbors = neighbors.astype(np.int64)
        
        coord = np.zeros((max_num,3))
        coord[:num,0] = coords.mean(1)[:, 0]
        coord[:num,1] = coords.mean(1)[:, 1]
        
        angles = self.angles_with_horizontal(coords)
        # ['Line', 'Arc','circle', 'ellipse']
        f1, f2 = np.cos(2*angles), np.sin(2*angles)
        ind = np.where(ctypes>=2)[0]
        f1[ind], f2[ind] = 0, 0
        
        max_lengths = max(width, height)
        lens = np.array(lens) / max_lengths
        ctype = np.eye(4)[ctypes]
        
        feat = np.zeros((max_num,7))
        feat[:num, 0] = f1
        feat[:num, 1] = f2
        feat[:num, 2] = lens
        feat[:num, 3:] = ctype
        
        label = np.zeros((max_num,2))
        label[:, 0] = 35
        label[:, 1] = -1
        label[:num] = labels
        
        return coord, feat, label, neighbors
   
    
    def transform(self, coords, lens, labels, ctypes, neighbors, width, height):
        
        
        if self.split=="train":
            
            coords, lens, ctypes, neighbors, labels = self.train_aug(
                coords, lens, labels, ctypes, neighbors, width, height
                )
            coord, feat, label, neighbors = self.extract_feat(coords, lens, labels, 
                                                              ctypes, neighbors, width, height)
            
            # shuffle
            shuf_idx = np.arange(coord.shape[0])
            np.random.shuffle(shuf_idx)
            coord = coord[shuf_idx]
            feat = feat[shuf_idx]
            label = label[shuf_idx]
            
            neighbors[neighbors<=-1] = coord.shape[0] - 1 
            neighbors= neighbors[shuf_idx]
            index_map = {old: new  for new, old in enumerate(shuf_idx)}
            neighbors = np.vectorize(index_map.get)(neighbors)
            
        else:
            coord, feat, label, neighbors = self.extract_feat(coords, lens, labels, ctypes, 
                                                              neighbors, width, height)
            neighbors[neighbors<=-1] = coord.shape[0] - 1 
        
            
        # coord norm
        if self.data_norm == 'mean':
            coord -= np.mean(coord, 0)
        elif self.data_norm == 'min':
            coord -= np.min(coord, 0)
        return torch.FloatTensor(coord), torch.FloatTensor(feat), torch.LongTensor(label), torch.LongTensor(neighbors), torch.FloatTensor(lens)
    
    

    
    def __getitem__(self, idx):
        
        data_idx = self.data_idx[idx % len(self.data_idx)]
        json_file = self.data_list[data_idx]
        
        coords, lens, ctypes, neighbors, labels, width, height = self.load(json_file, idx)
        return self.transform(coords, lens, labels, ctypes, neighbors, width, height)
       
        
    def find_positions(self,lst, arr):
        lst_dict = {value: idx for idx, value in enumerate(lst)}
        get_index = np.vectorize(lst_dict.get)
        return get_index(arr, -1)

    def collate_fn(self,batch):
        coord, feat, label, neighbors,lengths = list(zip(*batch))
        offset, count = [], 0
        for item in coord:
            count += item.shape[0]
            offset.append(count)
        lengths = torch.cat(lengths) if lengths[0] is not None else None
        return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset), lengths
  
