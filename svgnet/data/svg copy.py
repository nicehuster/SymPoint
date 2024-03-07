import numpy as np
import torch
from torch.utils.data import Dataset

import os.path as osp
from glob import glob
import json
import math
import random
from .aug_utils import *

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

    CLASSES = tuple([x["name"] for x in SVG_CATEGORIES])

    def __init__(self, data_root, split,data_norm,aug, repeat=1, logger=None):
        
        self.split = split
        self.data_norm = data_norm
        self.aug = aug
        self.repeat = repeat
        self.data_list = glob(osp.join(data_root,split,"*_s.json"))
        logger.info(f"Load {split} dataset: {len(self.data_list)} svg")
        self.data_idx = np.arange(len(self.data_list))
        
        self.instance_queues = []

    def __len__(self):
        return len(self.data_list)*self.repeat
    
    @staticmethod
    def load(json_file,idx,min_points=2048):
        data = json.load(open(json_file))
        args = np.array(data["args"]).reshape(-1,8)/ 140
        num = args.shape[0]
        max_num = max(num,min_points)
        
        coord = np.zeros((max_num,3))
        coord_x = np.mean(args[:,0::2],axis=1) 
        coord_y = np.mean(args[:,1::2],axis=1) 
        coord_z = np.zeros((num,))
        
        #coord_x = 2 * coord_x - 1
        #coord_y = 2 * coord_y - 1
        
        coord[:num,0] = coord_x
        coord[:num,1] = coord_y
        coord[:num,2] = coord_z
       
        lengths = np.zeros(max_num)
        lengths[:num] = np.array(data["lengths"])
       
        feat = np.zeros((max_num,6))
        arc = np.arctan(coord_y/(coord_x + 1e-8)) / math.pi
        lens = np.array(data["lengths"]).clip(0,140) / 140
        ctype = np.eye(4)[data["commands"]]
        
        feat[:num,0] = arc
        feat[:num,1] = lens
        feat[:num,2:] = ctype
        
        semanticIds = np.full_like(coord[:,0],35) # bg sem id = 35
        seg = np.array(data["semanticIds"])
        semanticIds[:num] = seg
        semanticIds = semanticIds.astype(np.int64)
        
        instanceIds = np.full_like(coord[:,0],-1) # stuff id = -1
        ins = np.array(data["instanceIds"])
        valid_pos = ins != -1
        ins[valid_pos] += idx*min_points
        
        instanceIds[:num] = ins
        instanceIds = instanceIds.astype(np.int64)
        label = np.concatenate([semanticIds[:,None],instanceIds[:,None]],axis=1)
        return coord, feat, label,lengths
    
    def __getitem__(self, idx):
        
        data_idx = self.data_idx[idx % len(self.data_idx)]
        json_file = self.data_list[data_idx]
        coord, feat, label,lengths = SVGDataset.load(json_file,idx)
        
        if self.split=="train":
            return self.transform_train(coord, feat, label)
        else:
            return self.transform_test(coord, feat, label,lengths)
    
    def transform_train(self,coord, feat, label):
        
        # hflip
        if self.aug.hflip and np.random.rand() < self.aug.aug_prob:
            args = RandomHorizonFilp(coord[:,:2],width=1)
            coord[:,:2] = args
        
        # vflip
        if self.aug.vflip and np.random.rand() < self.aug.aug_prob:
            args = RandomVerticalFilp(coord[:,:2],Hight=1)
            coord[:,:2] = args
            
        # rotate
        if self.aug.rotate.enable and np.random.rand() < self.aug.aug_prob:
            _min, _max = self.aug.rotate.angle
            angle = random.uniform(_min,_max)
            args = rotate_xy(coord[:,:2],width=1,height=1,angle=angle)
            coord[:,:2] = args
        
        if self.aug.rotate2 and np.random.rand() < self.aug.aug_prob:
            args = random_rotate(coord[:,:2],width=1,height=1)
            coord[:,:2] = args
        
        # random shift
        if self.aug.shift.enable and np.random.rand() < self.aug.aug_prob:
            _min, _max = self.aug.shift.scale
            scale = np.random.uniform(_min, _max,3)
            scale[2] = 0
            coord += scale
            
        # random scale
        if self.aug.scale.enable and np.random.rand() < self.aug.aug_prob: 
            _min, _max = self.aug.scale.ratio
            scale = np.random.uniform(_min, _max,1)
            coord *= scale
            feat[:,1] = feat[:,1] * scale
            
        mix_coord, mix_feat, mix_label = [], [], []
        mix_coord.append(coord)
        mix_feat.append(feat)
        mix_label.append(label)
        
        # random cutmix
        if self.aug.cutmix.enable and np.random.rand() < self.aug.aug_prob:
            
            unique_label = np.unique(label,axis=0)
            for sem,ins in unique_label:
                if sem >=30: continue
                valid = np.logical_and(label[:,0]==sem,label[:,1]==ins)
                if len(self.instance_queues)<=self.aug.cutmix.queueK: 
                    self.instance_queues.insert(0,{
                        "coord":coord[valid],
                        "feat": feat[valid],
                        "label": label[valid]
                        })
                else:
                    self.instance_queues.pop()
            _min, _max = self.aug.cutmix.relative_shift
            rand_pos = np.random.uniform(_min, _max,3)
            rand_pos[2] = 0
            for instance in self.instance_queues:
                mix_coord.append(instance["coord"]+rand_pos) # random shift
                mix_feat.append(instance["feat"])
                mix_label.append(instance["label"])
        
        coord = np.concatenate(mix_coord,axis=0)
        feat = np.concatenate(mix_feat,axis=0)
        feat[:,0] = np.arctan(coord[:,1]/(coord[:,0] + 1e-8)) / math.pi     # feature should be change
        
        label = np.concatenate(mix_label,axis=0)
        
        # shuffle
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat = coord[shuf_idx], feat[shuf_idx]
        if label is not None:
            label = label[shuf_idx]
            
        # coord norm
        if self.data_norm == 'mean':
            coord -= np.mean(coord, 0)
        elif self.data_norm == 'min':
            coord -= np.min(coord, 0)
        return torch.FloatTensor(coord), torch.FloatTensor(feat), torch.LongTensor(label) , None

    def transform_test(self,coord, feat, label,lengths):
        
        # coord norm
        if self.data_norm == 'mean':
            coord -= np.mean(coord, 0)
        elif self.data_norm == 'min':
            coord -= np.min(coord, 0)
        return torch.FloatTensor(coord), torch.FloatTensor(feat), torch.LongTensor(label), torch.FloatTensor(lengths)
        

    def collate_fn(self,batch):
        coord, feat, label,lengths = list(zip(*batch))
        offset, count = [], 0
        for item in coord:
            count += item.shape[0]
            offset.append(count)
        lengths = torch.cat(lengths) if lengths[0] is not None else None
        return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset),lengths
  
