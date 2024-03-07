

import random
import numpy as np

def hfilp(args, width):
    
    args[:,:,0::2] = width - args[:,:,0::2]
    return args

def vflip(args, hight):
    
    args[:,:,1::2] = hight - args[:,:,1::2]
    return args


def random_rotate(points,width,height):
    
    angle = random.uniform(-180,180)
    # 将点移动到原点
    centroid = [width/2, height/2]
    points_centered = points - centroid

    # 旋转矩阵
    angle_radians = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians),  np.cos(angle_radians)]
    ])

    # 旋转点并移回质心
    points_rotated = np.dot(points_centered, rotation_matrix.T) + centroid

    return points_rotated

def random_scale(points,lengths,scale=0.5):
    scale = np.random.uniform(1-scale,1+scale,1)
    return points*scale, lengths*scale

def random_shift(points, width, height, scale=0.5):
    
    delta_x = np.random.uniform(-scale, scale,1) * width
    delta_y = np.random.uniform(-scale, scale,1) * height
    points[:,:,0] += delta_x
    points[:,:,1] += delta_y
    return points

def random_delete(coords, lens, labels, ctypes, neighbors):
    
    uni_labels = np.unique(labels, axis=0)
    del_id = np.random.randint(uni_labels.shape[0])
    sem, ins = uni_labels[del_id]
    ind = np.logical_and(labels[:,0]==sem, labels[:,1]==ins)
    valid = np.where(ind==True)[0]
    
    coords = np.delete(coords, valid, axis=0)
    lens = np.delete(lens, valid, axis=0)
    labels = np.delete(labels, valid, axis=0)
    ctypes = np.delete(ctypes, valid, axis=0)
    neighbors = np.delete(neighbors, valid, axis=0)
    
    neighbors[np.isin(neighbors, valid)] = -1
    
    return coords, lens, labels, ctypes, neighbors
    
    
    
    