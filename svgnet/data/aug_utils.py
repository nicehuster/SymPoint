

import numpy as np
import math
import random

def RandomHorizonFilp(args,width=140):
    
    args[:,0::2] = width - args[:,0::2]
    return args

def RandomVerticalFilp(args,Hight=140):
    
    args[:,1::2] = Hight - args[:,1::2]
    return args

def rotate_xy(args,width,height,angle):
    Pi_angle = angle * math.pi / 180.0
    a,b=width/2, height/2
    x0, y0 = args[:,::2], args[:,1::2]
    X0 = (x0-a) * math.cos(Pi_angle) - (y0-b) * math.sin(Pi_angle) + a
    Y0 = (x0-a) * math.sin(Pi_angle) + (y0-b) * math.cos(Pi_angle) + b
    
    return np.concatenate([X0,Y0],axis=1)

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