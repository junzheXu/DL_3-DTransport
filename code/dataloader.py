# -*- coding: utf-8 -*-
"""
U-NET model.
"J.Z. Xu; H. R. Zhang, Z. Cheng, J. Y. Liu, Y. Y. Xu and Y. C. Wang. Approximating Three-dimensional (3-D) Transport of Atmospheric Pollutants via Deep Learning"
"""
from matplotlib.cm import ScalarMappable
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
import random

PATH_PREFIX = "/lustre/home/phydata/"
class AirDatasetTrain(Dataset):
    def __init__(self,X_DIR=PATH_PREFIX+"X/", Y_DIR=PATH_PREFIX+"Y/",X_2D_DIR=PATH_PREFIX+"KZ/"):
        self.xdir = X_DIR
        self.ydir=  Y_DIR
        self.x_2d = X_2D_DIR
        self.path_3d = os.listdir(self.xdir)
        self.y_path = os.listdir(self.ydir)
        self.path_3d = [x for x in self.path_3d if 'CO' in x ]
        self.y_path = [x for x in self.y_path if 'CO'  in x ]
    def __getitem__(self, index):
        filename = self.path_3d[index]
        name_2d = filename[:8] + filename[filename.rfind('_'):]
        x_get = np.load(self.xdir+filename)
        y_get = np.load(self.ydir+filename)
        x_get2d = np.load(self.x_2d+name_2d)
        tag = filename[9:filename.rfind('_')]
        con_emis = x_get[:2,:].reshape(32,512,512)
        uw = x_get[5,:]
        vw = x_get[6,:]
        kz = x_get2d
        x_get = np.concatenate([con_emis,uw,vw,kz], 0)
        print(x_get.shape)
        return x_get, y_get.reshape(16,512,512),tag

    def __len__(self):
        return len(self.path_3d)

class AirDatasetTest(Dataset):
    def __init__(self,X_DIR=PATH_PREFIX+"eval/X/", Y_DIR=PATH_PREFIX+"eval/Y/",X_2D_DIR=PATH_PREFIX+"eval/KZ/"):
        self.xdir = X_DIR
        self.ydir=  Y_DIR
        self.x_2d = X_2D_DIR
        self.path_3d = os.listdir(self.xdir)
        self.y_path = os.listdir(self.ydir)
        self.path_3d = [x for x in self.path_3d if 'CO' in x ]
        self.y_path = [x for x in self.y_path if 'CO'  in x ]
    def __getitem__(self, index):
        filename = self.path_3d[index]
        name_2d = filename[:8] + filename[filename.rfind('_'):]
        x_get = np.load(self.xdir+filename)
        y_get = np.load(self.ydir+filename)
        x_get2d = np.load(self.x_2d+name_2d)
        tag = filename[9:filename.rfind('_')]
        con_emis = x_get[:2,:].reshape(32,512,512)
        uw = x_get[5,:]
        vw = x_get[6,:]
        kz = x_get2d
        x_get = np.concatenate([con_emis,uw,vw,kz], 0)
        return x_get, y_get.reshape(16,512,512),tag
    def __len__(self):
        return len(self.path_3d)