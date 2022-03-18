import numpy as np
import torch
import os
from torch.utils.data import DataLoader
import time
from sklearn import metrics
import sys
from dataloader import AirDatasetTest

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
version = 'v1.3'
model_path = "epoch200_normal.pth"
model = torch.load(model_path,map_location=device)
model.eval()
outputpath = '/output_save/{}/'.format(version)
month = '01'
MAE_LOSS = torch.nn.L1Loss()
MSE_LOSS = torch.nn.MSELoss()
def REL_L2Loss(pred,truth):
    loss = MSE_LOSS(pred.reshape(-1),truth.reshape(-1))
    rel =MSE_LOSS(truth.reshape(-1), torch.zeros_like(truth.reshape(-1)))
    return loss/(rel+1e-32)
ds = AirDatasetTest()
for hour_index in range(24):
    batch_x1, label , tag= ds.__getitem__(hour_index)
    batch_x1 = torch.FloatTensor(batch_x1)
    label = torch.FloatTensor(label)
    batch_x1 = batch_x1.unsqueeze(0).float()
    print(batch_x1.shape)
    batch_x1[:, 16:32] = 0.1 * torch.log(1e-2 + 1e2 * batch_x1[:, 16:32])
    label = label.unsqueeze(0).float()  # 16 512 512
    output = model(batch_x1).squeeze() #  16 512 512
    loss = REL_L2Loss(output, label)
    output_arr = output.detach().numpy()
    label_arr = label.detach().numpy()

    np.save(outputpath+version + "P" + str(tag) + ".npy", output_arr)
    np.save(outputpath+version + "T" + str(tag) + ".npy", label_arr)
    print(tag)
    print('r2',metrics.r2_score(output_arr.reshape(-1),label_arr.reshape(-1)))

