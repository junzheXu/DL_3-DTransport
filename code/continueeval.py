 # -*- coding: utf-8 -*
import numpy as np
import torch
import os 
from torch.utils.data import Dataset, DataLoader
import random
from sklearn import metrics


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

PATH_PREFIX = "/lustre/home/acct-esehazenet/hazenet-pg1/phydata/cmaq_train_phyonly_0928_noscale/"
def load_data_17(date_index, time_index):
    X_DIR = PATH_PREFIX + "X/"
    Y_DIR = PATH_PREFIX + "Y/"
    X_2D_DIR = PATH_PREFIX + "KZ/"
    filename = "{}_CO_{}.npy".format(date_index,str(time_index))
    x_get = np.load(X_DIR+filename)
    y_get = np.load(Y_DIR+filename)
    filename_2d = "{}_{}.npy".format(date_index,str(time_index))
    x_get2d = np.load(X_2D_DIR+filename_2d)
    con_emis = x_get[:2, :].reshape(32, 512, 512)
    uw = x_get[5, :]
    vw = x_get[6, :]
    kz = x_get2d
    x_get = np.concatenate([con_emis, uw, vw, kz], 0)
    print(x_get.shape)
    return torch.from_numpy(x_get), torch.from_numpy(y_get.reshape(16,512,512))

def load_data_16(date_index, time_index):
    X_DIR = PATH_PREFIX + "eval/X/"
    Y_DIR = PATH_PREFIX + "eval/Y/"
    X_2D_DIR = PATH_PREFIX + "eval/KZ/"
    filename = "{}_CO_{}.npy".format(date_index,str(time_index))
    x_get = np.load(X_DIR+filename)
    y_get = np.load(Y_DIR+filename)
    filename_2d = "{}_{}.npy".format(date_index,str(time_index))
    x_get2d = np.load(X_2D_DIR+filename_2d)
    con_emis = x_get[:2, :].reshape(32, 512, 512)
    uw = x_get[5, :]
    vw = x_get[6, :]
    kz = x_get2d
    x_get = np.concatenate([con_emis, uw, vw, kz], 0)
    print(x_get.shape)
    return torch.from_numpy(x_get), torch.from_numpy(y_get.reshape(16,512,512))
#TODO.1
version='1.3'
modelpath = '/epoch200_normal.pth'
model = torch.load(modelpath,map_location=device)
model.eval()
#TODO.2
outputpath = "/continuesave/v1.3/"

#TODO.3 any date you want
FIRST = True
for date_index in range(20160108,20160132):
    print(date_index)
    for time_index in range(24):
        #TODO.4 one output every 5 min 
        if time_index%1==0: 
            print(time_index)
            if date_index == 20160108:
                x_3d, y = load_data_16(date_index, time_index)
            elif date_index == 20160116:
                x_3d, y = load_data_16(date_index, time_index)
            else:
                x_3d,y = load_data_17(date_index,time_index) #
            batch_x2 = x_3d.to(device).float().unsqueeze(0)
            this_conc = batch_x2[:,:16,:,:].clone()
            if FIRST:
                FIRST=False
                next_conc= this_conc.clone()
            batch_x2[:,:16,:,:] = next_conc.clone()
            batch_y = y.reshape(1,16,512,512).to(device).float()
            #output: scaled change pred
            batch_x2[:, 16:32] = 0.1 * torch.log(1e-2 + 1e2 * batch_x2[:, 16:32])
            output = model(batch_x2).detach()
            #TODO
            print("== ouput max and min ==")
            print(output.max())
            print(output.min())
            print('r2', metrics.r2_score(output.detach().numpy().reshape(-1), batch_y.detach().numpy().reshape(-1)))
            next_conc = output+next_conc
            label_conc = batch_y+this_conc
            next_conc_arr = next_conc.detach().numpy()
            print("== next_conc_arr max and min ==")
            print(next_conc_arr.max())
            print(next_conc_arr.min())
            label_conc_arr = label_conc.detach().numpy()
            print("== label_conc_arr max and min ==")
            print(label_conc_arr.max())
            print(label_conc_arr.min())
            next_conc_arr = next_conc.detach().numpy()
            print("== next_conc_arr_t max and min ==")
            print(next_conc_arr.max())
            print(next_conc_arr.min())
            print('r2', metrics.r2_score(next_conc_arr.reshape(-1), label_conc_arr.reshape(-1)))
            filename ='{}_CO_based_on_{}'.format(date_index,time_index)
            np.save(outputpath+filename+'_predict.npy',next_conc_arr)
            np.save(outputpath+filename+'_truth.npy',label_conc_arr)


        
 
