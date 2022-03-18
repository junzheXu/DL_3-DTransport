# -*- coding: utf-8 -*-
'''
x3d conc+meic+phy3d
x2d phy2d(9)
y conc2-conc1
used for trainset
'''
#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
import os

outputpath = '/pydata/'
mcippath = '/mcip_data/'
datelist = ['08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
month = '01'
timestep = 24
MET_2D = ['PBL','RN', 'RC','MOLI','RGRND','VEG', 'LAI','RADYNI', 'RSTOMI','WSTAR']#[288,1,512,512]
MET_3D = ['TA', 'QV', 'PRES','UWINDC', 'VWINDC','DENS','DENSA_J','QC','QR']
spckeylist = ['CO']
for sss in range(len(datelist)-1):
    #3d
    print(datelist[sss])
    #environment
    np_envfile = '0'
    for key in MET_3D:
        print('met3dkey:',key)
        if np_envfile=='0':
            np_envfile=np.load(mcippath+'2016'+month+datelist[sss]+key+'.npy')[:-1,:,:,:].reshape(timestep,1,16,512,512)#(288,16,512,512)
        else:
            np_envkey = np.load(mcippath+'2016'+month+datelist[sss]+key+'.npy')[:-1,:,:,:].reshape(timestep,1,16,512,512)
            np_envfile = np.append(np_envfile, np_envkey, axis=1)
    print("np_envfile.shape={}".format(np_envfile.shape))
    del np_envkey
    #cctm conc
    cctmfile = Dataset('cmaq_data/' + datelist[sss] + '.nc')
    cctmfile_y = Dataset('cmaq_data/' + datelist[sss+1] + '.nc')
    for key in spckeylist:
        meicfile = Dataset('/meicpath/' + datelist[sss])
        key_file = meicfile[key]
        np_key_file = np.array(key_file)[:-1,:,:,:]
        np_meicfile = np_key_file.reshape(timestep,1,16,512,512)
        print("np_meicfile.shape={}".format(np_meicfile.shape))
        np_cctmfile=np.array(cctmfile[key]).reshape(timestep,1,16,512,512)
        print('cctmfile:{}'.format(np_cctmfile.shape))
        # 卡点

        np_cctmfile = np.append(np_cctmfile, np_meicfile, axis=1)
        np_cctmfile = np.append(np_cctmfile, np_envfile, axis=1)
        print('np_cctmfile.shape={}'.format(np_cctmfile.shape))
        np_cctmfile = np_cctmfile[-1,:,:,:,:]
        print('np_cctmfile_after.shape={}'.format(np_cctmfile.shape))

        py1_path = outputpath+'X/'
        if not os.path.exists(py1_path):
            os.makedirs(py1_path)

        np.save(py1_path+'2016'+month+datelist[sss]+'_'+key,np_cctmfile)
        del  np_meicfile

        np_cctmfile2=np.array(cctmfile_y[key][0,:,:,:])
        print('np_cctmfile2.shape={}'.format(np_cctmfile2.shape))

        py2_path = outputpath+'Y/'
        if not os.path.exists(py2_path):
            os.makedirs(py2_path)

        np.save(py2_path+'2016'+month+datelist[sss]+'_'+key,np_cctmfile2)
        del np_cctmfile2,np_cctmfile
    del cctmfile,np_envfile
    #2d
    np_envfile_2d = '0'
    for key in MET_2D:
        print('metkey2d:',key)
        filepath = mcippath+'2016'+month+datelist[sss]+key+'.npy'
        print('env2dpath:',filepath)
        if np_envfile_2d=='0':
            np_envfile_2d =np.load(filepath)[:-1,:,:,:].reshape(timestep,1,512,512)
        else:
            np_envkey_2d =np.load(filepath)[:-1,:,:,:].reshape(timestep,1,512,512)
            np_envfile_2d = np.append(np_envfile_2d, np_envkey_2d, axis=1)
    del np_envkey_2d
    print("np_envfile_2d.shape={}".format(np_envfile_2d.shape))
    np_envfile_2d =np_envfile_2d[-1,:,:,:]
    print("np_envfile_2d_after.shape={}".format(np_envfile_2d.shape))
    phy_path = outputpath+'2D/'
    if not os.path.exists(phy_path):
        os.makedirs(phy_path)
    
    np.save(phy_path+'2016'+month+datelist[sss],np_envfile_2d)
    del np_envfile_2d