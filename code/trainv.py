from dataloader import *
from unet import *
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
import time


version = 'v1.3'
device='cpu'
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Running on GPU')

model = UNet(80,16).to(device)
MAE_LOSS = torch.nn.L1Loss()
MSE_LOSS = torch.nn.MSELoss()
train_dataset = AirDatasetTrain()
train_loader=DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

test_dataset = AirDatasetTest()
test_loader=DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

lossnp = np.array([])
best_loss = float('inf')
loss_target = float('inf')
BATCH_SIZE = 4
EPOCH = 200
SEVERAL = 10

modelpath = '/modelsave/%s/'%(version)
if not os.path.exists(modelpath):
    os.mkdir(modelpath)

def REL_L2Loss(pred,truth):
    loss = MSE_LOSS(pred.reshape(-1),truth.reshape(-1))
    rel =MSE_LOSS(truth.reshape(-1), torch.zeros_like(truth.reshape(-1)))
    return loss/(rel+1e-32)
month = '01'
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

for epoch in range(EPOCH):
    model.train()
    loss_sum = 0
    loss_sum_test=0
    print('starting...')
    model.train()
    for batch, (batch_x1, batch_y1,tag) in enumerate(train_loader):
        optimizer.zero_grad()
        stime=time.time()
        s=512
        batch_x1=batch_x1.float().to(device).contiguous() # BS 10 512 512
        batch_x1[:,16:32]=0.1*torch.log(1e-2+1e2*batch_x1[:,16:32])
        label1=batch_y1.float().to(device)# BS 1 512 512
        output = model(batch_x1)
        mae_train = MAE_LOSS(output,label1)
        mse_train = MSE_LOSS(output,label1)
        loss = REL_L2Loss(output,label1)
        loss.backward()
        optimizer.step()
        loss_batch = loss.item()
        loss_sum += loss_batch
        etime = time.time()
        f=open('BATCH_LOSS_'+version+'.txt','a+')
        f.write('Epoch: '+ str(epoch)+ ' | Batch: '+ str(batch)+ '\n')
        f.write('MAE_BATCH:  %.8f ' % (mae_train) )
        f.write('MSE_BATCH:  %.8f ' % (mse_train) )
        f.write('LOSS_BATCH_AVE:  %.8f ' % (loss_batch) )
        f.close()
        del  batch_x1, batch_y1, output, loss
    name = modelpath + 'epoch%s_normal' % (epoch) + time.strftime('%m%d_%H:%M:%S.pth')
    torch.save(model, name)
    scheduler.step()
    model.eval()

    for batch, (batch_x1, batch_y1,tag) in enumerate(test_loader):
        s=512
        batch_x1=batch_x1.float().to(device).contiguous() # BS 10 512 512
        batch_x1[:,16:32]=0.1*torch.log(1e-2+1e2*batch_x1[:,16:32])
        label1=batch_y1.float().to(device)# BS 1 512 512
        output = model(batch_x1)
        loss = REL_L2Loss(output,label1)
        loss_batch = loss.item()
        loss_sum_test += loss_batch

        del  batch_x1, batch_y1, output, loss
    f=open('BATCH_LOSS_'+version+'.txt','a+')
    f.write('Epoch: '+ str(epoch) )
    f.write('\nREL_Loss in Train :  %.8f ' % (loss_sum) )
    f.write('\nREL_Loss in Test :  %.8f ' % (loss_sum_test))
    f.close()
    name = modelpath + 'epoch%s_normal' % (epoch) + time.strftime('%m%d_%H:%M:%S.pth')
    torch.save(model, name)
