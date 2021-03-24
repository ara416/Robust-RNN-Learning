import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from Data_Loader import AddGaussianNoise,trainset,trainloader,testset,testloader,BATCH_SIZE
from Model import *
import pickle

torch.manual_seed(0)
np.random.seed(0)
# Code name
# U :unstable
# S : Stable
# RU : uperbound
# REKF : Robust estimation 


savename='REKF1.pt'
# savename='Unstable_Relu_model.pt'
#######################################################
# Training
######################################################
# Device selection =======================================================================
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print('-' * 89)
print(device)
print('-' * 89) 
lr=0.001
# Model instance =======================================================================
model = ImageRNN(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr)#, weight_decay=1e-03)
ListAcc=0
# Training =======================================================================
# torch.autograd.set_detect_anomaly(True)
# params
Tr_acc=0
Te_acc=0

# ==============================================



best_val_loss = None
for epoch in range(N_EPHOCS):  # loop over the dataset multiple times
    train_running_loss = 0.0
    train_acc = 0.0
    model.train()    
    # TRAINING ROUND
    for i, data in enumerate(trainloader):
         # zero the parameter gradients
        optimizer.zero_grad()
        # reset hidden states
        model.hidden = model.init_hidden() 
        # get the inputs
        inputs, labels = data
        inputs = inputs.view(-1, 28,28) 
        # forward + backward + optimize 
        outputs = model(inputs)
#         PerfBound=robustBound(model.basic_rnn.weight_hh_l0,model.basic_rnn.weight_ih_l0,model.FC.weight)
# 
#         PerfBound=robustBound_Update(model.basic_rnn.weight_hh_l0,model.basic_rnn.weight_ih_l0,model.FC.weight)
#         
        Perf_APX=EKF_Perf_Aprx(model.basic_rnn.weight_hh_l0,model.basic_rnn.weight_ih_l0,
                               model.basic_rnn.bias_hh_l0+model.basic_rnn.bias_ih_l0,
                               model.FC.weight,
                               N_NEURONS,N_OUTPUTS,inputs)
        # ===================================== Objective function ======================================
#         loss = criterion(outputs, labels)+.015*PerfBound/(epoch+1)
        loss = criterion(outputs, labels)+.01*Perf_APX
        loss.backward()
        optimizer.step()
#          ======================= Stability conditions
#         with torch.no_grad():
#             Whh=model.basic_rnn.weight_hh_l0.data
#             U, S, V = torch.svd(Whh)
#             SN=torch.clamp(S,max=.99)
#             model.basic_rnn.weight_hh_l0.data=torch.mm(torch.mm(U,torch.diag(SN)),V.t())
        # ======================== Train Accuracy measure 
        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(outputs, labels, BATCH_SIZE)
    model.eval()
    with torch.no_grad():
        A=(model.basic_rnn.weight_hh_l0.data).cpu().numpy()
        test_acc=accuracymeasure(model,testloader)
        ListAcc=np.append(ListAcc,test_acc)    
        val_loss= 100-test_acc
        if not best_val_loss or val_loss < best_val_loss:
                with open(savename, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
#         else:
#             lr=lr/2
            
    print('-' * 89)
    print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' 
          %(epoch, train_running_loss / i, train_acc/i),lr)
    print('=======> norm A',np.linalg.norm(A,2))
#     print('Perfbound ====>',PerfBound.data)
    print('Perf_Apx ====>',Perf_APX)
    print('Test Accuracy: %.2f'%(test_acc))
    Tr_acc=np.append(Tr_acc,train_acc/i)
    Te_acc=np.append(Te_acc,test_acc)
f = open('REKF.pckl', 'wb')
pickle.dump([Tr_acc,Te_acc], f)
f.close()