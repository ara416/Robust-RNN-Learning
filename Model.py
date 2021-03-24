import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from Data_Loader import AddGaussianNoise,trainset,trainloader,testset,testloader,BATCH_SIZE

N_STEPS = 28
N_INPUTS = 28
N_NEURONS = 60
N_OUTPUTS = 10
N_EPHOCS = 200

def robustBound(A,B,C,iterm=N_STEPS):
    Q=torch.mm(C.t(),C)
    q2=torch.norm(Q)
    P=torch.zeros(A.size())
    for i in range(2*iterm):
        Ai=torch.matrix_power(A*q2,i)
        P=P+torch.mm(Ai,Ai.t())
    At=torch.matrix_power(A*q2,iterm)
    return torch.trace(torch.mm(torch.mm(B.t(),P),B))+torch.trace(torch.mm(At.t(),At))


def robustBound_Update(A,B,C,iterm=N_STEPS):
    UA, SA, VA = torch.svd(A)
    A2=SA.max()**2
    UB, SB, VB = torch.svd(B)
    B2=SB.max()**2
    UC, SC, VC = torch.svd(C)
    C2=SC.max()**2
    beta=B2
    Bound=0
    for i in range(iterm):
        Bound=A2*(Bound+beta)
    Bound=C2*Bound
    return Bound

class ImageRNN(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs):
        super(ImageRNN, self).__init__()
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons,nonlinearity ='relu')
        self.FC = nn.Linear(self.n_neurons, self.n_outputs)

    def init_hidden(self,):
        # (num_layers, batch_size, n_neurons)
        return (torch.zeros(1, self.batch_size, self.n_neurons))

    def forward(self, X):
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        X = X.permute(1, 0, 2)

        self.batch_size = X.size(1)
        self.hidden = self.init_hidden()

        lstm_out, self.hidden = self.basic_rnn(X, self.hidden)
        out = self.FC(self.hidden)

        return out.view(-1, self.n_outputs) # batch_size X n_output




def accuracymeasure(model,testloader):
    test_acc = 0.0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs = inputs.view(-1, 28, 28)
        outputs = model(inputs)
        test_acc += get_accuracy(outputs, labels, BATCH_SIZE)
    return test_acc/i

def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

# Test and load data sets :P
def Discrete_Lyap(A,Q,iterm=30):
    P=torch.zeros(A.size())
    for i in range(iterm):
        Ai=torch.matrix_power(A,i)
        P=P+torch.mm(Ai,torch.mm(Q,Ai.t()))
    return P

def perfmeasure(model,testloader,Noiseloader):
    perf = 0.0
    outT=torch.empty(0,10)
    outN=torch.empty(0,10)
    for i, data in enumerate(testloader, 0):
        inpT, lbt = data
        inpT = inpT.view(-1, 28, 28)
        outT = torch.cat((outT,model(inpT)),0)
    for i, data in enumerate(Noiseloader, 0):
        inpN, lbN = data
        inpN = inpN.view(-1, 28, 28)
        outN = torch.cat((outN,model(inpN)),0)

    perf=((outN-outT)**2).sum()/outT.size()[0]
    return (perf.data).cpu().numpy()


def perfbound(model):
    A=model.basic_rnn.weight_hh_l0.data
    B=model.basic_rnn.weight_ih_l0.data
    C=model.FC.weight.data
    Q=torch.mm(C.t(),C)
    P=Discrete_Lyap(A,Q)
    perfb=torch.trace(torch.mm(torch.mm(B.t(),P),B))
    At=torch.matrix_power(A,2*iterm)
    return perfb+torch.trace(torch.mm(torch.mm(At.t(),Q),At))

def heaviside(data):
    """
    A `heaviside step function <https://en.wikipedia.org/wiki/Heaviside_step_function>`_
    that truncates numbers <= 0 to 0 and everything else to 1.
    """
    return torch.where(
        data <= torch.zeros_like(data),
        torch.zeros_like(data),
        torch.ones_like(data),)


def SigPr(x):
    return heaviside(x)
#     return (1-torch.tanh(x)**2)
def Sigmoid(x):
    return torch.clamp(x,min=0)
#     return torch.tanh(x)

def EKF_Perf_Aprx(A,B,b,C,N_NEURONS,N_OUTPUTS,inputs):
    TrR=0
    Zigma=torch.eye(N_INPUTS)
    LEN=5
    for j in np.random.randint(inputs.size()[0],size=LEN):
        ht=torch.zeros(1,N_NEURONS)
        Pt=torch.zeros(N_NEURONS, N_NEURONS)
        Rt=torch.zeros(N_OUTPUTS,N_OUTPUTS)
        for i in range(N_STEPS):
            ht= Sigmoid(torch.mm(ht,A.t())+torch.mm(inputs[j,i,:].unsqueeze(0),B.t())+b)
            Dfx=torch.mm(torch.diag(SigPr(ht.squeeze(0))),A)
            Dfu=torch.mm(torch.diag(SigPr(ht.squeeze(0))),B)
            Pt=torch.mm(torch.mm(Dfx,Pt),Dfx.t())+torch.mm(torch.mm(Dfu,Zigma),Dfu.t())
        RT=torch.mm(torch.mm(C,Pt),C.t())
#         TrR=TrR-torch.log(torch.det(RT)+.1)
        TrR=TrR+torch.trace(RT)
    return TrR/LEN
#     ht=torch.zeros(inputs.size()[0], N_NEURONS)
#     Pt=torch.zeros(inputs.size()[0],N_NEURONS, N_NEURONS)
#     Rt=torch.zeros(inputs.size()[0],N_OUTPUTS,N_OUTPUTS)
#     TrR=torch.zeros(inputs.size()[0])
#     Zigma=torch.eye(N_INPUTS)
#     for i in range(N_STEPS):
#         ht= Sigmoid(torch.mm(ht,A.t())+torch.mm(inputs[:,i,:],B.t())+b)
#         for j in range(inputs.size()[0]):
#             Dfx=torch.mm(torch.diag(SigPr(ht[j,:])),A)
#             Dfu=torch.mm(torch.diag(SigPr(ht[j,:])),B)
#             Pt[j,:,:]=torch.mm(torch.mm(Dfx,Pt[j,:,:]),Dfx.t())+torch.mm(torch.mm(Dfu,Zigma),Dfu.t())
#             Rt[j,:,:]=torch.mm(torch.mm(C,Pt[j,:,:]),C.t())
#             TrR[j]=torch.trace(Rt[j,:,:])
#     return TrR.sum()/inputs.size()[0]
