# -*- coding: utf-8 -*-
"""
Training code for "Low-complexity end-to-end deep learning framework for 100G-PON"

Process:
TxBits--PAM4 Mod--TxNN--------Physical Channel----------RxNN--PAM4 DeMod--RxBits
                        |                            |  
                        |--Noise Adaptation Channel--| 
                        
@author: Yongxin Xu, xuyongxin@sjtu.edu.cn
"""

from torch.autograd import Variable
import datetime
from torch.optim import Adam,AdamW
import torch.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
from torch.distributions import Normal
from TruncatedNormal import TruncatedNormal
from collections import OrderedDict
from math import sqrt,log2,log10,pi,ceil
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from copy import deepcopy
from utils import bit2sym,sym2bit,ber,ser,PAM4_MOD,PAM4_DEMOD,PAM4_Decoder,rcosdesign,to_tensor,to_numpy
import torchaudio
import scipy.io as sio
import scipy.fftpack as fftpack
import scipy.signal as scisig
import scipy.fft as scifft
from memory import MemoryBuffer

import matlab
import matlab.engine
eng = matlab.engine.start_matlab()

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)

### TxNN: (1)NNPDLUT
class TxNN_main(nn.Module):
    def __init__(self):
        super(TxNN_main, self).__init__()
        self.in_dim = 1
        self.out_dim = 1
        self.seqLen = R['TxNN_seq_len_LUT']
        self.fc0 = nn.Linear(R['M']**R['TxNN_seq_len_LUT'],64)
        self.fc1 = nn.Linear(64,32)
        self.fc2 = nn.Linear(32,32)
        self.fc6 = nn.Linear(32+64,self.out_dim)
        self.act = nn.ELU()

    def forward(self, x):  
        xx = self.fc0(x)
        out = self.act(self.fc1(xx))
        out = self.act(self.fc2(out))
        out = torch.cat([out,xx],1)
        out = self.fc6(out)
        return out

class TxNN(nn.Module):
    def __init__(self):
        super(TxNN, self).__init__()
        self.TxNN_main = TxNN_main()

    def forward(self, x):  
        out = self.TxNN_main(x).reshape(-1,1)
        return out 


### TxNN: (2)Conv layer
class TxNN_FFE_Base(nn.Module):
    def __init__(self):
        super(TxNN_FFE_Base, self).__init__()
        self.in_dim = 1
        self.out_dim = 2  # sps
        self.seqLen = R['TxNN_seq_len_FFE']
        self.conv1 = nn.Conv1d(self.in_dim,self.out_dim,self.seqLen,bias=True)
    def forward(self, x):  
        xx = self.conv1(x)
        out = xx.reshape(-1,self.out_dim)
        return out

class TxNN_FFE(nn.Module):
    def __init__(self):
        super(TxNN_FFE, self).__init__()
        self.TxNN_FFE_Base = TxNN_FFE_Base()
        self.act = nn.Tanh()
        self.batchnorm = nn.BatchNorm1d(1,affine=False)

    def forward(self, x):  
        out = self.TxNN_FFE_Base(x).reshape(-1,1)
        out = self.act(out)
        out = self.batchnorm(out.reshape(-1,1))
        return out 


### RxNN: Conv layer
class RxNN_FFE(nn.Module):
    def __init__(self):
        super(RxNN_FFE, self).__init__()
        self.in_dim = R['RxNN_input_dim']   # sps
        self.out_dim = 1
        self.conv1 = nn.Conv1d(self.in_dim,1,R['RxNN_seq_len'],bias=True) 
    def forward(self, x):  
        xx = self.conv1(x)
        out = xx.reshape(-1,self.out_dim)
        return out


### Noise Adaptation Network 
## mean network based on MscaleDNN
class ChannelNN_Mean_Base(nn.Module):
    def __init__(self):
        super(ChannelNN_Mean_Base, self).__init__()
        self.in_dim = R['ChanNN_input_dim']
        self.out_dim = R['ChanNN_output_dim']
        self.conv1 = nn.Conv1d(self.in_dim,128,kernel_size=R['Chan_seq_len'])
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,32)
        self.conv2 = nn.Conv1d(32,self.out_dim,kernel_size=1)
        self.act = nn.LeakyReLU()
        # self.act = phi2()

    def forward(self, x):  
        xx = self.conv1(x).reshape(-1,128)
        out = self.act(self.fc1(xx))
        out = self.act(self.fc2(out)).reshape(-1,32,1)
        out = self.conv2(out).reshape(-1,self.out_dim)
        return out

class ChannelNN_Mean(nn.Module):
    def __init__(self):
        super(ChannelNN_Mean,self).__init__()
        self.scale = R['ChanNN_scale']
        self.output_dim = R['ChanNN_output_dim']
        self.subNets = nn.ModuleList([ChannelNN_Mean_Base() for _ in range(len(self.scale))])
        self.subNets.append(nn.Sequential(OrderedDict([('linear_Final',nn.Linear(len(self.scale),1))])))

    def forward(self, x):   
        subNet_output = []
        for k in range(len(self.scale)):
            scale_x = self.scale[k] * x
            subNet_output.append(self.subNets[k](scale_x))
        output = torch.stack(subNet_output,dim=0)
        output = output.permute(1,2,0)
        output = output.reshape(-1,len(self.scale))
        output = self.subNets[len(self.scale)](output).reshape(-1,self.output_dim)     
        return output 
    
## variation network based on MscaleDNN   
class ChannelNN_Var_Base(nn.Module):
    def __init__(self):
        super(ChannelNN_Var_Base, self).__init__()
        self.in_dim = R['ChanNN_input_dim']
        self.out_dim = R['ChanNN_output_dim']
        self.conv1 = nn.Conv1d(self.in_dim,128,kernel_size=R['Chan_seq_len']) #shape=[batchsize,1,hidden1_dim]
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,32)
        self.conv2 = nn.Conv1d(32,self.out_dim,kernel_size=1)
        self.act = nn.LeakyReLU()
        # self.act = phi2()

    def forward(self, x):  
        xx = self.conv1(x).reshape(-1,128)
        out = self.act(self.fc1(xx))
        out = self.act(self.fc2(out)).reshape(-1,32,1)
        out = self.conv2(out).reshape(-1,self.out_dim)
        return out

class ChannelNN_Var(nn.Module): 
    def __init__(self):
        super(ChannelNN_Var,self).__init__()
        self.scale = R['ChanNN_scale']
        self.output_dim = R['ChanNN_output_dim']
        self.subNets = nn.ModuleList([ChannelNN_Mean_Base() for _ in range(len(self.scale))])
        self.subNets.append(nn.Sequential(OrderedDict([('linear_Final',nn.Linear(len(self.scale),1))])))

    def forward(self, x):   
        # Define the output as the logarithm of variance, and the true variance is equal to exp(output)
        subNet_output = []
        for k in range(len(self.scale)):
            scale_x = self.scale[k] * x
            subNet_output.append(self.subNets[k](scale_x))
        output = torch.stack(subNet_output,dim=0)
        output = output.permute(1,2,0)
        output = output.reshape(-1,len(self.scale))
        output = self.subNets[len(self.scale)](output).reshape(-1,self.output_dim)     
        return output 

class ChannelNN(nn.Module):
    def __init__(self):
        super(ChannelNN, self).__init__()
        self.in_dim = R['ChanNN_input_dim']
        self.out_dim = R['ChanNN_output_dim']
        self.ChannelNN_Mean = ChannelNN_Mean()
        self.ChannelNN_Var = ChannelNN_Var()   

    def forward(self, x):  
        out_mean = self.ChannelNN_Mean(x)
        out_rho = self.ChannelNN_Var(x)
        return out_mean,out_rho  


### Pattern-dependent lookup table
def base2dec(s, base):
    """
    Conversion from any base to decimal.
    Convert a list 's' of symbols to a decimal number
    (most significant symbol first)
    """
    num = 0
    for i in range(len(s)):
        num = num * base + s[i]
    return num

def LUTIndexMake(txdataIn, L, M):
    window = L
    slide = 1
    # data process
    txdataIn = txdataIn.reshape(-1)
    col = window
    row = (len(txdataIn)-window)//slide+1
    idx = np.arange(col)[None,:] + slide*np.arange(row)[:,None]
    txdataIn = txdataIn[idx]
    txdataIn = txdataIn.reshape(-1,window)
    num = 0
    for i in range(txdataIn.shape[1]):
        num = num * M + txdataIn[:,i]
    num = num.long()
    OnehotVector = torch.nn.functional.one_hot(num,num_classes=M**L)
    OnehotVector  = OnehotVector.float()
    return OnehotVector


### Create input for TxNN and RxNN 
def createTxRxInput(txSignal,seq_len,feature):
    window = seq_len * feature
    slide = feature
    # data process
    txSignal = txSignal.reshape(-1)
    col = window
    row = (len(txSignal)-window)//slide+1
    idx = np.arange(col)[None,:] + slide*np.arange(row)[:,None]
    txSignal = txSignal[idx]
    txSignal = txSignal.reshape(-1,seq_len,feature)  
    data = txSignal
    return data


### Create input for ChannelNN
def createChanInput(txSignal):
    seq_len = R['Chan_seq_len']
    feature = R['Chan_feature_dim'] 
    window = seq_len * feature
    slide = feature
    # data process
    txSignal = txSignal.reshape(-1)
    col = window
    row = (len(txSignal)-window)//slide+1
    idx = np.arange(col)[None,:] + slide*np.arange(row)[:,None]
    txSignal = txSignal[idx]
    txSignal = txSignal.reshape(-1,seq_len,feature)  
    data = txSignal
    return data


### Create training dataset for ChannelNN
def createDataset(txSignal,rxSignal):
    seq_len = R['Chan_seq_len']
    feature = R['Chan_feature_dim'] 
    window = seq_len * feature
    slide = feature
    # data process
    txSignal = txSignal.reshape(-1)
    col = window
    row = (len(txSignal)-window)//slide+1
    idx = np.arange(col)[None,:] + slide*np.arange(row)[:,None]
    txSignal = txSignal[idx]
    txSignal = txSignal.reshape(-1,seq_len,feature)  
    data = txSignal
    # label process
    start_id = (seq_len-1)//2 * feature
    end_id = start_id + data.shape[0] * feature
    label = rxSignal[start_id:end_id].reshape(-1,R['sps'])
    return data,label


### Physical Channel: AWG+MZM+fiber+APD+OSC
def Channel(txSamples):
    # The code is an example, you should define the channel function according to your experiment
    PAPR_dB,rxSamples = MZM_ADD_Channel(txSamples, R['SymRate'], R['sps'])    
    return PAPR_dB,rxSamples


### Train Channel model
def train_Channel(ChannelNet,data,label,ChanOptimizer):   
    ChannelNet.train()
    data = Variable(data)
    label = Variable(label) 
    updateSteps = data.shape[0]//R['batchsize']+1
    for i in range(updateSteps):
        ChanOptimizer.zero_grad()
        mask = np.random.choice(data.shape[0],R['batchsize'],replace=False)
        output_mean,output_rho = ChannelNet(data[mask].permute(0,2,1))
        output_mean = output_mean.reshape(-1)
        output_rho = output_rho.reshape(-1)        
        output_var = torch.exp(output_rho)
        nll_loss = torch.log(2 * pi * output_var) \
              + (label[mask].reshape(-1) - output_mean) ** 2 / output_var
        nll_loss = torch.mean(nll_loss)
        nll_loss.backward()
        ChanOptimizer.step() 
    # print('Channel Model Training Epoch = %d, train loss = %.8f'%(epoch,loss.item()))
    return nll_loss.item()           

### validate (or test) channel model
def valid_Channel(ChannelNet,data,label):   
    ChannelNet.eval()
    data = Variable(data)
    label = Variable(label) 
    output_mean,output_rho = ChannelNet(data.permute(0,2,1))
    output_mean = output_mean.reshape(-1)
    output_rho = output_rho.reshape(-1)        
    output_var = torch.exp(output_rho)
    nll_loss = torch.log(2 * pi * output_var) \
          + (label.reshape(-1) - output_mean) ** 2 / output_var
    nll_loss = torch.mean(nll_loss)
    # print('Channel Model Valid Epoch = %d, valid loss = %.8f'%(epoch,loss.item()))
    return nll_loss.item()        


# Train TxNN and RxNN
def train_Transceiver(TxNet,TxNetFFE,RxNet,ChannelNet,txdataIn,TxOptimizer,TxFFEOptimizer,RxOptimizer):
    TxNet.train()
    TxNetFFE.train()
    RxNet.train()
    ChannelNet.eval()

    TxOptimizer.zero_grad()
    TxFFEOptimizer.zero_grad()
    RxOptimizer.zero_grad()

    OnehotVector = LUTIndexMake(txdataIn, L=R['TxNN_seq_len_LUT'], M=R['M'])
    deltaTx = TxNet(OnehotVector.to(device)).reshape(-1)
    
    # generate txSymbols
    txSymbols = to_tensor(PAM4_MOD(txdataIn,R['constel'])).reshape(-1)
    
    # NN-PDLUT
    txPreSymbols = txSymbols[(R['TxNN_seq_len_LUT']-1)//2:-(R['TxNN_seq_len_LUT']-1)//2] - deltaTx
    # txPreSymbols = txPreSymbols / torch.sqrt(torch.mean(torch.pow(torch.abs(txPreSymbols),2)))  
    
    # TxFFE
    Input_TxNetFFE = createTxRxInput(txPreSymbols.reshape(-1),seq_len=R['TxNN_seq_len_FFE'],feature=1)
    txPreSamples = TxNetFFE(Input_TxNetFFE.permute(0,2,1)).reshape(-1)
    
    # label
    start_id = (R['TxNN_seq_len_LUT']-1)//2 + (R['TxNN_seq_len_FFE']-1)//2 + (R['Chan_seq_len']-1)//2 + (R['RxNN_seq_len']-1)//2
    label = Variable(txSymbols)
    label = label[start_id:-start_id]
    start_id = (R['TxNN_seq_len_LUT']-1)//2 + (R['TxNN_seq_len_FFE']-1)//2 + (R['Chan_seq_len']-1)//2
    label1 = Variable(txSymbols)
    label1 = label1[start_id:-start_id]
    
    # transmission over ChannelNN
    Input_ChannelNet = createChanInput(txPreSamples.reshape(-1))
    rxSamples_mean,rxSamples_rho = ChannelNet(Input_ChannelNet.permute(0,2,1))
    rxSamples_mean = rxSamples_mean.reshape(-1)
    rxSamples_rho = rxSamples_rho.reshape(-1)        
    rxSamples_std = torch.sqrt(torch.exp(rxSamples_rho))   

    # (1) Use only mean values
    # rxSamples = rxSamples_mean

    # (2) Truncated Normal Distribution
    loc = rxSamples_mean
    scale = rxSamples_std
    lower = (rxSamples_mean - R['alpha']*rxSamples_std).clone().detach()
    higher = (rxSamples_mean + R['alpha']*rxSamples_std).clone().detach()
    rxSamples_dist = TruncatedNormal(loc,scale,lower,higher)
    rxSamples = rxSamples_dist.rsample()

    # (3) Gaussian Distribution
    # rxSamples_dist = Normal(rxSamples_mean,rxSamples_std)
    # rxSamples = rxSamples_dist.rsample()
  
    rxSymbols = (rxSamples[0::2] + rxSamples[1::2]) / 2
    rxSymbols = sqrt(R['power_constel']) * rxSymbols / torch.sqrt(torch.mean(torch.pow(torch.abs(rxSymbols),2))) 
                  
    # # Receiver Equalization 2sps
    # rxSamples = sqrt(R['power_constel']) * rxSamples / torch.sqrt(torch.mean(torch.pow(torch.abs(rxSamples),2))) 
    # Input_RxNet = createTxRxInput(rxSamples.reshape(-1),seq_len=R['RxNN_seq_len'],feature=R['RxNN_feature_dim'])

    # Receiver Equalization 1sps
    Input_RxNet = createTxRxInput(rxSymbols.reshape(-1),seq_len=R['RxNN_seq_len'],feature=R['RxNN_feature_dim'])

    rxEquSymbols = RxNet(Input_RxNet.permute(0,2,1))
    # # normalize the power of rxEquSymbols  to 1
    rxEquSymbols = rxEquSymbols  - torch.mean(rxEquSymbols)
    rxEquSymbols = sqrt(R['power_constel']) * rxEquSymbols / torch.sqrt(torch.mean(torch.pow(torch.abs(rxEquSymbols),2)))
    
    # calculate mse loss and uptade TxNN,RxNN 
    loss1 = F.mse_loss(rxEquSymbols.reshape(-1,1),label.reshape(-1,1),reduction='mean')
    loss2 = F.mse_loss(rxSymbols.reshape(-1,1),label1.reshape(-1,1),reduction='mean')
    loss = loss1 + loss2
    loss.backward()
    TxOptimizer.step()
    TxFFEOptimizer.step()
    RxOptimizer.step()
    # print('Transmitter Training Epoch = %d, train loss = %.8f'%(epoch,loss.item()))     
    return loss.item()


### validate TxNN and RxNN
def valid(TxNet,TxNetFFE,RxNet,txdataIn): 
    TxNet.eval()
    TxNetFFE.eval()
    RxNet.eval()  
 
    with torch.no_grad():
        OnehotVector = LUTIndexMake(txdataIn, L=R['TxNN_seq_len_LUT'], M=R['M'])
        deltaTx = TxNet(OnehotVector).reshape(-1)
        
        # generate txSymbols
        txSymbols = to_tensor(PAM4_MOD(txdataIn,R['constel'])).reshape(-1)
        
        # NN-PDLUT
        txPreSymbols = txSymbols[(R['TxNN_seq_len_LUT']-1)//2:-(R['TxNN_seq_len_LUT']-1)//2] - deltaTx
        # txPreSymbols = txPreSymbols / torch.sqrt(torch.mean(torch.pow(torch.abs(txPreSymbols),2)))  
        
        # TxFFE
        Input_TxNetFFE = createTxRxInput(txPreSymbols.reshape(-1),seq_len=R['TxNN_seq_len_FFE'],feature=1)
        txPreSamples = TxNetFFE(Input_TxNetFFE.permute(0,2,1)).reshape(-1)
        
        start_id = (R['TxNN_seq_len_LUT']-1)//2 + (R['TxNN_seq_len_FFE']-1)//2 + (R['RxNN_seq_len']-1)//2
        label = Variable(txSymbols)
        
        # transmission over physical channel
        PAPR_dB,rxSamples = Channel(to_numpy(txPreSamples).reshape(-1))
        # print('PAPR = %.2f dB'%(PAPR))
        rxSamples = to_tensor(rxSamples).float()
        
        # # Receiver Equalization 2sps
        # rxSamples = sqrt(R['power_constel']) * rxSamples / torch.sqrt(torch.mean(torch.pow(torch.abs(rxSamples),2)))        
        # Input_RxNet = createTxRxInput(rxSamples.reshape(-1),seq_len=R['RxNN_seq_len'],feature=R['RxNN_feature_dim'])

        # Receiver Equalization 1sps
        rxSymbols = (rxSamples[0::2] + rxSamples[1::2]) / 2
        rxSymbols = sqrt(R['power_constel']) * rxSymbols / torch.sqrt(torch.mean(torch.pow(torch.abs(rxSymbols),2))) 
        Input_RxNet = createTxRxInput(rxSymbols.reshape(-1),seq_len=R['RxNN_seq_len'],feature=R['RxNN_feature_dim'])

        rxEquSymbols = RxNet(Input_RxNet.permute(0,2,1))
        label = label[start_id:start_id+Input_RxNet.shape[0]]       
        # # normalize the power of rxEquSymbols  to 1
        rxEquSymbols = rxEquSymbols  - torch.mean(rxEquSymbols)
        rxEquSymbols = sqrt(R['power_constel']) * rxEquSymbols / torch.sqrt(torch.mean(torch.abs(rxEquSymbols)**2))

        # PAM4 Decision
        # DeSymbols_tx = PAM4_DEMOD(label.numpy(),R['constel'])
        # DeSymbols_rx = PAM4_DEMOD(rxEquSymbols.numpy(),R['constel'])
        txData,txBits = PAM4_Decoder(label.numpy(),R['constel'],R['k'])
        rxData,rxBits = PAM4_Decoder(rxEquSymbols.numpy(),R['constel'],R['k'])
        valid_BER = ber(txBits,rxBits)   
    return valid_BER,rxEquSymbols


# Create dataset for RxNN
def createRxDataset(tx,rx):
    seq_len = R['RxNN_seq_len']
    feature = R['RxNN_feature_dim'] 
    window = seq_len * feature
    slide = feature

    # data process
    rx = rx.reshape(-1)
    col = window
    row = (len(rx)-window)//slide+1
    idx = np.arange(col)[None,:] + slide*np.arange(row)[:,None]
    rx = rx[idx]
    rx = rx.reshape(-1,seq_len,feature)  
    data = rx
    
    # label process
    start_id = (seq_len-1)//2 * 1
    end_id = start_id + data.shape[0] * 1
    label = tx[start_id:end_id].reshape(-1,1)
    return data,label

### validate TxNN and RxNN, and finetune the RxNN
def valid_withRxTrain(TxNet,TxNetFFE,txdataIn,epoch=50): 
    TxNet.eval()
    TxNetFFE.eval()
    RxNetFFE_forTest = RxNN_FFE()
    RxNetFFE_forTest.to(device)
    RxOptimizer_mse = Adam(RxNetFFE_forTest.parameters(),lr = R['lr_Rx'])   

    with torch.no_grad():
        OnehotVector = LUTIndexMake(txdataIn, L=R['TxNN_seq_len_LUT'], M=R['M'])
        deltaTx = TxNet(OnehotVector.to(device) ).reshape(-1)
        # generate txSymbols
        txSymbols = to_tensor(PAM4_MOD(txdataIn,R['constel'])).reshape(-1)
        # PD-LUT
        txPreSymbols = txSymbols[(R['TxNN_seq_len_LUT']-1)//2:-(R['TxNN_seq_len_LUT']-1)//2] - deltaTx
        # TxFFE
        Input_TxNetFFE = createTxRxInput(txPreSymbols.reshape(-1),seq_len=R['TxNN_seq_len_FFE'],feature=1)
        txPreSamples = TxNetFFE(Input_TxNetFFE.permute(0,2,1)).reshape(-1)
        # transmission over physical channel
        PAPR,rxSamples = Channel(to_numpy(txPreSamples).reshape(-1))
        # print('PAPR = %.2f dB'%(PAPR))
        rxSamples = to_tensor(rxSamples).float()
        rxSamples = rxSamples / torch.sqrt(torch.mean(torch.pow(torch.abs(rxSamples),2))) 
                
        start_id = (R['TxNN_seq_len_LUT']-1)//2 + (R['TxNN_seq_len_FFE']-1)//2
        
        txSymbols_cut = txSymbols[start_id:-start_id]
    
    trainLen = 10000
    Input_RxNet,label_train = createRxDataset(txSymbols_cut[:trainLen],rxSamples[:2*trainLen])
    
    data = Input_RxNet
    label = label_train
    for k in range(epoch):
        updateSteps = data.shape[0]//R['batchsize_finetune']+1
        for i in range(updateSteps):
            RxOptimizer_mse.zero_grad()
            mask = np.random.choice(data.shape[0],R['batchsize_finetune'],replace=False)
            rxEquSymbols = RxNetFFE_forTest(data[mask].permute(0,2,1))
            # calculate mse loss and uptade TxNN,RxNN 
            loss = F.mse_loss(rxEquSymbols.reshape(-1,1),label[mask].reshape(-1,1),reduction='mean')
            loss.backward()
            RxOptimizer_mse.step()
            # print('Transmitter Training Epoch = %d, train loss = %.8f'%(epoch,loss.item()))

    Input_RxNet,label_valid = createRxDataset(txSymbols_cut[trainLen:],rxSamples[2*trainLen:])
    rxEquSymbols = RxNetFFE_forTest(Input_RxNet.permute(0,2,1)).detach()
    
    # # normalize the power of rxEquSymbols  to 1
    rxEquSymbols = rxEquSymbols  - torch.mean(rxEquSymbols)
    rxEquSymbols = rxEquSymbols / torch.sqrt(torch.mean(torch.abs(rxEquSymbols)**2))
    # PAM4 Decision
    txData,txBits = PAM4_Decoder(label_valid.cpu().numpy(),R['constel'],R['k'])
    rxData,rxBits = PAM4_Decoder(rxEquSymbols.cpu().numpy(),R['constel'],R['k'])
    valid_BER = ber(txBits,rxBits)       

    return valid_BER,txPreSymbols.cpu().detach().numpy(),txPreSamples.cpu().detach().numpy(),\
            PAPR,rxSamples.cpu().detach().numpy(),rxEquSymbols.cpu().detach().numpy(),label_test.cpu().numpy()


# Finetune Receiver
def train_Receiver(TxNet,TxNetFFE,RxNet,txdataIn,RxOptimizer):
    TxNet.eval()
    TxNetFFE.eval()
    RxNet.train()
    
    with torch.no_grad():
        OnehotVector = LUTIndexMake(txdataIn, L=R['TxNN_seq_len_LUT'], M=R['M'])
        deltaTx = TxNet(OnehotVector).reshape(-1)
        
        # generate txSymbols
        txSymbols = to_tensor(PAM4_MOD(txdataIn,R['constel'])).reshape(-1)
        
        # PD-LUT
        txPreSymbols = txSymbols[(R['TxNN_seq_len_LUT']-1)//2:-(R['TxNN_seq_len_LUT']-1)//2] - deltaTx
        # txPreSymbols = txPreSymbols / torch.sqrt(torch.mean(torch.pow(torch.abs(txPreSymbols),2)))  
        
        # TxFFE
        Input_TxNetFFE = createTxRxInput(txPreSymbols.reshape(-1),seq_len=R['TxNN_seq_len_FFE'],feature=1)
        txPreSamples = TxNetFFE(Input_TxNetFFE.permute(0,2,1)).reshape(-1) 
        
        # transmission over physical channel
        PAPR_dB,rxSamples = Channel(to_numpy(txPreSamples).reshape(-1))
        rxSamples = to_tensor(rxSamples).float()

        # power normalization
        rxSamples = sqrt(R['power_constel']) * rxSamples / torch.sqrt(torch.mean(torch.pow(torch.abs(rxSamples),2))) 
        
    # label
    start_id = (R['TxNN_seq_len_LUT']-1)//2 + (R['TxNN_seq_len_FFE']-1)//2 + (R['RxNN_seq_len']-1)//2    
    label = Variable(txSymbols)   
    # Receiver Equalization
    Input_RxNet = createTxRxInput(rxSamples.reshape(-1),seq_len=R['RxNN_seq_len'],feature=R['RxNN_feature_dim'])
    label = label[start_id:start_id+Input_RxNet.shape[0]]    
    updateSteps = Input_RxNet.shape[0]//R['batchsize_finetune']+1
    for i in range(updateSteps):
        RxOptimizer.zero_grad()
        mask = np.random.choice(Input_RxNet.shape[0],R['batchsize_finetune'],replace=False)
        rxEquSymbols = RxNet(Input_RxNet[mask].permute(0,2,1))
        # calculate mse loss and uptade TxNN,RxNN 
        loss = F.mse_loss(rxEquSymbols.reshape(-1,1),label[mask].reshape(-1,1),reduction='mean')
        loss.backward()
        RxOptimizer.step()
        # print('Transmitter Training Epoch = %d, train loss = %.8f'%(epoch,loss.item()))
    return loss.item()    



def main():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    loss_ch_train_list = []
    loss_ch_valid_list = []    
    valid_BER_list = []
    
    TxNet = TxNN()
    TxNetFFE = TxNN_FFE()    
    RxNetFFE = RxNN_FFE()
    ChannelNet = ChannelNN()    

    FolderName = './ChannelPreTrainModel/'
    savePath1 = '%sChannelNN.pkl'%(FolderName)
    ChannelNet.load_state_dict(torch.load(savePath1))
                 
    TxOptimizer = Adam(TxNet.parameters(),lr = R['lr_Tx'])
    TxFFEOptimizer = Adam(TxNetFFE.parameters(),lr = R['lr_TxFFE'])    
    RxOptimizer_mse = Adam(RxNetFFE.parameters(),lr = R['lr_Rx'])    
    ChanOptimizer = Adam(ChannelNet.parameters(),lr = R['lr_Chan'])
    
    TxNet.to(device)
    TxNetFFE.to(device)
    RxNetFFE.to(device)
    ChannelNet.to(device)    
    
    memory = MemoryBuffer(sample_size = R['sample_size'], memory_size = R['memory_size'], warmup_memory_flag = R['warmup_memory_flag'])
        
    best_valid_BER = 1
    best_TxNet = deepcopy(TxNet)
    best_RxNetFFE = deepcopy(RxNetFFE)   
    best_ChannelNet = deepcopy(ChannelNet)
    
    for epoch_id in range(R['MainEpoch']):
        
        txBits_valid = np.random.randint(0,2,(R['bitNum_valid'],))
        txdataIn_valid = bit2sym(txBits_valid,R['k'],mode='gray')
        txdataIn_valid = torch.from_numpy(txdataIn_valid)
        txdataIn_valid = txdataIn_valid.long().reshape(-1)   
        txdataIn_valid.to(device) 
        
        ChanEpoch = R['ChanEpoch_min'] + 1/2*(R['ChanEpoch_max']-R['ChanEpoch_min'])*(1+np.cos(pi*epoch_id/R['MainEpoch']))
        ChanEpoch = ceil(ChanEpoch) 
        for i in range(ChanEpoch):
            TxNet.eval()
            with torch.no_grad():
                # generate txSymbols
                txPreSamples = np.random.rand(R['symNum_train_Chan']*R['sps']) * 2 - 1
                # txPreSamples = np.random.normal(loc=0, scale=1,size=R['symNum_train_Chan']*R['sps'])
                txPreSamples = to_tensor(txPreSamples)
                txPreSamples = txPreSamples / torch.sqrt(torch.mean(torch.pow(torch.abs(txPreSamples),2)))
                
                # transmission over physical channel
                PAPR_dB,rxSamples = Channel(to_numpy(txPreSamples).reshape(-1))
                rxSamples = to_tensor(rxSamples).float()     
                rxSamples = rxSamples / torch.sqrt(torch.mean(torch.pow(torch.abs(rxSamples),2))) 
            memory.append([txPreSamples,rxSamples])
            
            # Sample from memory
            txSignal, rxSignal = memory.sample_and_split()
            data, label = createDataset(txSignal[0],rxSignal[0])
            for i in range(1,min(len(txSignal),R['sample_size'])):
                data_temp, label_temp = createDataset(txSignal[i],rxSignal[i])
                data = torch.cat([data,data_temp],0)
                label = torch.cat([label,label_temp],0)            
            # train the ChannelNet
            loss_Channel_train = train_Channel(ChannelNet,data,label,ChanOptimizer)
            loss_ch_train_list.append(loss_Channel_train)
        # print('Main Epoch = %d, ChannelNet train loss = %.8f' %(epoch_id,loss_Channel_train))

        txBits_train = np.random.randint(0,2,(R['bitNum_train_Chan'],))
        txdataIn_train = bit2sym(txBits_train,R['k'],mode='gray')
        txdataIn_train = torch.from_numpy(txdataIn_train)
        txdataIn_train = txdataIn_train.long().reshape(-1)
        # txdataIn_train.to(device) 
        TxNet.eval()
        with torch.no_grad():
            OnehotVector = LUTIndexMake(txdataIn_train, L=R['TxNN_seq_len_LUT'], M=R['M'])
            deltaTx = TxNet(OnehotVector.to(device)).reshape(-1)
            # generate txSymbols
            txSymbols = to_tensor(PAM4_MOD(txdataIn_train,R['constel'])).reshape(-1)
            # NN-PDLUT
            txPreSymbols = txSymbols[(R['TxNN_seq_len_LUT']-1)//2:-(R['TxNN_seq_len_LUT']-1)//2] - deltaTx
            # txPreSymbols = txPreSymbols / torch.sqrt(torch.mean(torch.pow(torch.abs(txPreSymbols),2)))  
            
            # TxFFE
            Input_TxNetFFE = createTxRxInput(txPreSymbols.reshape(-1),seq_len=R['TxNN_seq_len_FFE'],feature=1)
            txPreSamples = TxNetFFE(Input_TxNetFFE.permute(0,2,1).to(device) ).reshape(-1)
            
            # transmission over physical channel
            PAPR_dB,rxSamples = Channel(to_numpy(txPreSamples).reshape(-1))
            rxSamples = to_tensor(rxSamples).float()    
            rxSamples = rxSamples / torch.sqrt(torch.mean(torch.pow(torch.abs(rxSamples),2))) 

            # train the ChannelNet
            data, label = createDataset(txPreSamples,rxSamples)
            loss = valid_Channel(ChannelNet,data.to(device) ,label.to(device) )
            loss_ch_valid_list.append(loss)
        print('Main Epoch = %d, ChannelNet valid loss = %.8f' %(epoch_id,loss))

        if epoch_id < R['TxEpoch_boundary']:
            TxEpoch = R['TxEpoch'] - 1/2*(R['TxEpoch']-R['TxEpoch_init'])*(1+np.cos(pi*epoch_id/R['TxEpoch_boundary']))
            TxEpoch = ceil(TxEpoch) 
        else:
            TxEpoch = R['TxEpoch']       
        for i in range(TxEpoch):    
            txBits_train = np.random.randint(0,2,(R['bitNum_train_TxNN'],))
            txdataIn_train = bit2sym(txBits_train,R['k'],mode='gray')
            txdataIn_train = torch.from_numpy(txdataIn_train)
            txdataIn_train = txdataIn_train.long().reshape(-1)  
            txdataIn_train.to(device) 
            # train the ModNet and TxNN
            loss_transceiver_train = train_Transceiver(TxNet,TxNetFFE,RxNetFFE,ChannelNet,txdataIn_train,TxOptimizer,TxFFEOptimizer,RxOptimizer_mse)
        print('Main Epoch = %d, Transceiver train loss = %.8f' %(epoch_id,loss_transceiver_train))
                
        # validate TxNN and RxNN, and finetune the RxNN
        valid_BER,txPreSymbols_valid,txPreSamples_valid,PAPR_valid,rxSamples_valid,rxEquSymbols_valid,label_valid_valid = valid_withRxTrain(TxNet,TxNetFFE,txdataIn_valid) 
        valid_BER_list.append(valid_BER)
        # save the best network
        if valid_BER <= best_valid_BER and epoch_id >=R['MainEpoch']//2:
            best_valid_BER = valid_BER
            best_TxNet = deepcopy(TxNet)
            best_TxNetFFE = deepcopy(TxNetFFE)
            best_RxNetFFE = deepcopy(RxNetFFE)
            best_ChannelNet = deepcopy(ChannelNet)
            best_txdataIn_valid,best_txPreSymbols_valid,best_txPreSamples_valid,best_PAPR_valid,best_rxSamples_valid,best_rxEquSymbols_valid,best_label_valid_valid = \
            txdataIn_valid,txPreSymbols_valid,txPreSamples_valid,PAPR_valid,rxSamples_valid,rxEquSymbols_valid,label_valid_valid
        print('Main Epoch = %d, Valid BER = %.8f' %(epoch_id,valid_BER))

    return best_txdataIn_valid,best_txPreSymbols_valid,best_txPreSamples_valid,best_PAPR_valid,best_rxSamples_valid,best_rxEquSymbols_valid,best_label_valid_valid,\
        best_TxNet,best_TxNetFFE,best_RxNetFFE,best_ChannelNet,\
        best_valid_BER,valid_BER_list,loss_ch_train_list,loss_ch_valid_list


R = {}

# Parameter for Truncated Normal Distribution
R['alpha'] = 1

# System parameters
R['M'] = 4  # PAM-4
R['k'] = int(log2(R['M']))
R['constel'] = np.array([-3,-1,1,3])
R['constel'] = R['constel']/sqrt(np.mean(np.absolute(R['constel'])**2))
R['power_constel'] = 1
R['symNum_train_TxNN'] = 2048
R['bitNum_train_TxNN'] = R['symNum_train_TxNN']*R['k']
R['symNum_train_Chan'] = 4096
R['bitNum_train_Chan'] = R['symNum_train_Chan']*R['k']
R['batchsize'] = 256
R['symNum_train_finetune'] = 16384
R['bitNum_train_finetune'] = R['symNum_train_finetune']*R['k']
R['batchsize_finetune'] = 32
R['symNum_valid'] = 32768
R['bitNum_valid'] = R['symNum_valid']*R['k']
R['SymRate'] = 50e9     # Baud
R['sps'] = 2            # sample per symbol  
R['SamRate'] = R['SymRate']*R['sps']     # sample rate
R['DAC_Sample_Rate'] = 120e9             # AWG sampling rate

# Parameters for Memory Buffer
R['sample_size'] = 5
R['memory_size'] = 100
R['warmup_memory_flag'] = False

# TxNN parameters
R['TxNN_Type'] = 'NNPDLUT_Conv'
R['TxNN_scale'] = [1]
R['TxNN_seq_len_LUT'] = 5
R['TxNN_seq_len_FFE'] = 55
R['TxNN_feature_dim']  = 1
R['TxNN_input_dim'] = R['TxNN_feature_dim']


# RxNN parameters
R['RxNN_Type'] = 'Conv'
R['RxNN_seq_len'] = 55
R['RxNN_feature_dim']  = 1
R['RxNN_input_dim'] = R['RxNN_feature_dim']

# ChannelNN parameters
R['ChannelNN_Type'] = 'MscaleDNN'
R['ChanNN_scale'] = [1,2,4,8,16]
R['Chan_seq_len'] = 129
R['Chan_feature_dim']  = R['sps']
R['ChanNN_input_dim'] = R['sps']
R['ChanNN_output_dim'] = R['sps']

# ChannelNet training parameters
R['ChanEpoch_min'] = 2
R['ChanEpoch_max'] = 2
R['lr_Chan'] = 1e-4

# Transceiver training parameters
R['TxEpoch_boundary'] = 10
R['TxEpoch_init'] = 100
R['TxEpoch'] = 150
R['lr_Tx'] = 5e-4
R['lr_TxFFE'] = 5e-4
R['lr_Rx'] = 5e-4

# main training parameters
R['MainEpoch'] = 60
# Receiver finetune training parameters
R['RxEpoch_mse'] = 20


txdataIn,txPreSymbols,txPreSamples,PAPR,rxSamples,rxEquSymbols,label,\
    best_TxNet,best_TxNetFFE,best_RxNetFFE,best_ChannelNet,\
    best_valid_BER,valid_BER_list,loss_ch_train_list,loss_ch_valid_list = main()
plt.figure(1)
plt.plot(valid_BER_list, label="Test BER")
plt.legend(loc='upper right')   

indexList = np.arange(1,len(rxEquSymbols)+1)
plt.figure(2)
plt.scatter(indexList,rxEquSymbols,s=5,color='b',label="Rx Equalized signal")
plt.legend()

### save TxNN, RxNN and Noise adaptation channel

# FolderName = './PDLUT_E2E_ModelSave/ModelSave_1/'

# save_path_1 = '%sbest_TxNet.pkl'%(FolderName)
# torch.save(best_TxNet.state_dict(),save_path_1)

# save_path_2 = '%sbest_TxNetFFE.pkl'%(FolderName)
# torch.save(best_TxNetFFE.state_dict(),save_path_2)

# save_path_3 = '%sbest_ChannelNet.pkl'%(FolderName)
# torch.save(best_ChannelNet.state_dict(),save_path_3)

# save_path_4 = '%sbest_RxNetFFE.pkl'%(FolderName)
# torch.save(best_RxNetFFE.state_dict(),save_path_4)


