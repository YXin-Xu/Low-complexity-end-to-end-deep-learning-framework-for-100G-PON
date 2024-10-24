# -*- coding: utf-8 -*-
"""
Test code for "Low-complexity end-to-end deep learning framework for 100G-PON"

Process:
TxBits--PAM4 Mod--TxNN--Physical Channel--RxNN--PAM4 DeMod--RxBits

Keep TxNN fixed, finetune RxNN

@author: Yongxin Xu, xuyongxin@sjtu.edu.cn
"""

from torch.optim import Adam,AdamW
import torch.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
from torch.distributions import Normal
from collections import OrderedDict
import math
from math import sqrt,log2,log10,pi,ceil
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from copy import deepcopy
from utils import bit2sym,sym2bit,ber,ser,PAM4_MOD,PAM4_DEMOD,PAM4_Decoder,rcosdesign,to_tensor,to_numpy,minmaxScale,resample_poly,powerNorm

import torchaudio
import scipy.io as sio
import scipy.fftpack as fftpack
import scipy.signal as scisig
import scipy.fft as scifft

# import matlab
# import matlab.engine
# eng = matlab.engine.start_matlab()

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)

## TxNN: (1)NNPDLUT
class TxNN_main(nn.Module):
    def __init__(self):
        super(TxNN_main, self).__init__()
        self.in_dim = 1
        self.out_dim = 1
        self.seqLen = R['TxNN_seqLen_LUT']
        self.fc0 = nn.Linear(R['M']**R['TxNN_seqLen_LUT'],64)
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

## TxNN: (2)Conv layer
class TxNN_FFE_Base(nn.Module):
    def __init__(self):
        super(TxNN_FFE_Base, self).__init__()
        self.in_dim = 1
        self.out_dim = 2
        self.seqLen = R['TxNN_seqLen_Conv']
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


## RxNN: Conv layer
class RxNN_FFE(nn.Module):
    def __init__(self):
        super(RxNN_FFE, self).__init__()
        self.in_dim = 1
        self.out_dim = 1
        self.conv1 = nn.Conv1d(self.in_dim,self.out_dim,R['RxNN_seqLen_1sps'],bias=True)
    def forward(self, x):  
        xx = self.conv1(x)
        out = xx.reshape(-1,self.out_dim)
        return out
    

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


# Create Input for TxNN
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

# Create dataset for RxNN
def createRxDataset(tx,rx,seq_len,feature):
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


# Finetune Receiver
def train_Receiver_mse(RxNet,data,label,RxOptimizer,epoch):
    RxNet.train()
    for k in range(epoch):
        updateSteps = data.shape[0]//R['batchsize_finetune']+1
        for i in range(updateSteps):
            RxOptimizer.zero_grad()
            mask = np.random.choice(data.shape[0],R['batchsize_finetune'],replace=False)
            rxEquSymbols = RxNet(data[mask].permute(0,2,1))
            # calculate mse loss and uptade RxNN 
            loss = F.mse_loss(rxEquSymbols.reshape(-1,1),label[mask].reshape(-1,1),reduction='mean')
            loss.backward()
            RxOptimizer.step()
            # print('Training Epoch = %d, train loss = %.8f'%(epoch,loss.item()))
    return RxNet,loss.item()    


def main():
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # load model
    TxPDLUTNet = TxNN()
    TxConvNet = TxNN_FFE()
    TxPDLUTNet.to(device)
    TxConvNet.to(device)
    FolderName = './ModelSave/'
    savePath1 = '%sTxPDLUTNet.pkl'%(FolderName)
    TxPDLUTNet.load_state_dict(torch.load(savePath1,map_location=torch.device('cpu')))
    savePath1 = '%sTxConvNet.pkl'%(FolderName)
    TxConvNet.load_state_dict(torch.load(savePath1,map_location=torch.device('cpu')))
    

    RxNet_1sps = RxNN_FFE()
    RxNet_1sps.to(device)
    RxOptimizer_1sps = Adam(RxNet_1sps.parameters(),lr = R['lr_Rx'])   
    
    FolderName = './Dataset/E2E/-18.5dBm/'
    save_path_1 = '%stest1.mat'%(FolderName)
    mat_data = sio.loadmat(save_path_1)
    txSymbols = mat_data['txSymbols'].reshape(-1)  
    rxSamples = mat_data['rxSamples'].reshape(-1)
    
    with torch.no_grad():
        dataIn = mat_data['dataIn'].reshape(-1)
        txdataIn_test = to_tensor(dataIn)
        OnehotVector = LUTIndexMake(txdataIn_test, L=R['TxNN_seqLen_LUT'], M=R['M'])
        deltaTx = TxPDLUTNet(OnehotVector.to(device)).reshape(-1) # NNLUT(i)
        # txSymbols = to_tensor(PAM4_MOD(txdataIn_test,R['constel'])).reshape(-1)
        # txPreSymbols = txSymbols[(R['TxNN_seqLen_LUT']-1)//2:-(R['TxNN_seqLen_LUT']-1)//2] - deltaTx
        
        index = torch.arange(0,1024)
        indexOnehot = torch.nn.functional.one_hot(index,num_classes=1024)
        indexOnehot  = indexOnehot.float()
        LUT_e = TxPDLUTNet(indexOnehot.to(device)).reshape(-1)
        
        # E2E NNPDLUT's value
        LUT_e = to_numpy(LUT_e)
        x = np.arange(len(LUT_e))
        plt.figure()
        plt.scatter(x,LUT_e,label='E2E')
        plt.xlabel("LUT Index")
        plt.show()    

    start_id = (R['TxNN_seqLen_LUT']-1)//2 + (R['TxNN_seqLen_Conv']-1)//2
    txSymbols_cut = txSymbols[start_id:-start_id]
    # resample to 1sps
    rxSymbols = resample_poly(rxSamples,R['SymRate'],R['SymRate']*R['sps'])
    rxSymbols = rxSymbols[0:len(rxSamples)//2]

    # power spectral density of rxSamples of E2E in real channel
    plt.figure()
    symRate = 50e9
    SpS = 2
    plt.psd(powerNorm(rxSamples),Fs=symRate*SpS, NFFT = 1024, sides='onesided', label = 'Noise Adaptation Channel')

    rxSamples = to_tensor(rxSamples).float()
    rxSymbols = to_tensor(rxSymbols).float()
    txSymbols_cut = to_tensor(txSymbols_cut).float()
    
    # 1sps rx equalization
    # train
    trainLen = 10000
    Input_RxNet,label_train = createRxDataset(txSymbols_cut[:trainLen],rxSymbols[:trainLen],seq_len=R['RxNN_seqLen_1sps'],feature=1)
    RxNet_1sps,loss_train = train_Receiver_mse(RxNet_1sps,Input_RxNet,label_train,RxOptimizer_1sps,epoch=50)
    print('RxNN Train loss=%f'%loss_train)
    # test
    Input_RxNet,label_test = createRxDataset(txSymbols_cut[trainLen:],rxSymbols[trainLen:],seq_len=R['RxNN_seqLen_1sps'],feature=1)
    rxEquSymbols = RxNet_1sps(Input_RxNet.permute(0,2,1)).detach()
    rxEquSymbols = rxEquSymbols  - torch.mean(rxEquSymbols)
    rxEquSymbols = rxEquSymbols / torch.sqrt(torch.mean(torch.abs(rxEquSymbols)**2))
    # PAM4 Decision
    txData,txBits = PAM4_Decoder(label_test.cpu().numpy(),R['constel'],R['k'])
    rxData,rxBits = PAM4_Decoder(rxEquSymbols.cpu().numpy(),R['constel'],R['k'])
    test_BER = ber(txBits,rxBits)       
    print('1sps RxNN Test BER = %.8f' %(test_BER))
    
    return rxEquSymbols,label_test,test_BER


## System Parameters
R = {}
R['M'] = 4  # PAM-4
R['k'] = int(log2(R['M']))
R['constel'] = np.array([-3,-1,1,3])
R['constel'] = R['constel']/sqrt(np.mean(np.absolute(R['constel'])**2))
R['SymRate'] = 50e9     # Baud
R['sps'] = 2            # sample per symbol  
R['SamRate'] = R['SymRate']*R['sps']    # sample rate
R['DAC_Sample_Rate'] = 120e9            # AWG sampling rate
# R['roll_off'] = 0.1

## Training Parameters
R['batchsize_finetune'] = 32
R['symNum_test'] = 32768
R['bitNum_test'] = R['symNum_test']*R['k']

# TxNN parameters
R['TxNN_seqLen_LUT'] = 5
R['TxNN_seqLen_Conv'] = 55
R['TxNN_feature_dim']  = 1
R['TxNN_input_dim'] = R['TxNN_feature_dim']
R['TxNN_output_dim'] = R['sps']

# RxNN parameters 1sps
R['RxNN_seqLen_1sps'] = 65*2+1

# Receiver finetune training parameters
R['RxEpoch_mse'] = 50
R['lr_Rx'] = 2.5e-4

rxEquSymbols,label_test,test_BER = main()

