# -*- coding: utf-8 -*-
"""

utils.py

@author: YX X
"""
import fractions
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam
from scipy import signal as scisig
import torch.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt,log2,log10,pi
from cmath import exp
import matplotlib.pyplot as plt
import time
import os
import random
import numpy as np
import seaborn as sns
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal


def minmaxScale(signal):
    min_values = torch.min(signal, dim=0).values
    max_values = torch.max(signal, dim=0).values
    
    normalized_signal = 2*(signal - min_values) / (max_values - min_values) - 1
    return normalized_signal

def to_numpy(var):
    use_cuda = torch.cuda.is_available()
    return var.cpu().data.numpy() if use_cuda else var.data.numpy()


def to_tensor(ndarray, requires_grad=False):    # return a float tensor by default
    tensor = torch.from_numpy(ndarray).float()  # by default does not require grad
    if requires_grad:
        tensor.requires_grad_()
    return tensor.cuda() if torch.cuda.is_available() else tensor    


def PAM4_MOD(txdataIn,constel):
    idx = to_numpy(txdataIn.reshape(-1)).astype(np.int64)
    Mod_data = constel[idx]
    
    return Mod_data
    

def PAM4_DEMOD(rxSymbols,constel):
    rxSymbols = rxSymbols.reshape(1,-1)
    constel = constel.reshape(-1,1)
    symLen = rxSymbols.shape[1]
    s2 = abs(constel)**2
    s2Array = np.tile(s2,(1,symLen))
    rxReArray = constel.dot(rxSymbols)
    rxEud = s2Array-2*rxReArray
    rxIndex = np.argmin(rxEud,axis=0)
    DeMod_data = constel[rxIndex]
    
    return DeMod_data 


def PAM4_Decoder(Symbols,constel,k):
    Symbols = Symbols.reshape(1,-1)
    constel = constel.reshape(-1,1)
    rxEud = abs(Symbols - constel)**2
    rxIndex = np.argmin(rxEud,axis=0)
    DeMod_data = constel[rxIndex]
    dataOut = rxIndex
    Bits = sym2bit(dataOut,k,mode='gray',out='seq')
    return dataOut,Bits


def bit2sym(x,k,mode):
    """
    converts k column-wise bit elements in x to integer indexes of modulation symbols.

    Parameters
    ----------
    x: bit sequence to be converted
    k : bit num, for PAM-4, k=2
    mode : 'nature' for natural binary code,'gray' for Gray code

    Returns
    -------
    sym : integer indexes of modulation symbols with natural binary code
    sym_gray : integer indexes of modulation symbols with Gray code

    """    
    if x.shape[0]%k==0:
        sym_num=int(x.shape[0]/k)
        x1=x.reshape(-1,k)
        sym=np.zeros((sym_num,1))
        for i in range(sym_num):
            for j in range(k):
                sym[i]+=x1[i][k-j-1]*(2**j)
        if mode=='nature': 
            return sym
        elif mode=='gray': 
            sym_gray=np.zeros((sym_num,1))
            bit_gray=np.zeros((sym_num,k))
        list=['0','1']
        for i in range(1,k):
            left=['0'+i for i in list]
            right=['1'+i for i in list[::-1]]
            list=left+right
        for i in range(sym_num):
            for j in range(k):
                bit_gray[i][k-j-1]=list[int(sym[i])][k-j-1]
                sym_gray[i]+=bit_gray[i][k-j-1]*(2**j)
        return sym_gray
    else:
        print('Unable to divide')
        
        
def sym2bit(sym,k,mode,out='seq'):
    """
    converts each integer element in sym to k column-wise bits.

    Parameters
    ----------
    sym: integer indexes of modulation symbols to be converted
    k : bit num, for PAM-4, k=2
    mode : 'nature' for natural binary code,'gray' for Gray code
    out : 'seq' for sequential output form, 'para' for parallel output form

    Returns
    -------
    bit_ : results in natural binary code
    bit_gray : results in Gray code

    """    
    sym_num=sym.shape[0]
    bit_=np.zeros((sym_num,k))
    for i in range(sym_num):
        num=sym[i]
        for j in range(k):
            num,remainder=divmod(num,2)
            bit_[i][k-j-1]=remainder
    if mode=='nature':
        if out=='seq':
            bit_=bit_.reshape(-1)
            return bit_
        else:
            return bit_
    elif mode=='gray':
        bit_gray=np.zeros((sym_num,k))
        for i in range(k):
            if i==0:
                bit_gray[:,i]=bit_[:,i]
            else:
                bit_gray[:,i]=(bit_[:,i]+bit_gray[:,i-1])%2
        if out=='seq':    #bit_gray  shape (sym_num,)
            bit_gray=bit_gray.reshape(-1)
            return bit_gray
        elif out=='para':  #bit_gray  shape (sym_num/k,k)
            return bit_gray

def ber(x, y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    num = x.shape[0]
    correct = sum(x==y)
    return (num - correct) / num

def ser(x, y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    num = x.shape[0]
    correct = sum(x==y)
    return (num - correct) / num


def resample_poly(signal,fnew,fold,window=None,renormalise=False):
    """
    Resamples a signal from an old frequency to a new. Preserves the whole data
    but adjusts the length of the array in the process.

    Parameters
    ----------
    signal: array_like
        signal to be resampled
    fold : float
        Sampling frequency of the signal
    fnew : float
        New desired sampling frequency.
    window : array_like, optional
        sampling windowing function
    renormalise : bool, optional
        whether to renormalise and recenter the signal to a power of 1.

    Returns
    -------
    out : array_like
        resampled signal of length fnew/fold*len(signal)

    """
    signal = signal.flatten()
    L = len(signal)
    up, down = resamplingfactors(fold, fnew)
    if window is None:
        sig_new = scisig.resample_poly(signal, up, down)
    else:
        sig_new = scisig.resample_poly(signal, up, down, window=window)
    if renormalise:
        p = np.mean(np.abs(signal)**2)
        sig_new = normalise_and_center(sig_new)*np.sqrt(p)
    return sig_new

def resamplingfactors(fold, fnew):
    ratn = fractions.Fraction(fnew/fold).limit_denominator()
    return ratn.numerator, ratn.denominator

def normalise_and_center(E):
    """
    Normalise and center the input field, by calculating the mean power for each polarisation separate and dividing by its square-root
    """
    if E.ndim > 1:
        E = E - np.mean(E, axis=-1)[:, np.newaxis]
        P = np.sqrt(np.mean(cabssquared(E), axis=-1))
        E /= P[:, np.newaxis]
    else:
        E = E.real - np.mean(E.real) + 1.j * (E.imag-np.mean(E.imag))
        P = np.sqrt(np.mean(cabssquared(E)))
        E /= P
    return E

def cabssquared(x):
    """Calculate the absolute squared of a complex number"""
    return x.real**2 + x.imag**2

def powerNorm(x,DCremove=False):
    """
    Normalize the average power of each componennt of x.

    Parameters
    ----------
    x : np.array
        Signal.

    Returns
    -------
    np.array
        Signal x with each component normalized in power.

    """
    if DCremove==False:
        return x / np.sqrt(np.mean(x * np.conj(x)).real)
    else:
        x = x - np.mean(x)
        return x / np.sqrt(np.mean(x * np.conj(x)).real)
    

def rcosdesign(beta: float, span: float, sps: float, shape='normal'):
    """ Raised cosine FIR filter design
    Calculates square root raised cosine FIR
    filter coefficients with a rolloff factor of `beta`. The filter is
    truncated to `span` symbols and each symbol is represented by `sps`
    samples. rcosdesign designs a symmetric filter. Therefore, the filter
    order, which is `sps*span`, must be even. The filter energy is one.
    Keyword arguments:
    beta  -- rolloff factor of the filter (0 <= beta <= 1)
    span  -- number of symbols that the filter spans
    sps   -- number of samples per symbol
    shape -- `normal` to design a normal raised cosine FIR filter or
             `sqrt` to design a sqre root raised cosine filter
    """

    if beta < 0 or beta > 1:
        raise ValueError("parameter beta must be float between 0 and 1, got {}"
                         .format(beta))

    if span < 0:
        raise ValueError("parameter span must be positive, got {}"
                         .format(span))

    if sps < 0:
        raise ValueError("parameter sps must be positive, got {}".format(span))

    if ((sps*span) % 2) == 1:
        raise ValueError("rcosdesign:OddFilterOrder {}, {}".format(sps, span))

    if shape != 'normal' and shape != 'sqrt':
        raise ValueError("parameter shape must be either 'normal' or 'sqrt'")

    eps = np.finfo(float).eps

    # design the raised cosine filter

    delay = span*sps/2
    t = np.arange(-delay, delay)

    if len(t) % 2 == 0:
        t = np.concatenate([t, [delay]])
    t = t / sps
    b = np.empty(len(t))

    if shape == 'normal':
        # design normal raised cosine filter

        # find non-zero denominator
        denom = (1-np.power(2*beta*t, 2))
        idx1 = np.nonzero(np.fabs(denom) > np.sqrt(eps))[0]

        # calculate filter response for non-zero denominator indices
        b[idx1] = np.sinc(t[idx1])*(np.cos(np.pi*beta*t[idx1])/denom[idx1])/sps

        # fill in the zeros denominator indices
        idx2 = np.arange(len(t))
        idx2 = np.delete(idx2, idx1)

        b[idx2] = beta * np.sin(np.pi/(2*beta)) / (2*sps)

    else:
        # design a square root raised cosine filter

        # find mid-point
        idx1 = np.nonzero(t == 0)[0]
        if len(idx1) > 0:
            b[idx1] = -1 / (np.pi*sps) * (np.pi * (beta-1) - 4*beta)

        # find non-zero denominator indices
        idx2 = np.nonzero(np.fabs(np.fabs(4*beta*t) - 1) < np.sqrt(eps))[0]
        if idx2.size > 0:
            b[idx2] = 1 / (2*np.pi*sps) * (
                np.pi * (beta+1) * np.sin(np.pi * (beta+1) / (4*beta))
                - 4*beta           * np.sin(np.pi * (beta-1) / (4*beta))
                + np.pi*(beta-1)   * np.cos(np.pi * (beta-1) / (4*beta))
            )

        # fill in the zeros denominator indices
        ind = np.arange(len(t))
        idx = np.unique(np.concatenate([idx1, idx2]))
        ind = np.delete(ind, idx)
        nind = t[ind]

        b[ind] = -4*beta/sps * (np.cos((1+beta)*np.pi*nind) +
                                np.sin((1-beta)*np.pi*nind) / (4*beta*nind)) / (
                                np.pi * (np.power(4*beta*nind, 2) - 1))

    # normalize filter energy
    b = b / np.sqrt(np.sum(np.power(b, 2)))
    return b