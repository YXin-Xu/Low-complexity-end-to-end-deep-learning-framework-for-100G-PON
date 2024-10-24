# -*- coding: utf-8 -*-
"""
基于信道建模的端到端 DML/DD系统: 
    信道训练数据缓存
    
memory.py

"""
import numpy as np
import pickle
from collections import deque

class MemoryBuffer(object):
  def __init__(self, sample_size, memory_size, warmup_memory_flag):
    self.sample_size = sample_size
    self.memory_size = memory_size
    self.warmup_memory_flag = warmup_memory_flag
    self.memory = deque(maxlen=self.memory_size)

    if warmup_memory_flag == True:
        self.memory = pickle.load(open('save/warmup_memory_file', 'rb'))

  def __len__(self):
    return len(self.memory)

  def append(self, item):
    self.memory.append(item)
    
  def save_warmup_memory(self):
    pickle.dump(self.memory, open('save/warmup_memory_file', 'wb'))

  def sample_batch(self):
    idx = np.random.permutation(len(self.memory))[:self.sample_size]
    return [self.memory[i] for i in idx]

  def sample_and_split(self):
    txSignal, rxSignal = zip(*self.sample_batch())

    # txSignal = np.array(txSignal, 'double').reshape(self.sample_size, -1)
    # rxSignal = np.array(rxSignal, 'double').reshape(self.sample_size, -1)

    return txSignal, rxSignal


