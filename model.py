#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 15:04:17 2021

@author: yesh
"""

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, letter_size, n_classes, padding_idx, 
                 num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.letter_embeddings = nn.Embedding(letter_size, embedding_dim, padding_idx=padding_idx)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.num_layers, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.fc = nn.Linear(hidden_dim, n_classes)
        
        self.dropout_layer = nn.Dropout(p=0.2)
        
    def forward(self, word):        
        embeds = self.letter_embeddings(word)
        lstm_out, _ = self.lstm(embeds)
        out = self.dropout_layer(lstm_out)
        out = self.fc(out)
        out = F.log_softmax(out, dim=2)
        return out


class DatasetNames(Dataset):    
    def __init__(self, names, sexes, pad_len = 35):
        self.pad_len = pad_len
        
        self.sexes = sexes
        self.names = [name.lower() for name in names]
        self.names_letters = [self.__split__(name) for name in self.names]
        
        self.uniq_letters = self.__get_uniq_letters__(self.names)
        self.uniq_letters = np.append(self.uniq_letters, [' ', '0'])
        
        self.index_to_letter = {index: letter for index, letter in enumerate(self.uniq_letters)}
        self.letter_to_index = {letter: index for index, letter in enumerate(self.uniq_letters)}
        
        # self.letter_indexes = [self.letter_to_index[letter] for letter in self.letters]
        
    def __split__(self, word):
        return [str(char) for char in word]
    
    def __get_uniq_letters__(self, names):
        return np.unique([char for char in ' '.join(names) if char != ' '])
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, index):
        name_letters = np.array(self.names_letters[index]).astype(str)
        padding = self.pad_len - len(name_letters)
        
        name_letters = np.pad(name_letters, (padding,0), mode='constant', constant_values='0')
        name_numbers = []
        for l in name_letters:
            try:
                name_numbers.append(self.letter_to_index[l])
            except Exception as e:
                import pdb; pdb.set_trace()
         
        
        return {'name': np.array(name_numbers),
                'sex': self.sexes[index],
                'name_eng': [self.names[index]]}
    
    def get_custom_name(self, name):
        name_letters = self.__split__(name.lower())
        name_numbers = [self.letter_to_index[letter] for letter in name_letters]
        return np.array(name_numbers)