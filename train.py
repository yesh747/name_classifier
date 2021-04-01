#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 23:11:29 2021

@author: yesh
"""

import os
#os.chdir('/Users/yesh/Documents/ent_ai/name_classifier')
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim


# Currently: seeing if new code does as well as old code with batch_size =1
# TODO:
#   - add frequency as input factor M:F ratio
#   - look for database for international names as well
#   - trying adding these things: https://github.com/claravania/lstm-pytorch/blob/master/model.py
#   - expierment - which runs better: fc_out = self.fc(lstm_out[:,-1]) or fc_out = self.fc(lstm_out[:,-1]) 


class DatasetNames(Dataset):    
    def __init__(self, names, sexes, pad_len = 20):
        self.pad_len = pad_len
        
        self.sexes = sexes
        self.names = [name.lower() for name in names]
        self.names_letters = [self.__split__(name) for name in self.names]
        
        self.uniq_letters = self.__get_uniq_letters__(self.names)
        self.uniq_letters = np.append(self.uniq_letters, ['0'])
        
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
            name_numbers.append(self.letter_to_index[l])
         
        
        return {'name': np.array(name_numbers),
                'sex': self.sexes[index],
                'name_eng': [self.names[index]]}
    
    def get_custom_name(self, name):
        name_letters = self.__split__(name.lower())
        name_numbers = [self.letter_to_index[letter] for letter in name_letters]
        return np.array(name_numbers)


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



# PARAMS
DATADIR = './data'


# 1. Load data
# Data from: https://www.ssa.gov/oact/babynames/limits.html
data_fps = [os.path.join(DATADIR,file) for file in os.listdir(DATADIR) if file.endswith('.txt')]

colnames = ['name', 'sex', 'freq']
df = pd.DataFrame(columns=colnames)
for fp in data_fps:
    df_sub = pd.read_csv(fp, header=None)
    df_sub.columns = colnames
    df = df.append(df_sub)
    
# - convert sex to binary
df['sex'] = np.where(df['sex'] == 'M', 1, 0)
    
# - drop repeats
df = df.groupby(['name', 'sex'], as_index=False).agg('sum')

# - count how many male only and how many female only
female_names = df[df['sex'] == 0]['name'].values
male_names = df[df['sex'] == 1]['name'].values
both_names = np.intersect1d(female_names, male_names)
female_names = set(both_names) ^ set(female_names)
male_names =  set(both_names) ^ set(male_names)

print('{} ({:.2f}%) names are both male and female'.format(len(both_names),
                                                       100*len(both_names)/len(np.unique(df['name']))))

# - train/val/test split
train_names, val_names = train_test_split(np.unique(df['name']),
                               test_size=0.1, random_state=1)
val_names, test_names = train_test_split(val_names, test_size=0.5, random_state=1)
    
    
# Dataloaders
dataset_train = DatasetNames(df[df['name'].isin(train_names)]['name'].values,
                       df[df['name'].isin(train_names)]['sex'].values)
dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
    
dataset_val = DatasetNames(df[df['name'].isin(val_names)]['name'].values,
                       df[df['name'].isin(val_names)]['sex'].values)
dataloader_val = DataLoader(dataset_val, batch_size=64, shuffle=True)  

dataset_test = DatasetNames(df[df['name'].isin(test_names)]['name'].values,
                       df[df['name'].isin(test_names)]['sex'].values)
dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)  


# - IMPORTANT! Using the training set index_to_letter converters
dataset_val.index_to_letter = dataset_train.index_to_letter
dataset_val.letter_to_index = dataset_train.letter_to_index

dataset_test.index_to_letter = dataset_train.index_to_letter
dataset_test.letter_to_index = dataset_train.letter_to_index
    
# Model initiation
model = LSTMClassifier(embedding_dim=192,
                       hidden_dim=128, 
                       letter_size=len(dataset_train.uniq_letters), 
                       n_classes=2, 
                       padding_idx=dataset_train.letter_to_index['0'],
                       num_layers=4)

loss_function = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0001)


def calc_accuracy(y_preds, y_gts):
    y_preds = np.argmax(y_preds,axis=1)
    
    accuracy = []
    for i in range(len(y_gts)):
        if y_preds[i] == y_gts[i]:
            accuracy.append(1)
        else:
            accuracy.append(0)
    return np.mean(accuracy)
    
    
model.train()
t0 = time.time()
i_t = None
for epoch in range(40):
    running_loss = []
    running_accuracy = []
    for i, data in enumerate(dataloader_train):
        inputs = data['name']
        target = data['sex']

        outputs = model(inputs)
        
        loss = loss_function(outputs[:,-1], target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        accuracy = calc_accuracy(outputs[:,-1].detach().numpy(), 
                                 target.detach().numpy())
        
        running_loss.append(loss.detach().numpy())
        running_accuracy.append(accuracy)
        
        interval = int(10000 / len(data['sex']))
        if i % interval == 0:
            print('[{}, {:7d}] - loss: {:.3f} - accuracy: {:.3f} - time {:.2f}min'.format(
                                                                         epoch, i, 
                                                                         np.mean(running_loss),
                                                                         np.mean(running_accuracy),
                                                                         (time.time()-t0)/60))
            running_loss = []
            running_accuracy = []
    
    # Evaluation metrics
    with torch.no_grad():
        val_loss = []
        val_accuracy = []
        for i, data in enumerate(dataloader_val):
            inputs = data['name']
            target = data['sex']
            
            
            outputs = model(inputs)
            
            loss = loss_function(outputs[:,-1], target)
            accuracy = calc_accuracy(outputs[:,-1].detach().numpy(), 
                                 target.detach().numpy())
            
            val_loss.append(loss.detach().numpy())
            val_accuracy.append(accuracy)
            
        print('[{} - VALIDATON] - loss: {:.3f} - accuracy: {:.3f}'.format(epoch,
                                                                         np.mean(val_loss),
                                                                         np.mean(val_accuracy)))

# Evaluation metrics
with torch.no_grad():
    test_loss = []
    test_accuracy = []
    for i, data in enumerate(dataloader_test):
        inputs = data['name']
        target = data['sex']
        
        outputs = model(inputs)
        
        loss = loss_function(outputs[:,-1], target)
        accuracy = calc_accuracy(outputs[:,-1].detach().numpy(), 
                             target.detach().numpy())
        
        test_loss.append(loss.detach().numpy())
        test_accuracy.append(accuracy)
        
    print('[{} - TEST] - loss: {:.3f} - accuracy: {:.3f}'.format(epoch,
                                                                     np.mean(test_loss),
                                                                     np.mean(test_accuracy)))




# TESTING ON CUSTOM NAMES

# see if name in database and if M/F/both
name = 'benny'
print(df[df['name'].str.lower() == name.lower()])
with torch.no_grad():
    inputs = dataset_train.get_custom_name(name)
    outputs = model(torch.tensor([inputs]))
    print(torch.softmax(outputs, dim=2)[:,-1])
    
    
    


    
    
    
    
    
    
    
    

    
