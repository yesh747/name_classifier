#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:33:36 2021

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


# TODO:
#   - look for database for international names as well
#   - try increasing num_layers=2 for the nn.LSTM 

class DatasetNames(Dataset):    
    def __init__(self, names, sexes):
        
        self.sexes = sexes
        self.names = [name.lower() for name in names]
        self.names_letters = [self.__split__(name) for name in self.names]
        
        self.uniq_letters = self.__get_uniq_letters__(self.names)
        
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
        name_letters = self.names_letters[index]
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
    def __init__(self, embedding_dim, hidden_dim, letter_size, n_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.letter_embeddings = nn.Embedding(letter_size, embedding_dim)
        
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, word):
        embeds = self.letter_embeddings(word)
        lstm_out, _ = self.lstm(embeds.view(len(word), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(word), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

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
train_names, val_names = train_test_split(np.unique(df['name'].str.lower()),
                               test_size=0.1, random_state=1)
val_names, test_names = train_test_split(val_names, test_size=0.5, random_state=1)


# Dataloaders
dataset = DatasetNames(df['name'].values, df['sex'].values)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)




model = LSTMClassifier(embedding_dim=128, hidden_dim=128, 
               letter_size=len(dataset.uniq_letters), 
               n_classes=2)
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
    for i, data in enumerate(dataloader):
        
        inputs = data['name'][0]
        target = data['sex']
        
        # only train on train_names
        if data['name_eng'][0][0] not in train_names:
            continue
        
        outputs = model(inputs)
        
        loss = loss_function(outputs[-1:], target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        accuracy = calc_accuracy(outputs[-1:].detach().numpy(), 
                                 target.detach().numpy())
        
        running_loss.append(loss.detach().numpy())
        running_accuracy.append(accuracy)
        
        interval = 10000
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
        for i, data in enumerate(dataloader):
            inputs = data['name'][0]
            target = data['sex']
            
            # only train on training names
            if data['name_eng'][0][0] not in val_names:
                continue
            
            outputs = model(inputs)
            
            loss = loss_function(outputs[-1:], target)
            accuracy = calc_accuracy(outputs[-1:].detach().numpy(), 
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
    for i, data in enumerate(dataloader):
        inputs = data['name'][0]
        target = data['sex']
        
        # only train on training names
        if data['name_eng'][0][0] not in test_names:
            continue
        
        outputs = model(inputs)
        
        loss = loss_function(outputs[-1:], target)
        accuracy = calc_accuracy(outputs[-1:].detach().numpy(), 
                             target.detach().numpy())
        
        test_loss.append(loss.detach().numpy())
        test_accuracy.append(accuracy)
    
    print('[{} - TEST] - loss: {:.3f} - accuracy: {:.3f}'.format(epoch,
                                                                     np.mean(test_loss),
                                                                     np.mean(test_accuracy)))




# TESTING ON CUSTOM NAMES

# see if name in database and if M/F/both
name = 'Lily'
print(df[df['name'].str.lower() == name])
with torch.no_grad():
    inputs = dataset.get_custom_name(name)
    outputs = model(torch.tensor(inputs))
    print(torch.softmax(outputs[-1:], dim=1))
    