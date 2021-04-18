#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 23:11:29 2021

@author: yesh
"""

import os
import datetime
#os.chdir('/Users/yesh/Documents/ent_ai/name_classifier')
import numpy as np
import pandas as pd
import time
import pickle

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from model import LSTMClassifier, DatasetNames


# Currently: seeing if new code does as well as old code with batch_size =1
# TODO:
#   - trying adding these things: https://github.com/claravania/lstm-pytorch/blob/master/model.py


# PARAMS
DATADIR = './data'


# 1. Load data
# a. Data from: https://www.ssa.gov/oact/babynames/limits.html
data_fps = [os.path.join(DATADIR,file) for file in os.listdir(DATADIR) if file.endswith('.txt')]

colnames = ['name', 'sex', 'freq']
df = pd.DataFrame(columns=colnames)
for fp in data_fps:
    df_sub = pd.read_csv(fp, header=None)
    df_sub.columns = colnames
    df = df.append(df_sub)
    
# - convert sex to binary
df['sex'] = np.where(df['sex'] == 'M', 1, 0)
    

# b. add names from World Gender Name Dataset
# https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/YPRQH8/SO6SXA&version=1.1
data_fp = '/world-gender-name-dictionary/wgnd_source.csv'
df_w = pd.read_csv(DATADIR+data_fp)
df_w = df_w.rename(columns={'gender': 'sex'})

# drop missing names and gender
df_w = df_w.dropna(subset=['name', 'sex'])
df_w = df_w[df_w['sex'] != '?']

# - convert sex to binary
df_w['sex'] = np.where(df_w['sex'] == 'M', 1, 0)


# - remove chinese characters
df_w = df_w[~df_w['name'].str.contains(r'[^\x00-\x7F]+', '')]


len(np.unique(df_w['name']))

# c combined the name databases
df = df.append(df_w[['name', 'sex']])

# - lowercase names
df['name'] = df['name'].str.lower()

# - drop repeats
df = df.groupby(['name', 'sex'], as_index=False).agg('sum')


# - remove double white spaces
def remove_double_ws(words):
    return " ".join(words.split())
df['name'] = df.apply(lambda x: remove_double_ws(x['name']), axis=1)

# - count how many male only and how many female only
female_names = df[df['sex'] == 0]['name'].values
male_names = df[df['sex'] == 1]['name'].values
both_names = np.intersect1d(female_names, male_names)
female_names = set(both_names) ^ set(female_names)
male_names =  set(both_names) ^ set(male_names)


print('{} ({:.2f}%) names are both male and female'.format(len(both_names),
                                                       100*len(both_names)/len(np.unique(df['name']))))

# - save male and female names for use later
pd.DataFrame({'name': list(male_names)}).to_csv('./male_names.csv', index=False)
pd.DataFrame({'name': list(female_names)}).to_csv('./female_names.csv', index=False)
pd.DataFrame({'name': list(both_names)}).to_csv('./mixed_mf_names.csv', index=False)

# - exclude names that are both male/female
df = df[~df['name'].isin(both_names)]

# - train/val/test split
train_names, val_names = train_test_split(np.unique(df['name']),
                               test_size=0.1, random_state=1)
val_names, test_names = train_test_split(val_names, test_size=0.5, random_state=1)
    
    
    
# Dataloaders
dataset_train = DatasetNames(df[df['name'].isin(train_names)]['name'].values,
                       df[df['name'].isin(train_names)]['sex'].values)
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
    
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
params = {'embedding_dim': 192,
            'hidden_dim': 128, 
            'letter_size': len(dataset_train.uniq_letters), 
            'n_classes': 2, 
            'padding_idx': dataset_train.letter_to_index['0'],
            'num_layers': 4}
model = LSTMClassifier(**params)

loss_function = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# Save model
dt = datetime.datetime.today()
save_path_weights = './models/model_{}-{}-{}_{}.{}.{}.pth'.format(dt.year, dt.month, dt.day, 
                                                                  dt.hour, dt.minute, dt.second)
save_path_l2i = './models/l2i_{}-{}-{}_{}.{}.{}.pkl'.format(dt.year, dt.month, dt.day, 
                                                                  dt.hour, dt.minute, dt.second)
save_path_params = './models/params_{}-{}-{}_{}.{}.{}.pkl'.format(dt.year, dt.month, dt.day, 
                                                                  dt.hour, dt.minute, dt.second)


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

# SAVE weights and letter-to-index used
torch.save(model.state_dict(), save_path_weights)
with open(save_path_l2i, 'wb') as handle:
    pickle.dump(dataset_train.letter_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(save_path_params, 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
name = 'penge'
print(df[df['name'].str.lower() == name.lower()])
with torch.no_grad():
    inputs = dataset_train.get_custom_name(name)
    outputs = model(torch.tensor([inputs]))
    print(torch.softmax(outputs, dim=2)[:,-1])
    
    
    


    
    
    
    
    
    
    
    

    
