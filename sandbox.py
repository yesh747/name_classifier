#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 21:30:46 2021

@author: yesh
"""

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence

#input
sexes = [1, 0, 1, 0]
names = ['John', 'Jill', 'Jeffrey', 'Susan']


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
    
    
dataset = DatasetNames(names, sexes)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for i, data in enumerate(dataloader):
    print(data)

    




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
    
    
    
model = LSTMClassifier(embedding_dim=12, hidden_dim=12, 
               letter_size=len(dataset.uniq_letters), 
               n_classes=2)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


for epoch in range(300):
    for i, data in enumerate(dataloader):
        print(data)
        inputs = data['name'][0]
        target = data['sex']
        outputs = model(inputs)
        
        loss = loss_function(outputs[-1:], target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)

with torch.no_grad():
    for i, data in enumerate(dataloader):
        print(data)
        inputs = data['name'][0]
        target = data['sex']
        outputs = model(inputs)
        print(outputs[-1:])


    
# TUTORIALS

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    # Tags are: DET - determiner; NN - noun; V - verb
    # For example, the word "The" is a determiner
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
# For each words-list (sentence) and tags-list in each tuple of training_data
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}  # Assign each tag with a unique index

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6 
    
    
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
    

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)


for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    
    
    
    
    
    
    
    
    
    
    
    


