import json

from sympy import Idx
from nltk_utils import tokenize
from nltk_utils import stem
from nltk_utils import bag_Of_words
import numpy as np
import torch
import torch.nn as nn
from  torch.utils.data import dataset, dataloader

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent.get('tag', None)
    if tag is not None:
        tags.append(tag)
    for pattern in intent.get('patterns',[]):
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
Y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_Of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    Y_train.append(label)


X_train=np.array(X_train)
Y_train=np.array(Y_train)

class DUTChatdataset(dataset):
    def __init__(self):
        self.n_sample=len(X_train)
        self.x_data=X_train
        self.y_data=Y_train
   # dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_sample
    
#hyperparameters
batch_size=8
    
dataset=DUTChatdataset()
train_loader = dataloader(dataset, batch_size=8, shuffle=True, num_workers=0)
    


