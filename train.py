import json

from sympy import Idx
from nltk_utils import tokenize
from nltk_utils import stem
from nltk_utils import bag_Of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

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
  
class DUTChatdataset(Dataset):
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
hidden_size=8
output_size=len(tags)
input_size=len(X_train[0])
learning_rate=0.001
num_epochs=1000

if __name__ == "__main__":
    dataset = DUTChatdataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'final loss: {loss.item():.4f}')
    data={
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size":output_size,
        "hiddent_size": hidden_size,
        "all_word": all_words,
        "tags": tags
    }

    file="data.pth"
    torch.save(data, file)

    print(f'training complete. file saved to {file}')




   