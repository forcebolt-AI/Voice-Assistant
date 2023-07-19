import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_1 import bag_of_words, tokenize, stem, tag_parts_of_speech
from model import NeuralNet

# Load intents from JSON file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Load the trained model
data = torch.load("data.pth")

all_words = data['all_words']
tags = data['tags']
input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(data['model_state'])
model.eval()

# Preprocess data
X_test = []
y_test = []
for intent in intents['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        w, pos = tag_parts_of_speech(pattern)
        bag = bag_of_words(w, all_words)
        X_test.append(bag)
        label = tags.index(tag)
        y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

class ChatDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.n_samples = len(x_data)
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

test_dataset = ChatDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Evaluate the model on the test dataset
total_correct = 0
total_samples = 0

for (words, labels) in test_loader:
    words = words.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(words)
    _, predicted = torch.max(outputs.data, 1)

    total_samples += labels.size(0)
    total_correct += (predicted == labels).sum().item()

accuracy = 100 * total_correct / total_samples
print(f'Test Accuracy: {accuracy:.2f}%')
