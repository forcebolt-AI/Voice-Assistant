import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_1 import bag_of_words, tokenize, stem, tag_parts_of_speech
from model import NeuralNet

# Load the existing model from the .pth file
# existing_model_file = "data.pth"
# existing_model_data = torch.load(existing_model_file)

json_files = ['data/data_tolokers.json', 'data/data_volunteers.json']

all_words = []
tags = []
xy = []

# Process the dialog data
for json_file in json_files:
    with open(json_file, 'r') as f:
        dialog_data = json.load(f)

    for dialog in dialog_data:
        for message in dialog['dialog']:
            sender_class = message['sender_class']
            text = message['text']
            if sender_class == 'Bot':
                # Preprocess and tokenize the input from the Bot
                words, pos = tag_parts_of_speech(text)
                words = [stem(word) for word in words]
                all_words.extend(words)
                tags.append(sender_class)  # Add the tag to the list
                xy.append((words, pos, sender_class))
            else:
                # Preprocess and tokenize the input from the Human
                words, pos = tag_parts_of_speech(text)
                words = [stem(word) for word in words]
                all_words.extend(words)
                tags.append(sender_class)  # Add the tag to the list
                xy.append((words, pos, sender_class))

# Stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# Remove duplicates and sort
all_words = sorted(set(all_words))
# Remove duplicates and sort
tags = sorted(set(tags))

# Create training data
X_train = []
y_train = []
for (pattern_sentence, pos, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Create the dataset and data loader
dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model instance
model = NeuralNet(input_size, hidden_size, output_size).to(device)
#model.load_state_dict(existing_model_data["model_state"])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(words)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Training complete. Final loss: {loss.item():.4f}')

# Save the trained model to the existing .pth file
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}
FILE = "data1.pth"
torch.save(data, FILE)
print(f'Trained model saved to {FILE}')
