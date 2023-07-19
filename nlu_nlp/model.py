import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
    
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Example RNN (LSTM) model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.rnn(x)
        output = self.fc(hidden[-1])
        return output

# Example Transformer model
class TransformerModel(nn.Module):
    def __init__(self, pretrained_model_name, output_size):
        super(TransformerModel, self).__init__()
        self.encoder = BertModel.from_pretrained(pretrained_model_name)
        self.fc = nn.Linear(self.encoder.config.hidden_size, output_size)

    def forward(self, x):
        _, pooled_output = self.encoder(x)
        output = self.fc(pooled_output)
        return output
    
    
import math    
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
class TransformerNeuralNet(nn.Module):  
    def __init__(self, input_size, hidden_size, num_classes, num_layers, num_heads, dropout):
        super(TransformerNeuralNet, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        encoder_layer = TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_size = hidden_size

    def forward(self, src):
        src_embedding = self.embedding(src) * math.sqrt(self.hidden_size)
        src_embedding = self.positional_encoding(src_embedding)
        src_padding_mask = self.generate_padding_mask(src)
        src_key_padding_mask = self.generate_key_padding_mask(src)
        memory = self.encoder(src_embedding, src_key_padding_mask=src_key_padding_mask)
        output = self.fc(memory)
        return output

    def generate_padding_mask(self, x):
        return (x == 0).transpose(0, 1)

    def generate_key_padding_mask(self, x):
        return (x == 0).transpose(0, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

