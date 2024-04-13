import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, input_size, hidden_size : list, output_size=1, dropout=0.1):
        super(NN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size[0])
        self.layers = nn.ModuleList([nn.Linear(hidden_size[i-1], hidden_size[i]) for i in range(1, len(hidden_size))])
        self.output_layer = nn.Linear(hidden_size[-1], output_size)
        self.activation1 = nn.Tanh()
        self.activation2 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.ModuleList([nn.BatchNorm1d(hidden_size[i]) for i in range(1, len(hidden_size))])
        self.output_norm = nn.BatchNorm1d(output_size)
    def forward(self, x):
        x = self.activation1(self.input_layer(x))
        for  layer, norm in  zip(self.layers, self.layer_norm):
            x = self.dropout(x)
            x = layer(x) 
            x = self.activation2(x)
            
            x = norm(x)
        
        output = self.output_layer(x)
        output = self.output_norm(output)
        output = self.activation1(output)
        return output
        

