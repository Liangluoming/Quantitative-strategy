import torch 
import torch.nn as  nn
import xgboost as xgb

class MultiVarRNN(nn.Module):
    def __init__(self, input_size, rnn_hidden_size, ff_hidden_size, output_size, num_layers, dropout=0.1):
        super(MultiVarRNN, self).__init__()
        ## input = batch_size * feature_num
        self.rnn_hidden_size = rnn_hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, rnn_hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, ff_hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.ac1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(ff_hidden_size)
        self.outputfc = nn.Linear(rnn_hidden_size, output_size)
        self.dropout2 = nn.Dropout(dropout)
        self.ac2 = nn.Tanh()
        self.bn2 = nn.BatchNorm1d(output_size)

    def forward(self, x):
        ## 初始化隐藏层状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.rnn_hidden_size).to(x.device)

        ## 向前传播RNN
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        # ff_out = self.bn1(self.ac1(self.dropout1(self.fc(out))))
        output = self.bn2(self.ac2(self.dropout2(self.outputfc(out))))
        return output