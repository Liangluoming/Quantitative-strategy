import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np


def positional_encoding(max_len, d_model):
    """
    Positional Encoding
    """
    position = torch.arange(0, max_len).unsqueeze(1).float() ## (maxlen) -> (maxlen, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pos_enc = torch.zeros(1, max_len, d_model)
    pos_enc[:, :, 0::2] = torch.sin(position * div_term) ## 偶
    pos_enc[:, :, 1::2] = torch.cos(position * div_term) ## 奇
    return pos_enc

class Time2Vec(nn.Module):
    def __init__(self, time_window, features_num):
        super(Time2Vec, self).__init__()
        self.T = time_window
        self.features_num = features_num
        self.linear_sin = nn.Linear(features_num, 1)
    def forward(self, x):
        for t in range(self.T):
            if t == 0:
                t2v = self.linear_sin(x[:, 0, :]).unsqueeze(1)
            else:
                t2v = torch.cat((t2v, torch.sin(self.linear_sin(x[:, t, :])).unsqueeze(1)), dim = 1)
        return t2v

    def forward_fixparam(self, x):
        x = x.mean(dim=2, keepdim=True) * torch.pi / 29 - torch.pi / 2
        t2v = torch.cat((x[:,0,:].unsqueeze(1), torch.sin(x[:,1:,:])), dim=1)
        return t2v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass of the multi-head attention layer.

        Args:
            query (torch.Tensor): Input tensor for query with shape (batch_size, time_window, feature_nums + 1).
            key (torch.Tensor): Input tensor for key with shape (batch_size, time_window, feature_nums + 1).
            value (torch.Tensor): Input tensor for value with shape (batch_size, time_window, feature_nums + 1).
            mask (torch.Tensor): Mask tensor with shape (batch_size, 1, time_window) or None.

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, time_window, d_model).
        """
        batch_size = query.shape[0]

        # Linear projections
        Q = self.linear_q(query)
        K = self.linear_k(key)
        V = self.linear_v(value)
        
        # Split the heads
        #  view: reshapes the tensor without copying memory 
        # permute: 调整维度位置 (batch_size, self.n_heads, -1, self.head_dim)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.head_dim**0.5

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, V)

        # Concatenate heads
        # contiguous: 使得x中的储存元素是相邻的
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        # Linear projection for the output
        output = self.linear_out(x)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, ff_hid_dim):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, ff_hid_dim)
        self.linear2 = nn.Linear(ff_hid_dim, d_model)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward pass of the feedforward layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, time_window, d_model).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, time_window, d_model).
        """
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.tanh(x)
        return x


class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        """
        Forward pass of the layer normalization.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, time_window, d_model).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, time_window, d_model).
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_hid_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feedforward = FeedForward(d_model, ff_hid_dim)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Forward pass of the encoder layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, time_window , d_model).
            mask (torch.Tensor): Mask tensor with shape (batch_size, 1, time_window) or None.

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, time_window, d_model).
        """
        # Self-Attention
        attention_out = self.self_attention(x, x, x, mask)
        x = x + self.dropout(attention_out)
        x = self.norm1(x)

        # Feedforward
        ff_out = self.feedforward(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, ff_hid_dim, n_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, ff_hid_dim, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Mask tensor with shape (batch_size, 1, seq_len) or None.

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x



class Transformer(nn.Module):
    def __init__(self, time_window, features_num, n_heads, ff_hid_dim, output_dim, n_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.t2vc = Time2Vec(time_window, features_num)
        self.encoder = Encoder(features_num+1, n_heads, ff_hid_dim, n_layers, dropout)
        self.fc_out = nn.Linear(features_num+1, output_dim)
        self.norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.ac = nn.Tanh()
    def forward(self, src, mask):
        """
        Forward pass of the transformer model.

        Args:
            src (torch.Tensor): Input tensor with shape (batch_size, time_window, features_num).
            mask (torch.Tensor): Mask tensor with shape (batch_size, 1, time_window) or None.

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, output_dim).
        """
        x = torch.cat((self.t2vc.forward(src), src), dim = 2)
        # x = torch.cat((self.t2vc.forward_fixparam(src), src), dim = 2)
        # Encoder
        x = self.encoder(x, mask)
        
        x = x[:,-1,:]
        # Output layer
        x = self.fc_out(x)
        x = self.dropout(x)
        x = self.ac(x)
        x = self.norm(x)
        return x
