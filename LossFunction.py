import torch 
import torch.nn as nn 

class NegativeICLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(NegativeICLoss, self).__init__()
        self.eps = eps

    def forward(self, X, Y):
        # assert not torch.any(torch.isnan(X))
        # assert not torch.any(torch.isnan(Y))
        X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + self.eps) 
        Y = (Y - Y.mean(dim=0, keepdim=True)) / (Y.std(dim=0, keepdim=True) + self.eps) 
        Z = X * Y
        loss =  - Z.mean()
        return loss



class FrobeniusPenalty(nn.Module):
    """
    参照东方证券(2023):《基于循环神经网络多频率因子挖掘》研报构建的惩罚项
    """
    def __init__(self, C):
        super(FrobeniusPenalty, self).__init__()
        self.penalty = C
    def forward(self, X):
        return self.penalty * torch.norm(torch.mul(X, X), 'fro').item() / (X.size(-1)**2 * X.size(0))

class NegativeRankICLoss(nn.Module):
    def __init__(self):
        super(NegativeRankICLoss, self).__init__()

    def forward(self, X, Y):
        # assert not torch.any(torch.isnan(X))
        # assert not torch.any(torch.isnan(Y))
        x_rank = torch.argsort(X,dim=0)
        y_rank = torch.argsort(Y,dim=0)
        d = x_rank - y_rank

        n = len(X)
        r = 1 - 6 * torch.sum(d * d) / (n * (n * n - 1))
        loss = -r
        return loss