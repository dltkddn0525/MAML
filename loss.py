import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding_loss(nn.Module):
    def __init__(self, margin):
        super(Embedding_loss,self).__init__()
        self.margin = margin

    def forward(self,dist_p,dist_n):
        dist_p = dist_p.unsqueeze(1)
        loss = self.margin + dist_p**2 - dist_n**2
        loss = F.relu(loss)

        return torch.sum(loss)

class Feature_loss(nn.Module):
    def __init__(self):
        super(Feature_loss,self).__init__()

    def forward(self, q_i, q_i_feature, q_k, q_k_feature):
        id_vec = torch.cat((q_i.unsqueeze(1),q_k),axis=1)
        feature_vec = torch.cat((q_i_feature.unsqueeze(1),q_k_feature),axis=1)
        loss = torch.norm(feature_vec - id_vec, dim=-1, p=2)
        
        return torch.sum(loss)

