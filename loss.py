import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding_loss(nn.Module):
    def __init__(self, margin, num_item):
        super(Embedding_loss,self).__init__()
        self.margin = margin
        self.num_item = num_item

    def forward(self,dist_p,dist_n):
        closest_dist_n, indices = torch.min(dist_n,axis=1)
        # dist_p = dist_p.unsqueeze(1)
        # loss = self.margin + dist_p**2 - dist_n**2
        loss = self.margin + dist_p - closest_dist_n
        loss = F.relu(loss)

        # num_imposter = loss>0
        num_imposter = self.margin + dist_p.unsqueeze(1) - dist_n
        num_imposter = num_imposter > 0
        num_imposter = torch.sum(num_imposter,axis=1)
        # num_imposter = num_imposter.unsqueeze(1).float()
        num_imposter = num_imposter.float()
        rank = num_imposter * self.num_item / dist_n.shape[1]
        weight = torch.log(rank+1)
        weight.requires_grad_(False)

        loss *= weight
        # return torch.mean(loss)
        return torch.sum(loss)

class Feature_loss(nn.Module):
    def __init__(self):
        super(Feature_loss,self).__init__()

    def forward(self, q_i, q_i_feature, q_k, q_k_feature):
        id_vec = torch.cat((q_i.unsqueeze(1),q_k),axis=1)
        feature_vec = torch.cat((q_i_feature.unsqueeze(1),q_k_feature),axis=1)
        # loss = torch.norm(feature_vec - id_vec, dim=-1, p=2)
        loss = (feature_vec - id_vec)**2
        # return torch.mean(loss)
        return torch.sum(loss)

class Covariance_loss(nn.Module):
    def __init__(self):
        super(Covariance_loss,self).__init__()

    def forward(self, p_u, q_i, q_k):
        q_k = q_k.reshape(q_k.shape[0]*q_k.shape[1],-1)
        cov = torch.cat((p_u,q_i,q_k),axis=0)
        num_row = cov.shape[0]
        mean = torch.mean(cov,axis=0)
        cov = cov - mean
        cov = torch.matmul(cov.t(),cov) / num_row
        loss = torch.norm(cov)**2 - torch.norm(cov.diag())**2

        return loss

