import torch
import torch.nn as nn
import torch.nn.functional as F

class MAML(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, dropout_rate):
        super(MAML,self).__init__()
        self.embed_dim = embed_dim
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        

        # Embedding Layers
        self.embedding_user = nn.Embedding(n_users, embed_dim, max_norm=1.0)
        self.embedding_item = nn.Embedding(n_items, embed_dim, max_norm=1.0)

        # Feature Fusion Layers
        """
        input dim = 4096+512 = 4608
        final output dim = embed_dim
        """
        input_dim = 4608
        hidden_dim = 256
        modules = []
        for i in range(3):
            if i == 2:
                hidden_dim *= 2
            modules.append(NormalizeLayer())
            modules.append(nn.Linear(input_dim, hidden_dim))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
            hidden_dim //= 2
        modules.append(nn.Linear(hidden_dim*2, embed_dim))
        modules.append(NormalizeLayer())
        
        self.feature_fusion = nn.Sequential(*modules)
        
        # Attention
        self.attention = nn.Sequential(
            NormalizeLayer(),
            nn.Linear(3*embed_dim, 3*embed_dim),
            nn.Tanh(),
            nn.Dropout(0.05),
            NormalizeLayer(),
            nn.Linear(3*embed_dim, embed_dim),
            nn.ReLU()
        )
        
    def forward(self,user,item, item_feature):
        # Embed user, item
        p_u = self.embedding_user(user)
        q_i = self.embedding_item(item)

        # Fused feature
        q_i_feature = self.feature_fusion(item_feature)

        # Attention
        if p_u.shape != q_i.shape:
            # unsqueeze
            p_u = p_u.unsqueeze(1).expand(-1,q_i.shape[1],-1)
        input_cat = torch.cat((p_u,q_i,q_i_feature),axis=-1)
        attention = self.attention(input_cat)
        # attention = self.embed_dim * modified_softmax(attention)
        attention = self.embed_dim * F.softmax(attention, dim=-1)
        
        # Distance
        # dist = torch.norm(torch.mul(attention, p_u)-torch.mul(attention,q_i),dim=-1, p=2)
        temp = torch.mul(attention, p_u)-torch.mul(attention,q_i)
        dist = torch.sum(temp**2,axis=-1)
        
        return p_u, q_i, q_i_feature, dist






class NormalizeLayer(nn.Module):
    def __init__(self):
        super(NormalizeLayer, self).__init__()
    def forward(self, input):
        return F.normalize(input, dim=-1, p=2)

def modified_softmax(x):
    exp = torch.exp(x)
    if len(exp.shape)>2:
        sum = torch.sum(x,axis=-1).reshape(x.shape[0],x.shape[1],-1)
    else:
        sum = torch.sum(x,axis=-1).reshape(x.shape[0],-1)
    return exp/sum