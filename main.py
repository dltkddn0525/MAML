import torch
import argparse
import json
import time
import os

from utils import Logger, AverageMeter
from model import MAML
from loss import Embedding_loss, Feature_loss

parser = argparse.ArgumentParser(description='MAML')
parser.add_argument('--save_path', default='./result', type=str,
                     help='savepath')

args = parser.parse_args()


def main():
    # Set save path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save configuration
    with open(save_path+'/configuration.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Load dataset
    '''
    userID = [N]
    positive itemID = [N]
    positive item feature = [N,4096+512]
    negative itemID = [N,num_neg]
    negative item feature = [N,num_neg,4096+512]
    '''

    # Model
    n_users = 100
    n_items = 120
    embed_dim = 64
    dropout_rate = 0.2
    model = MAML(n_users, n_items, embed_dim, dropout_rate)
    
    ### Sample input
    batch_size = 2
    num_neg = 4
    sample_user = torch.randint(0,10,(batch_size,))
    sample_item_p = torch.randint(0,10,(batch_size,))
    sample_item_feature_p = torch.randn(batch_size,4608)
    sample_item_n = torch.randint(0,10,(batch_size,num_neg))
    sample_item_feature_n = torch.randn(batch_size,num_neg,4608)
    p_u, q_i, q_i_feature, dist_p = model(sample_user, sample_item_p,sample_item_feature_p)
    p_u, q_k, q_k_feature, dist_n = model(sample_user, sample_item_n,sample_item_feature_n)
    #####
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    
    # Loss 
    embedding_loss = Embedding_loss(0.2)
    feature_loss = Feature_loss()

    # Logger
    train_logger = Logger(f'{save_path}/train.log')
    test_logger = Logger(f'{save_path}/test.log')

    # Train & Eval


if __name__ == "__main__":
    main()
