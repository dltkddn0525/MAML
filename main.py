import torch
import argparse
import json
import time
import os
from torch.utils.data import DataLoader
import numpy as np

from utils import Logger, AverageMeter
from model import MAML
from loss import Embedding_loss, Feature_loss, Covariance_loss
import dataset as D
from metric import get_performance

parser = argparse.ArgumentParser(description='MAML')
parser.add_argument('--save_path', default='./result', type=str,
                     help='savepath')
parser.add_argument('--batch_size', default=1024, type=int,
                     help='batch size')
parser.add_argument('--epoch', default=1000, type=int,
                     help='train epoch')
parser.add_argument('--data_path', default='./Data/Office', type=str,
                     help='Path to dataset')
parser.add_argument('--embed_dim', default=64, type=int,
                     help='Embedding Dimension')
parser.add_argument('--dropout_rate', default=0.2, type=float,
                     help='Dropout rate')
parser.add_argument('--lr', default=0.001, type=float,
                     help='Learning rate')
parser.add_argument('--margin', default=1.6, type=float,
                     help='Margin for embedding loss')
parser.add_argument('--feat_weight', default=7, type=float,
                     help='Weight of feature loss')
parser.add_argument('--cov_weight', default=5, type=float,
                     help='Weight of covariance loss')
parser.add_argument('--top_k', default=10, type=int,
                     help='Top k Recommendation')
parser.add_argument('--num_neg', default=4, type=int,
                     help='Number of negative samples for training')

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
    Train Batch [user, item_p, item_n, feature_p, feature_n]
    user = [N]
    item_p = [N]
    item_n = [N x num_neg]
    feature_p = [N x (vis_feature_dim + text_feature_dim)]
    featuer_n = [N x num_neg x (vis_feature_dim + text_feature_dim)]

    Test Batch [user, item, feature]
    user = [N]
    item = [N]
    feature = [N x (vis_feature_dim + text_feature_dim)]
    '''
    path = args.data_path
    train_dataset, test_dataset, test_negative, train_ng_pool, num_user, num_item, t_features, v_features = D.load_data(path)
    train_dataset, train_negative, feature = D.concat_train_dataset(train_dataset, train_ng_pool, t_features, v_features, args.num_neg)
    train_dataset = D.CustomDataset(train_dataset, feature, negative=train_negative, istrain=True)
    test_dataset = D.CustomDataset(test_dataset, feature, negative=test_negative, istrain=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


    # Model
    model = MAML(num_user, num_item, args.embed_dim, args.dropout_rate).cuda()
        
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    
    # Loss 
    embedding_loss = Embedding_loss(margin=args.margin, num_item = num_item).cuda()
    feature_loss = Feature_loss().cuda()
    covariance_loss = Covariance_loss().cuda()

    # Logger
    train_logger = Logger(f'{save_path}/train.log')
    test_logger = Logger(f'{save_path}/test.log')

    # Train & Eval
    for epoch in range(args.epoch):
        train(model, embedding_loss, feature_loss, covariance_loss, optimizer, train_loader, train_logger, epoch)
        # Evaluate Every 100 epoch
        if (epoch+1) % 100 == 0:
            test(model, test_loader, test_logger, epoch)
            # Save Model
            torch.save(model.state_dict(), f"{save_path}/model_{epoch+1}.pth")
        
            

def train(model, embedding_loss, feature_loss, covariance_loss, optimizer, train_loader, train_logger, epoch):
    model.train()
    total_loss = AverageMeter()
    embed_loss = AverageMeter()
    feat_loss = AverageMeter()
    cov_loss = AverageMeter()
    data_time = AverageMeter()
    iter_time = AverageMeter()

    end = time.time()
    for i, (user, item_p, item_n, feature_p, feature_n) in enumerate(train_loader):
        data_time.update(time.time()-end)
        user, item_p, item_n, feature_p, feature_n = user.cuda(), item_p.cuda(), item_n.cuda(), feature_p.cuda(), feature_n.cuda()

        p_u, q_i, q_i_feature, dist_p = model(user, item_p, feature_p)
        _, q_k, q_k_feature, dist_n = model(user, item_n, feature_n)
        # Loss
        loss_e = embedding_loss(dist_p, dist_n)
        loss_f = feature_loss(q_i, q_i_feature, q_k, q_k_feature)
        loss_c = covariance_loss(p_u, q_i, q_k)
        loss = loss_e + args.feat_weight * loss_f + args.cov_weight * loss_c

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.update(loss.item())
        embed_loss.update(loss_e.item())
        feat_loss.update(loss_f.item())
        cov_loss.update(loss_c.item())
        iter_time.update(time.time()-end)
        end = time.time()

        if i % 10 == 0:
            print(f"[{epoch+1}/{args.epoch}][{i}/{len(train_loader)}] Total loss : {total_loss.avg:.4f} \
                Embedding loss : {embed_loss.avg:.4f} Feature loss : {feat_loss.avg:.4f} \
                Covariance loss : {cov_loss.avg:.4f} Iter time : {iter_time.avg:.4f} Data time : {data_time.avg:.4f}")

    train_logger.write([epoch, total_loss.avg, embed_loss.avg,
                        feat_loss.avg, cov_loss.avg])

def test(model, test_loader, test_logger, epoch):
    model.eval()
    
    end = time.time()
    for i, (user, item, feature, label) in enumerate(test_loader):
        with torch.no_grad():
            user, item, feature, label = user.cuda(), item.cuda(), feature.cuda(), label.cuda()

            _, _, _, score = model(user, item, feature)
            pos_idx = label.nonzero()
            neg_idx = (label==0).nonzero()

            positive_batch = torch.cat((user[pos_idx],item[pos_idx],score[pos_idx]),axis=1)
            negative_batch = torch.cat((user[neg_idx],item[neg_idx],score[neg_idx]),axis=1)

            if i == 0:
                positive = positive_batch
                negative = negative_batch
            else:
                positive = torch.cat((positive,positive_batch),axis=0)
                negative = torch.cat((negative,negative_batch),axis=0)
        if i % 1000 == 0 :
            print(f"[{i}/{len(test_loader)}] Iteration Processed")

    hr, nDCG = eval_recommend(positive.cpu(), negative.cpu(), top_k=args.top_k)
    test_logger.write([epoch, hr, nDCG])
    
def eval_recommend(positive, negative, top_k):
    hr = []
    nDCG = []
    for i in range(len(positive)):
        user = positive[i][0]
        temp = torch.cat((positive[i].unsqueeze(0), negative[negative[:,0]==user]),axis=0)
        score = -temp[:,2]
        _, indices = torch.topk(score, top_k)
        recommends = torch.take(temp[:,1], indices).numpy().tolist()
        gt_item = temp[0,1].item()
        performace = get_performance(gt_item, recommends)
        hr.append(performace[0])
        nDCG.append(performace[1])
    
    return sum(hr)/len(hr), sum(nDCG)/len(nDCG)


def my_collate(batch):
    user = [item[0]for item in batch]
    user = torch.LongTensor(user)
    item_p = [item[1] for item in batch]
    item_p = torch.LongTensor(item_p)
    item_n = [item[2] for item in batch]
    item_n = torch.LongTensor(item_n)
    feature_p = [item[3] for item in batch]
    feature_p = torch.Tensor(feature_p)
    feature_n = [item[4] for item in batch]
    feature_n = torch.Tensor(feature_n)
    return [user, item_p, item_n, feature_p, feature_n]

if __name__ == "__main__":
    main()
