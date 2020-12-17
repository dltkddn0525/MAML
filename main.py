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
parser.add_argument('--epoch', default=1, type=int,
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
parser.add_argument('--load_path', default='./', type=str,
                     help='Path to saved model')


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
    path = args.data_path
    train_df, test_df, train_ng_pool, test_negative, num_user, num_item, feature = D.load_data(path)
    train_dataset = D.CustomDataset(train_df, feature, negative=train_ng_pool, num_neg = args.num_neg, istrain=True)
    test_dataset = D.CustomDataset(test_df, feature, negative=test_negative, num_neg = None, istrain=False)
    # test_dataset = D.CustomDataset(train_df, feature, negative=train_ng_pool, num_neg = None, istrain=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=my_collate_trn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=my_collate_tst)

    # Model
    model = MAML(num_user, num_item, args.embed_dim, args.dropout_rate).cuda()
    print(model)
    if args.load_path is not None:
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint)
        print("Pretrained Model Loaded")
        
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
        train_loader.dataset.train_ng_sampling()
        train(model, embedding_loss, feature_loss, covariance_loss, optimizer, train_loader, train_logger, epoch)
        if (epoch+1) % 100 == 0 or epoch==0:
            test(model, test_loader, test_logger, epoch)
            # Save Model every 100 epoch
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

        # L2 reg for attention layer
        linear1 = model.attention[1].parameters()
        linear2 = model.attention[5].parameters()
        reg1 = l2_regularization(linear1, 100.0)
        reg2 = l2_regularization(linear2, 100.0)

        # loss = loss_e + args.feat_weight * loss_f + args.cov_weight * loss_c
        loss = loss_e + (args.feat_weight * loss_f) + (args.cov_weight * loss_c) + reg1 + reg2

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
    hr = AverageMeter()
    ndcg = AverageMeter()
    end = time.time()
    for i, (user, item, feature, label) in enumerate(test_loader):
        with torch.no_grad():
            user, item, feature, label = user.squeeze(0), item.squeeze(0), feature.squeeze(0), label.squeeze(0)
            user, item, feature, label = user.cuda(), item.cuda(), feature.cuda(), label.cuda()
            _, _, _, score = model(user, item, feature)

            pos_idx = label.nonzero()
            _, indices = torch.topk(-score, args.top_k)
            recommends = torch.take(item, indices).cpu().numpy()
            gt_item = item[pos_idx].cpu().numpy()
            performance = get_performance(gt_item, recommends)
            hr.update(performance[0])
            ndcg.update(performance[1])

            if i % 100 == 0:
                print(f"{i+1} Users tested.")
    
    print(f"Epoch : [{epoch+1}/{args.epoch}] Hit Ratio : {hr.avg:.4f} nDCG : {ndcg.avg:.4f} Test Time : {time.time()-end:.4f}")
    test_logger.write([epoch, hr.avg, ndcg.avg])
    
def l2_regularization(params, _lambda):
    l2_reg = torch.cat([x.view(-1) for x in params])
    l2_reg = _lambda * torch.norm(l2_reg,2)
    return l2_reg


def my_collate_trn(batch):
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

def my_collate_tst(batch):
    user = [items[0] for items in batch]
    user = torch.LongTensor(user)
    item = [items[1] for items in batch]
    item = torch.LongTensor(item)
    feature = [items[2] for items in batch]
    feature = torch.Tensor(feature)
    label = [items[3] for items in batch]
    label = torch.Tensor(label)
    return [user, item, feature, label]

if __name__ == "__main__":
    main()
