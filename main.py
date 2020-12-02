import torch
import argparse
import json
import time
import os
from torch.utils.data import DataLoader

from utils import Logger, AverageMeter
from model import MAML
from loss import Embedding_loss, Feature_loss
import dataset as D


parser = argparse.ArgumentParser(description='MAML')
parser.add_argument('--save_path', default='./result', type=str,
                     help='savepath')
parser.add_argument('--batch_size', default=128, type=int,
                     help='batch size')
parser.add_argument('--epoch', default=100, type=int,
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
    train_dataset, train_negative, feature = D.concat_train_dataset(train_dataset, train_ng_pool, t_features, v_features)
    train_dataset = D.CustomDataset(train_dataset, feature, negative=train_negative, istrain=True)
    test_dataset = D.CustomDataset(test_dataset, feature, negative=test_negative, istrain=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


    # Model
    model = MAML(num_user, num_item, args.embed_dim, args.dropout_rate).cuda()
        
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    
    # Loss 
    embedding_loss = Embedding_loss(margin=args.margin).cuda()
    feature_loss = Feature_loss().cuda()
    # + Covariance Loss

    # Logger
    train_logger = Logger(f'{save_path}/train.log')
    test_logger = Logger(f'{save_path}/test.log')

    # Train & Eval
    for epoch in range(args.epoch):
        train(model, embedding_loss, feature_loss, optimizer, train_loader, train_logger, epoch)

def train(model, embedding_loss, feature_loss, optimizer, train_loader, train_logger, epoch):
    model.train()
    total_loss = AverageMeter()
    embed_loss = AverageMeter()
    feat_loss = AverageMeter()
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
        loss = loss_e + 7 * loss_f

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.update(loss.item())
        embed_loss.update(loss_e.item())
        feat_loss.update(loss_f.item())
        iter_time.update(time.time()-end)
        end = time.time()

        if i % 10 == 0:
            print(f"[{epoch+1}/{args.epoch}][{i}/{len(train_loader)}] Total loss : {total_loss.avg:.4f} \
                Embedding loss : {embed_loss.avg:.4f} Feature loss : {feat_loss.avg:.4f} \
                Iter time : {iter_time.avg:.4f} Data time : {data_time.avg:.4f}")

    train_logger.write([epoch, total_loss.avg, embed_loss.avg,
                        feat_loss.avg])

def test(model, test_loader, test_logger, epoch):
    model.eval()
    

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
