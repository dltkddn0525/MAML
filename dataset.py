import numpy as np
import pandas as pd
import os
import json
import time
from gensim.models.doc2vec import Doc2Vec
from torch.utils.data import Dataset

def load_data(path):
    train_df = pd.read_csv(os.path.join(path,'train.csv'), index_col=None, usecols=None)
    test_df = pd.read_csv(os.path.join(path,'test.csv'), index_col=None, usecols=None)

    # item ID = 2405 split: train 3 test 2
    item_2405 = test_df[test_df["itemID"]==2405][:3]
    train_df = train_df.append(item_2405)
    test_df = test_df.drop(item_2405.index)

    # Inspect : at least 3 interaction per user for train, 2 per user for test
    #train_x_user, train_x_num_rating = inspect(train_df,3)
    #test_x_user, test_x_num_rating = inspect(test_df,2)
    #print(f"Inspectation : Train = {len(train_x_user)} Users, Test = {len(test_x_user)} Users")

    num_user = max(train_df["userID"])+1
    num_item = max(train_df["itemID"])+1

    end = time.time()
    test_negative = []
    train_ng_pool = []

    total_item = np.arange(0,num_item)

    for user in range(num_user):
        trn_positive_item = train_df[train_df['userID']==user]['itemID'].tolist()
        tst_positive_item = test_df[test_df['userID']==user]['itemID'].tolist()
        # train ng pool = Every item - user's train positive item
        train_ng_item_u = np.setdiff1d(total_item,trn_positive_item)
        # test ng item = Every item - user's train positive item & test positive item
        test_ng_item_u = np.setdiff1d(train_ng_item_u, tst_positive_item)
        train_ng_pool.append(train_ng_item_u.tolist())
        test_negative.append(test_ng_item_u.tolist())

    # train_dataset = train_df.values.tolist()
    test_df = pd.DataFrame(test_df[['userID','itemID']])
    test_df['rating'] = 1

    doc2vec_model = Doc2Vec.load(os.path.join(path, 'doc2vecFile'))
    vis_vec = np.load(os.path.join(path, 'image_feature.npy'),allow_pickle=True).item()

    asin_dict = json.load(open(os.path.join(path, 'asin_sample.json'), 'r'))

    text_vec = {}
    for asin in asin_dict:
        text_vec[asin] = doc2vec_model.docvecs[asin]

    asin_i_dic = {}
    for index, row in train_df.iterrows():
        asin, i = row['asin'], int(row['itemID'])
        asin_i_dic[i] = asin

    t_features = []
    v_features = []
    for i in range(num_item):
        t_features.append(text_vec[asin_i_dic[i]])
        v_features.append(vis_vec[asin_i_dic[i]])

    feature = np.concatenate((t_features,v_features),axis=1)
    train_df = pd.DataFrame(train_df[["userID","itemID"]])

    return train_df, test_df, train_ng_pool, test_negative, num_user, num_item, feature

class CustomDataset(Dataset):
    '''
    Train Batch [user, item_p, item_n, feature_p, feature_n]
    user = [N]
    item_p = [N]
    item_n = [N x num_neg]
    feature_p = [N x (vis_feature_dim + text_feature_dim)]
    featuer_n = [N x num_neg x (vis_feature_dim + text_feature_dim)]

    Test Batch [user, item, feature, label]
    N = number of positive + negative item for corresponding user
    user = [1]
    item = [N]
    feature = [N x (vis_feature_dim + text_feature_dim)]
    label = [N] 1 for positive, 0 for negative
    '''
    def __init__(self, dataset, feature, negative, num_neg = 4, istrain=False):
        super(CustomDataset,self).__init__()
        self.dataset = dataset
        self.feature = feature
        self.negative = negative
        self.istrain = istrain
        self.num_neg = num_neg
        self.train_dataset = None

        if istrain:
            # Something
            self.train_ng_sampling()
        else:
            self.make_testset()

    def train_ng_sampling(self):
        assert self.istrain
        end = time.time()
        print(f"Negative sampling for Train. {self.num_neg} Negative samples per positive pair")

        train_negative = []
        for index, row in self.dataset.iterrows():
            user = int(row["userID"])
            ng_pool = self.negative[user]
            ng_item_u = []
            # Sampling num_neg samples
            for i in range(self.num_neg):
                idx = np.random.randint(0,len(ng_pool))
                ng_item_u.append(ng_pool[idx])
            train_negative.append(ng_item_u)

        self.dataset["negative"] = train_negative
        self.train_dataset = self.dataset.values.tolist()
        print(f"Negative Sampling Complete ({time.time()-end:.4f} sec)")

    def make_testset(self):
        assert not self.istrain
        users = np.unique(self.dataset["userID"])
        test_dataset = []
        for user in users:
            test_negative = self.negative[user]
            test_positive = self.dataset[self.dataset["userID"]==user]["itemID"].tolist()
            item = test_positive + test_negative
            label = np.zeros(len(item))
            label[:len(test_positive)]=1
            label = label.tolist()
            test_user = np.ones(len(item))*user
            test_dataset.append([test_user.tolist(),item, label])

        self.dataset = test_dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.istrain:
            user, item_p, item_n = self.train_dataset[index]
            feature_p = self.feature[item_p]
            feature_n = self.feature[item_n]
            return user, item_p, item_n, feature_p, feature_n
        else:
            user, item, label = self.dataset[index]
            feature = self.feature[item]
            return user, item, feature, label


def inspect(df,num_inter):
    user = np.unique(df["userID"])
    x_user = []
    x_num_rating = []
    for i in user:
        if len(df[df["userID"]==i])<num_inter:
            x_user.append(i)
            x_num_rating.append(len(df[df["userID"]==i]))

    return x_user, x_num_rating