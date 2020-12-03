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
        # train ng pool = Every item - user's positive item
        positive_item = train_df[train_df['userID']==user]['itemID'].tolist()
        train_ng_item_u = np.setdiff1d(total_item,positive_item).tolist()
        test_ng_item_u = np.setdiff1d(total_item,positive_item).tolist()
        # Test negative = Every item - user's positive item
        test_negative.append(test_ng_item_u)
        train_ng_pool.append(train_ng_item_u)

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

    return train_df, test_df, test_negative, train_ng_pool, num_user, num_item, np.array(t_features), np.array(v_features)

def concat_train_dataset(df, negative, text_feature, visual_feature, num_neg):
    end = time.time()
    negative = np.array(negative)
    temp_df = pd.DataFrame(df[["userID","itemID"]])
    temp_df["negative"] = negative[temp_df["userID"]]
    ng_dataset=[]
    for index, row in temp_df.iterrows():
        user, item_p, ng_pool = int(row["userID"]), int(row["itemID"]), row["negative"]
        item_n = []
        for i in range(num_neg):
            idx = np.random.randint(0,len(ng_pool))
            item_n.append(ng_pool[idx])
        ng_dataset.append(item_n)
            
    feature = np.concatenate((text_feature, visual_feature),axis=1)
    del temp_df["negative"]
    print(f"Time : {time.time()-end:.4f}")
    return temp_df.values.tolist(), ng_dataset, feature


class CustomDataset(Dataset):
    def __init__(self, dataset, feature, negative, istrain=False):
        super(CustomDataset,self).__init__()
        self.dataset = dataset
        self.feature = feature
        self.negative = negative
        self.istrain = istrain

        if not istrain:
            self.dataset = self.make_testset()

    def make_testset(self):
        ng_samples = []
        users = np.unique(self.dataset["userID"])
        for user in users:
            ng_pool = self.negative[user]
            for item in ng_pool:
                ng_samples.append([user,item,0])

        dataset = self.dataset.values.tolist() + ng_samples
        return dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.istrain:
            user, item_p = self.dataset[index]
            item_n = self.negative[index]
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