import numpy as np


def get_performance(gt_item, recommends):
    recommends = recommends.tolist()
    # Hit ratio & DCG
    hr = 0
    dcg = 0.0
    for item in gt_item:
        if item in recommends:
            hr += 1
            index = recommends.index(item)
            dcg += np.reciprocal(np.log2(index+2))
    
    if hr>0:
        hr = 1
    else:
        hr = 0

    # nDCG
    idcg = 0.0
    for i in range(len(gt_item)):
        idcg += np.reciprocal(np.log2(i+2))
        
    ndcg = dcg / idcg

    return hr, ndcg