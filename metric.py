import numpy as np



def get_performance(gt_item, recommends):
    if gt_item in recommends:
        hr = 1
        index = recommends.index(gt_item)
        ndcg = np.reciprocal(np.log2(index+2))
    else:
        hr = 0
        ndcg = 0

    return hr, ndcg