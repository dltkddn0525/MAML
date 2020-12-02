# User Diverse Preferences Modeling By Multimodal Attentive Metric Learning
Pytorch implementation of [User Diverse Preferences Modeling By Multimodal Attentive Metric Learning(Liu et al., 2019)](https://dl.acm.org/doi/abs/10.1145/3343031.3350953)

## Data preparation
Data should be prepared as follows
- train.csv (Rating data for train)
- test.csv (Rating data for test)
- image_feature.npy (Image features of items extracted from pretrained network)
- doc2vecFile (Parameters of model pretrained with text data of items)

## Usage
```
# Train & Test Top 10 Recommendation
CUDA_VISIBLE_DEVICES=0 python main.py --save_path <Your save path> --data_path <Your data path> --top_k 10
```
The following results will be saved in ```<Your save path>```
- train.log ( epoch, total loss, embedding loss, feature loss, covariance loss )
- test.log ( epoch, hit ratio, nDCG )
- model.pth (model saved every 100 epoch)


## Arguments
| Argument | Type | Description | Default | Paper |
|:---:|:---:|:---:|:---:|:---:|
|save_path|str|Path to save result|'./result'|-|
|data_path|str|Path to dataset|'./Data/Office'|-|
|batch_size|int|Train batch size|1024|-|
|epoch|int|Train epoch|1000|maximum 1000|
|embed_dim|int|Dimension of latent vectors|64|64|
|dropout_rate|float|Dropout rate in feature fusion network|0.2|-|
|lr|float|Learning rate|0.001|0.0001~0.1|
|margin|float|Margin for embedding loss|1.6|1.6|
|feat_weight|float|Weight of feature loss|7|7|
|cov_weight|float|Weight of covariance loss|5|5|
|top_k|int|Top k Recommendation|10|10|
|num_neg|int|Number of negative samples for training|4|4|

## Result






## Reference
[Official Code](https://github.com/liufancs/MAML#user-diverse-preferences-modeling-by-multimodal-attentive-metric-learning)
