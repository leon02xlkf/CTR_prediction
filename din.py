# 导入deepctr
from deepctr.models import DIN
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import * 
import tensorflow as tf

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 数据准备函数
def get_din_feats_columns(df, dense_fea, sparse_fea, behavior_fea, his_behavior_fea, emb_dim=32, max_len=100):
    """
    数据准备函数:
    df: 数据集
    dense_fea: 数值型特征列
    sparse_fea: 离散型特征列
    behavior_fea: 用户的候选行为特征列
    his_behavior_fea: 用户的历史行为特征列
    embedding_dim: embedding的维度， 这里为了简单， 统一把离散型特征列采用一样的隐向量维度
    max_len: 用户序列的最大长度
    """
    
    sparse_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].nunique() + 1, embedding_dim=emb_dim) for feat in sparse_fea]
    
    dense_feature_columns = [DenseFeat(feat, 1, ) for feat in dense_fea]
    
    var_feature_columns = [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=df['click_article_id'].nunique() + 1,
                                    embedding_dim=emb_dim, embedding_name='click_article_id'), maxlen=max_len) for feat in hist_behavior_fea]
    
    dnn_feature_columns = sparse_feature_columns + dense_feature_columns + var_feature_columns
    
    # 建立x, x是一个字典的形式
    x = {}
    for name in get_feature_names(dnn_feature_columns):
        if name in his_behavior_fea:
            # 这是历史行为序列
            his_list = [l for l in df[name]]
            x[name] = pad_sequences(his_list, maxlen=max_len, padding='post')      # 二维数组
        else:
            x[name] = df[name].values
    
    return x, dnn_feature_columns

# 把特征分开
sparse_fea = ['user_id', 'click_article_id', 'category_id', 'click_environment', 'click_deviceGroup', 
              'click_os', 'click_country', 'click_region', 'click_referrer_type', 'is_cat_hab']

behavior_fea = ['click_article_id']

hist_behavior_fea = ['hist_click_article_id']

dense_fea = ['sim0', 'time_diff0', 'word_diff0', 'sim_max', 'sim_min', 'sim_sum', 'sim_mean', 'score',
             'rank','click_size','time_diff_mean','active_level','user_time_hob1','user_time_hob2',
             'words_hbo','words_count']

# dense特征进行归一化, 神经网络训练都需要将数值进行归一化处理
mm = MinMaxScaler()

# 下面是做一些特殊处理，当在其他的地方出现无效值的时候，不处理无法进行归一化，刚开始可以先把他注释掉，在运行了下面的代码
# 之后如果发现报错，应该先去想办法处理如何不出现inf之类的值
# trn_user_item_feats_df_din_model.replace([np.inf, -np.inf], 0, inplace=True)
# tst_user_item_feats_df_din_model.replace([np.inf, -np.inf], 0, inplace=True)

for feat in dense_fea:
    trn_user_item_feats_df_din_model[feat] = mm.fit_transform(trn_user_item_feats_df_din_model[[feat]])
    
    if val_user_item_feats_df_din_model is not None:
        val_user_item_feats_df_din_model[feat] = mm.fit_transform(val_user_item_feats_df_din_model[[feat]])
    
    tst_user_item_feats_df_din_model[feat] = mm.fit_transform(tst_user_item_feats_df_din_model[[feat]])

# 准备训练数据
x_trn, dnn_feature_columns = get_din_feats_columns(trn_user_item_feats_df_din_model, dense_fea, 
                                               sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)
y_trn = trn_user_item_feats_df_din_model['label'].values

if offline:
    # 准备验证数据
    x_val, dnn_feature_columns = get_din_feats_columns(val_user_item_feats_df_din_model, dense_fea, 
                                                   sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)
    y_val = val_user_item_feats_df_din_model['label'].values
    
dense_fea = [x for x in dense_fea if x != 'label']
x_tst, dnn_feature_columns = get_din_feats_columns(tst_user_item_feats_df_din_model, dense_fea, 
                                               sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)

# 建立模型
model = DIN(dnn_feature_columns, behavior_fea)

# 查看模型结构
model.summary()

# 模型编译
model.compile('adam', 'binary_crossentropy',metrics=['binary_crossentropy', tf.keras.metrics.AUC()])

# 模型训练
if offline:
    history = model.fit(x_trn, y_trn, verbose=1, epochs=10, validation_data=(x_val, y_val) , batch_size=256)
else:
    # 也可以使用上面的语句用自己采样出来的验证集
    # history = model.fit(x_trn, y_trn, verbose=1, epochs=3, validation_split=0.3, batch_size=256)
    history = model.fit(x_trn, y_trn, verbose=1, epochs=2, batch_size=256)

# 模型预测
tst_user_item_feats_df_din_model['pred_score'] = model.predict(x_tst, verbose=1, batch_size=256)
tst_user_item_feats_df_din_model[['user_id', 'click_article_id', 'pred_score']].to_csv(save_path + 'din_rank_score.csv', index=False)

# 预测结果重新排序, 及生成提交结果
rank_results = tst_user_item_feats_df_din_model[['user_id', 'click_article_id', 'pred_score']]
submit(rank_results, topk=5, model_name='din')

# 五折交叉验证，这里的五折交叉是以用户为目标进行五折划分
#  这一部分与前面的单独训练和验证是分开的
def get_kfold_users(trn_df, n=5):
    user_ids = trn_df['user_id'].unique()
    user_set = [user_ids[i::n] for i in range(n)]
    return user_set

k_fold = 5
trn_df = trn_user_item_feats_df_din_model
user_set = get_kfold_users(trn_df, n=k_fold)

score_list = []
score_df = trn_df[['user_id', 'click_article_id', 'label']]
sub_preds = np.zeros(tst_user_item_feats_df_rank_model.shape[0])

dense_fea = [x for x in dense_fea if x != 'label']
x_tst, dnn_feature_columns = get_din_feats_columns(tst_user_item_feats_df_din_model, dense_fea, 
                                                   sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)

# 五折交叉验证，并将中间结果保存用于staking
for n_fold, valid_user in enumerate(user_set):
    train_idx = trn_df[~trn_df['user_id'].isin(valid_user)] # add slide user
    valid_idx = trn_df[trn_df['user_id'].isin(valid_user)]
    
    # 准备训练数据
    x_trn, dnn_feature_columns = get_din_feats_columns(train_idx, dense_fea, 
                                                       sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)
    y_trn = train_idx['label'].values

    # 准备验证数据
    x_val, dnn_feature_columns = get_din_feats_columns(valid_idx, dense_fea, 
                                                   sparse_fea, behavior_fea, hist_behavior_fea, max_len=50)
    y_val = valid_idx['label'].values
    
    history = model.fit(x_trn, y_trn, verbose=1, epochs=2, validation_data=(x_val, y_val) , batch_size=256)
    
    # 预测验证集结果
    valid_idx['pred_score'] = model.predict(x_val, verbose=1, batch_size=256)   
    
    valid_idx.sort_values(by=['user_id', 'pred_score'])
    valid_idx['pred_rank'] = valid_idx.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')
    
    # 将验证集的预测结果放到一个列表中，后面进行拼接
    score_list.append(valid_idx[['user_id', 'click_article_id', 'pred_score', 'pred_rank']])
    
    # 如果是线上测试，需要计算每次交叉验证的结果相加，最后求平均
    if not offline:
        sub_preds += model.predict(x_tst, verbose=1, batch_size=256)[:, 0]   
    
score_df_ = pd.concat(score_list, axis=0)
score_df = score_df.merge(score_df_, how='left', on=['user_id', 'click_article_id'])
# 保存训练集交叉验证产生的新特征
score_df[['user_id', 'click_article_id', 'pred_score', 'pred_rank', 'label']].to_csv(save_path + 'trn_din_cls_feats.csv', index=False)
    
# 测试集的预测结果，多次交叉验证求平均,将预测的score和对应的rank特征保存，可以用于后面的staking，这里还可以构造其他更多的特征
tst_user_item_feats_df_din_model['pred_score'] = sub_preds / k_fold
tst_user_item_feats_df_din_model['pred_score'] = tst_user_item_feats_df_din_model['pred_score'].transform(lambda x: norm_sim(x))
tst_user_item_feats_df_din_model.sort_values(by=['user_id', 'pred_score'])
tst_user_item_feats_df_din_model['pred_rank'] = tst_user_item_feats_df_din_model.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

# 保存测试集交叉验证的新特征
tst_user_item_feats_df_din_model[['user_id', 'click_article_id', 'pred_score', 'pred_rank']].to_csv(save_path + 'tst_din_cls_feats.csv', index=False)

# 读取多个模型的排序结果文件
lgb_ranker = pd.read_csv(save_path + 'lgb_ranker_score.csv')
lgb_cls = pd.read_csv(save_path + 'lgb_cls_score.csv')
din_ranker = pd.read_csv(save_path + 'din_rank_score.csv')

# 这里也可以换成交叉验证输出的测试结果进行加权融合

rank_model = {'lgb_ranker': lgb_ranker, 
              'lgb_cls': lgb_cls, 
              'din_ranker': din_ranker}

def get_ensumble_predict_topk(rank_model, topk=5):
    final_recall = rank_model['lgb_cls'].append(rank_model['din_ranker'])
    rank_model['lgb_ranker']['pred_score'] = rank_model['lgb_ranker']['pred_score'].transform(lambda x: norm_sim(x))
    
    final_recall = final_recall.append(rank_model['lgb_ranker'])
    final_recall = final_recall.groupby(['user_id', 'click_article_id'])['pred_score'].sum().reset_index()
    
    submit(final_recall, topk=topk, model_name='ensemble_fuse')

get_ensumble_predict_topk(rank_model)

# 读取多个模型的交叉验证生成的结果文件
# 训练集
trn_lgb_ranker_feats = pd.read_csv(save_path + 'trn_lgb_ranker_feats.csv')
trn_lgb_cls_feats = pd.read_csv(save_path + 'trn_lgb_cls_feats.csv')
trn_din_cls_feats = pd.read_csv(save_path + 'trn_din_cls_feats.csv')

# 测试集
tst_lgb_ranker_feats = pd.read_csv(save_path + 'tst_lgb_ranker_feats.csv')
tst_lgb_cls_feats = pd.read_csv(save_path + 'tst_lgb_cls_feats.csv')
tst_din_cls_feats = pd.read_csv(save_path + 'tst_din_cls_feats.csv')

# 将多个模型输出的特征进行拼接

finall_trn_ranker_feats = trn_lgb_ranker_feats[['user_id', 'click_article_id', 'label']]
finall_tst_ranker_feats = tst_lgb_ranker_feats[['user_id', 'click_article_id']]

for idx, trn_model in enumerate([trn_lgb_ranker_feats, trn_lgb_cls_feats, trn_din_cls_feats]):
    for feat in [ 'pred_score', 'pred_rank']:
        col_name = feat + '_' + str(idx)
        finall_trn_ranker_feats[col_name] = trn_model[feat]

for idx, tst_model in enumerate([tst_lgb_ranker_feats, tst_lgb_cls_feats, tst_din_cls_feats]):
    for feat in [ 'pred_score', 'pred_rank']:
        col_name = feat + '_' + str(idx)
        finall_tst_ranker_feats[col_name] = tst_model[feat]

# 定义一个逻辑回归模型再次拟合交叉验证产生的特征对测试集进行预测
# 这里需要注意的是，在做交叉验证的时候可以构造多一些与输出预测值相关的特征，来丰富这里简单模型的特征
from sklearn.linear_model import LogisticRegression

feat_cols = ['pred_score_0', 'pred_rank_0', 'pred_score_1', 'pred_rank_1', 'pred_score_2', 'pred_rank_2']

trn_x = finall_trn_ranker_feats[feat_cols]
trn_y = finall_trn_ranker_feats['label']

tst_x = finall_tst_ranker_feats[feat_cols]

# 定义模型
lr = LogisticRegression()

# 模型训练
lr.fit(trn_x, trn_y)

# 模型预测
finall_tst_ranker_feats['pred_score'] = lr.predict_proba(tst_x)[:, 1]

# 预测结果重新排序, 及生成提交结果
rank_results = finall_tst_ranker_feats[['user_id', 'click_article_id', 'pred_score']]
submit(rank_results, topk=5, model_name='ensumble_staking')
