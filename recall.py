import pandas as pd  
import numpy as np
from tqdm import tqdm  
from collections import defaultdict  
import os, math, warnings, math, pickle
from tqdm import tqdm
import faiss
import collections
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss
warnings.filterwarnings('ignore')

data_path = './data_raw/'
save_path = './temp_results/'
# 做召回评估的一个标志, 如果不进行评估就是直接使用全量数据进行召回
metric_recall = False

'''
读取数据
'''
# 比赛中分为三种模式，分别对应不同的数据集：
# 1. debug模式：train_click_log_sample.csv
# 基于数据先搭建一个简易的baseline跑通，保证代码没有很大问题
# 由于比赛数据非常巨大，一上来直接采用全部的数据进行分析，会造成时间和设备损耗
# 从海量数据的训练集中随机抽取一部分样本调试
# 2. 线下验证模式：train_click_log.csv
# 在线下基于已有的训练集数据，选择合适的模型和一些超参数
# 需要加载整个训练集，再把整个训练集分为训练集（训练数据）和验证集（调整模型参数）
# 3. 线上模式：train_click_log.csv + testA_click_log.csv
# 对于给定的测试集进行预测，提交到线上，使用全量的数据集

# 读取点击日志，按照三种模式
# debug模式: 
def get_all_click_sample(data_path, sample_nums=10000):
    """
        训练集中采样一部分数据调试
        data_path: 原数据的存储路径
        sample_nums: 采样数目（这里由于机器的内存限制，可以采样用户做）
    """
    all_click = pd.read_csv(data_path + 'train_click_log.csv')
    all_user_ids = all_click.user_id.unique()
    # 随机选择sample_nums个用户，默认10000
    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False) 
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]
    # 去重
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click

# 线上模式与线下模式：
# 如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集
def get_all_click_df(data_path='./data_raw/', offline=True):
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
        all_click = trn_click.append(tst_click)
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click


# 读取文章的基本属性
def get_item_info_df(data_path):
    item_info_df = pd.read_csv(data_path + 'articles.csv')
    # 为了方便与训练集中的click_article_id拼接，需要把article_id修改成click_article_id
    item_info_df = item_info_df.rename(columns={'article_id': 'click_article_id'})
    return item_info_df


# 读取文章的Embedding数据
def get_item_emb_dict(data_path):
    item_emb_df = pd.read_csv(data_path + 'articles_emb.csv')
    # 提取向量，使用numpy的array储存
    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols])
    # 进行归一化（向量模长为1，统一尺度便于后续计算余弦相似度）
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)
    # 将article_id与向量一一对应，保存成字典
    item_emb_dict = dict(zip(item_emb_df['article_id'], item_emb_np))
    pickle.dump(item_emb_dict, open(save_path + 'item_content_emb.pkl', 'wb'))
    return item_emb_dict


# 归一化函数，将数据归一化到[0,1]之间
max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
# 采样数据
# all_click_df = get_all_click_sample(data_path)
# 全量训练集，用于线上模式
all_click_df = get_all_click_df(offline=False)
# 对时间戳进行归一化,用于在关联规则的时候计算权重（做规则/协同过滤召回时加权）
all_click_df['click_timestamp'] = all_click_df[['click_timestamp']].apply(max_min_scaler)
# 获取文章的基本属性（提供类别，字数等属性特征）
item_info_df = get_item_info_df(data_path)
# 获取文章的Embedding数据（用于基于内容做召回时的相似度计算）
item_emb_dict = get_item_emb_dict(data_path)



'''
工具函数
'''
# 获取用户-文章-时间函数
# 根据点击时间获取用户的点击文章序列   {user1: {item1: time1, item2: time2..}...}
# 在基于关联规则的UserCF召回中，需要使用到(也可以用于序列模型训练数据构建，或时间衰减建模)
def get_user_item_time(click_df):
    click_df = click_df.sort_values('click_timestamp') # 按照时间戳排序
    def make_item_time_pair(df): 
        return list(zip(df['click_article_id'], df['click_timestamp']))
    user_item_time_df = click_df.groupby('user_id')['click_article_id', 'click_timestamp'].apply(lambda x: make_item_time_pair(x))\
                                                            .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    return user_item_time_dict

# 获取文章-用户-时间函数
# 根据时间获取商品被点击的用户序列  {item1: {user1: time1, user2: time2...}...}
# 在基于关联规则的ItemCF召回中，需要使用到
def get_item_user_time_dict(click_df):
    def make_user_time_pair(df):
        return list(zip(df['user_id'], df['click_timestamp']))
    click_df = click_df.sort_values('click_timestamp')
    item_user_time_df = click_df.groupby('click_article_id')['user_id', 'click_timestamp'].apply(lambda x: make_user_time_pair(x))\
                                                            .reset_index().rename(columns={0: 'user_time_list'})
    item_user_time_dict = dict(zip(item_user_time_df['click_article_id'], item_user_time_df['user_time_list']))
    return item_user_time_dict

# 获取当前数据的历史点击和最后一次点击
# 将每个用户的点击数据分为历史点击和最后一次点击
# 评估召回结果，特征工程和制作标签转化成监督学习测试集时会用到
def get_hist_and_last_click(all_click):
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1) # 最后一次点击

    def hist_func(user_df):
        if len(user_df) == 1: # 如果用户只有一条点击记录，直接返回（避免训练时用户缺失）
            return user_df
        else:
            return user_df[:-1] # 否则返回除了最后一次点击的历史点击记录
    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)
    return click_hist_df, click_last_df

# 获取文章属性特征
# 获取文章id对应的基本属性，保存成字典的形式，方便后面召回阶段，冷启动阶段直接使用
def get_item_info_dict(item_info_df):
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x)) # 归一化
    item_info_df['created_at_ts'] = item_info_df[['created_at_ts']].apply(max_min_scaler)
    item_type_dict = dict(zip(item_info_df['click_article_id'], item_info_df['category_id']))
    item_words_dict = dict(zip(item_info_df['click_article_id'], item_info_df['words_count']))
    item_created_time_dict = dict(zip(item_info_df['click_article_id'], item_info_df['created_at_ts']))
    return item_type_dict, item_words_dict, item_created_time_dict


# 获取用户历史点击的文章信息
def get_user_hist_item_info_dict(all_click):
    # 获取user_id对应的用户历史点击文章类型的集合字典 {user1: {101, 103}, user2: {102}, ...}
    user_hist_item_typs = all_click.groupby('user_id')['category_id'].agg(set).reset_index()
    user_hist_item_typs_dict = dict(zip(user_hist_item_typs['user_id'], user_hist_item_typs['category_id']))
    
    # 获取user_id对应的用户点击文章的集合 {user1: {234, 2891, 45}, user2: {87, 91}, ...}
    user_hist_item_ids_dict = all_click.groupby('user_id')['click_article_id'].agg(set).reset_index()
    user_hist_item_ids_dict = dict(zip(user_hist_item_ids_dict['user_id'], user_hist_item_ids_dict['click_article_id']))
    
    # 获取user_id对应的用户历史点击的文章的平均字数字典 {user1: 456.3, user2: 873.0, ...}
    user_hist_item_words = all_click.groupby('user_id')['words_count'].agg('mean').reset_index()
    user_hist_item_words_dict = dict(zip(user_hist_item_words['user_id'], user_hist_item_words['words_count']))
    
    # 获取user_id对应的用户最后一次点击的文章的创建时间（归一化，转成字典）
    all_click_ = all_click.sort_values('click_timestamp')
    user_last_item_created_time = all_click_.groupby('user_id')['created_at_ts'].apply(lambda x: x.iloc[-1]).reset_index()
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    user_last_item_created_time['created_at_ts'] = user_last_item_created_time[['created_at_ts']].apply(max_min_scaler)
    # {user1: 0.83, user2: 0.11, ...}
    user_last_item_created_time_dict = dict(zip(user_last_item_created_time['user_id'], \
                                                user_last_item_created_time['created_at_ts']))
    
    return user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict


# 获取点击次数最多的topk文章
# 获取近期点击最多的文章
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click


# 定义多路召回字典
# 定义一个多路召回的字典，将各路召回的结果都保存在这个字典当中
user_multi_recall_dict =  {'itemcf_sim_itemcf_recall': {},
                           'embedding_sim_item_recall': {},
                           'youtubednn_recall': {},
                           'youtubednn_usercf_recall': {}, 
                           'cold_start_recall': {}}
# 提取最后一次点击作为召回评估（召回后对当前的召回方法调整以达到更好的召回效果，召回的结果决定的最终排序的上限）
# 如果不需要做召回评估直接使用全量的训练集进行召回(线下验证模型)，不用单独提取最后一次点击的数据
trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)

# 召回效果评估
# 依次评估召回的前10, 20, 30, 40, 50个文章中的击中率
# 即：召回的topk个结果中，有没有用户真实点击的文章（使用最后一次点击做评估）
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=5):
    # 获取用户最后一次点击的文章id {user1: 234, user2: 87, ...}
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    user_num = len(user_recall_items_dict)
    
    for k in range(10, topk+1, 10):
        hit_num = 0
        for user, item_list in user_recall_items_dict.items():
            # 获取前k个召回的结果
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1 # 统计命中数
        # 计算每个topk召回的命中率 = 命中用户数 / 用户总数
        hit_rate = round(hit_num * 1.0 / user_num, 5)
        print(' topk: ', k, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', hit_rate, 'user_num : ', user_num)



'''
计算相似性矩阵
'''
# ItemCF i2i_sim
# 根据KDD2020去偏商品推荐，计算相似性矩阵时使用关联规则，计算文章的相似性时还考虑了：
# 用户点击的时间权重，用户点击的顺序权重，用户创建的时间权重
def itemcf_sim(df, item_created_time_dict):
    """
        文章与文章之间的相似性矩阵计算
        :param df: 数据表
        :item_created_time_dict:  文章创建时间的字典
        return : 文章与文章的相似性矩阵
    """
    
    user_item_time_dict = get_user_item_time(df)
    
    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()): # 遍历每个用户的点击序列
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list): # 对用户每次点击的文章，与其余所有文章做成对组合
                if(i == j):
                    continue
                # 构造权重因子
                # 考虑文章的正向顺序点击和反向顺序点击（若j在i后点击，权重越高，因为i和j可能存在潜在因果关系）   
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                # 位置信息权重（j在i后越近的位置被点击，权重越高）
                loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                # 点击时间权重（i和j的点击时间越接近，权重越高）
                click_time_weight = np.exp(0.7 ** np.abs(i_click_time - j_click_time))
                # 两篇文章创建时间的权重（两个文章发布时间接近，倾向于视为更相似（热点or实时内容））
                created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
                i2i_sim[i].setdefault(j, 0)
                # 考虑多种因素的权重计算相似度，加入log惩罚活跃用户
                i2i_sim[i][j] += loc_weight * click_time_weight * created_time_weight / math.log(len(item_time_list) + 1)
                
    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j]) # 余弦相似度，归一化
    
    # 将得到的相似性矩阵保存到本地
    pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))
    return i2i_sim_

i2i_sim = itemcf_sim(all_click_df, item_created_time_dict)

# 根据点击次数计算用户活跃度
def get_user_activate_degree_dict(all_click_df):
    # 按照用户id分组
    all_click_df_ = all_click_df.groupby('user_id')['click_article_id'].count().reset_index()
    mm = MinMaxScaler()
    # 把活跃度归一化到[0,1]之间，并使用字典储存
    # {user_1: 0.82, user_2: 0.12, ...}
    all_click_df_['click_article_id'] = mm.fit_transform(all_click_df_[['click_article_id']])
    user_activate_degree_dict = dict(zip(all_click_df_['user_id'], all_click_df_['click_article_id']))
    return user_activate_degree_dict

# UserCF u2u_sim
# 根据用户活跃度计算用户相似性矩阵
# 使用简单的关联规则计算权重，这里使用用户的点击次数作为用户活跃度的指标
def usercf_sim(all_click_df, user_activate_degree_dict):
    """
        用户相似性矩阵计算
        :param all_click_df: 数据表
        :param user_activate_degree_dict: 用户活跃度的字典
        return 用户相似性矩阵
    """
    item_user_time_dict = get_item_user_time_dict(all_click_df)
    
    u2u_sim = {}
    user_cnt = defaultdict(int)
    for item, user_time_list in tqdm(item_user_time_dict.items()):
        for u, click_time in user_time_list:
            user_cnt[u] += 1
            u2u_sim.setdefault(u, {})
            for v, click_time in user_time_list:
                u2u_sim[u].setdefault(v, 0)
                if u == v:
                    continue
                # 用户平均活跃度作为活跃度的权重
                # 两个用户如果都比较活跃，那么他们对同一个item的点击共现，就更有代表性，更值得记录在相似度矩阵中
                activate_weight = 100 * 0.5 * (user_activate_degree_dict[u] + user_activate_degree_dict[v])   
                u2u_sim[u][v] += activate_weight / math.log(len(user_time_list) + 1)
    
    u2u_sim_ = u2u_sim.copy()
    for u, related_users in u2u_sim.items():
        for v, wij in related_users.items():
            u2u_sim_[u][v] = wij / math.sqrt(user_cnt[u] * user_cnt[v])
    # 将得到的相似性矩阵保存到本地
    pickle.dump(u2u_sim_, open(save_path + 'usercf_u2u_sim.pkl', 'wb'))
    return u2u_sim_

# 由于usercf计算时候太耗费内存了，这里就不直接运行了
# 如果是采样的话，是可以运行的
user_activate_degree_dict = get_user_activate_degree_dict(all_click_df)
u2u_sim = usercf_sim(all_click_df, user_activate_degree_dict)


# 向量检索相似度计算
# item embedding sim
# user和item在实际场景中，都是海量数据，无法直接使用两层循环遍历计算相似度矩阵
# faiss是Facebook的ai开源的向量检索库，使用PCA和PQ（乘积量化）进行向量压缩和编码
# 对每个item, faiss搜索后返回最相似的topk个item，在后续冷启动时，可以获取未出现在点击数据中的文章
def embdding_sim(click_df, item_emb_df, save_path, topk):
    """
        基于内容的文章embedding相似性矩阵计算
        :param click_df: 数据表
        :param item_emb_df: 文章的embedding
        :param save_path: 保存路径
        :patam topk: 找最相似的topk篇
        return 文章相似性矩阵
        思路: 对于每一篇文章， 基于embedding的相似性返回topk个与其最相似的文章， 只不过由于文章数量太多，这里用了faiss进行加速
    """
    
    # 文章索引与文章id的字典映射
    item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df['article_id']))
    
    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
    # 向量进行单位化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)
    
    # 建立faiss索引
    item_index = faiss.IndexFlatIP(item_emb_np.shape[1])
    item_index.add(item_emb_np)
    # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
    sim, idx = item_index.search(item_emb_np, topk) # 返回的是列表
    
    # 将向量检索的结果保存成原始id的对应关系
    item_sim_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(item_emb_np)), sim, idx)):
        target_raw_id = item_idx_2_rawid_dict[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]): 
            rele_raw_id = item_idx_2_rawid_dict[rele_idx]
            item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value
    
    # 保存i2i相似度矩阵
    pickle.dump(item_sim_dict, open(save_path + 'emb_i2i_sim.pkl', 'wb'))   
    
    return item_sim_dict

item_emb_df = pd.read_csv(data_path + '/articles_emb.csv')
emb_i2i_sim = embdding_sim(all_click_df, item_emb_df, save_path, topk=10) # topk可以自行设置


'''
召回阶段
'''
# 方法：
# 1. YouTube DNN召回
# 2. 基于文章的召回（Item CF/embedding）
# 3. 基于用户的召回（User CF/embedding）

'''
YouTube DNN召回（双塔）
'''
# 获取双塔召回时的训练验证数据
def gen_data_set(data, negsample=0):
    data.sort_values("click_timestamp", inplace=True) # 按照时间排序点击序列
    item_ids = data['click_article_id'].unique() # 获取去重所有文章id

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')): # 遍历每个用户
        pos_list = hist['click_article_id'].tolist() # 获取用户点击的文章序列
        
        # negsample指的是通过滑窗构建样本的时候，指定负样本的数量
        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))   # 用户没看过的文章里面选择负样本
            # 对于每个正样本，通过随机抽取的方式选择negsample个负样本
            neg_list = np.random.choice(candidate_set,size=len(pos_list)*negsample,replace=True)
            
        # 长度只有一个的时候，无法通过滑窗构造训练样本（因为滑窗需要至少两个样本）
        # 需要把这条数据也放到训练集中，不然的话最终学到的embedding就会有缺失
        if len(pos_list) == 1:
            train_set.append((reviewerID, [pos_list[0]], pos_list[0],1,len(pos_list)))
            test_set.append((reviewerID, [pos_list[0]], pos_list[0],1,len(pos_list)))
            
        # 滑窗构造正负样本
        for i in range(1, len(pos_list)):
            # 前i个点击作为历史序列hist，第i+1个作为通过前面序列预测的目标，这组值作为训练集中的一个正样本
            # 如：hist = [1,2,3], pos_list[i] = 4，反转hist是使得最近的点击在前面
            hist = pos_list[:i]
            
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1])))  # 正样本 [user_id, his_item, pos_item, label（正1负0）, len(his_item)]
                for negi in range(negsample): # 对于每个正样本，选择negsample个负样本，negi表示这个正样本对应下的负样本索引
                    train_set.append((reviewerID, hist[::-1], neg_list[i*negsample+negi], 0,len(hist[::-1]))) # 负样本 [user_id, his_item, neg_item, label, len(his_item)]
            else:
                # 将最后点击的物品（含历史点击日志对应最长的样本hist序列）作为测试数据
                test_set.append((reviewerID, hist[::-1], pos_list[i],1,len(hist[::-1])))
                
    random.shuffle(train_set)
    random.shuffle(test_set)
    
    return train_set, test_set

# 将输入的数据进行padding，使得序列特征的长度都一致
def gen_model_input(train_set,user_profile,seq_max_len):
    # 对于每个样本序列，提取出用户id，点击序列，目标物品，标签，序列长度
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    # 将序列特征进行padding，在较短序列特征的末尾填充0，使得序列特征的长度都一致
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {"user_id": train_uid, "click_article_id": train_iid, "hist_article_id": train_seq_pad,
                         "hist_len": train_hist_len}
    return train_model_input, train_label

# 使用DNN对每个用户召回topk个物品（主函数）
def youtubednn_u2i_dict(data, topk=20):    
    sparse_features = ["click_article_id", "user_id"] # 需要做embedding的两列名称
    SEQ_LEN = 30 # 用户点击序列的长度，短的填充，长的截断
    
    # ID去重，提取user和item的画像
    # 这次提取是为了把id编码成整数
    user_profile_ = data[["user_id"]].drop_duplicates('user_id')
    item_profile_ = data[["click_article_id"]].drop_duplicates('click_article_id')  
    # 类别编码
    features = ["click_article_id", "user_id"]
    feature_max_idx = {}
    for feature in features: # feature：item_id, user_id
        lbe = LabelEncoder() # 将原始id编码，转化为0-n-1的整数
        data[feature] = lbe.fit_transform(data[feature]) # 将转换的结果写回data[feature]
        feature_max_idx[feature] = data[feature].max() + 1 # 获取每个特征的最大值+1（该特征有多少个不同的值），作为该特征的embedding维度
    
    # ID去重，再次提取user和item的画像（这里是编码后的值，即0-n-1的整数）
    # 这里具体选择哪些特征还需要进一步的分析和考虑
    user_profile = data[["user_id"]].drop_duplicates('user_id')
    item_profile = data[["click_article_id"]].drop_duplicates('click_article_id')  
    # 建立一个字典，映射{编码后的整数id：原始id}
    user_index_2_rawid = dict(zip(user_profile['user_id'], user_profile_['user_id']))
    item_index_2_rawid = dict(zip(item_profile['click_article_id'], item_profile_['click_article_id']))
    
    # 划分训练和测试集
    # 由于深度学习需要的数据量通常都是非常大的，所以为了保证召回的效果，往往会通过滑窗的形式扩充训练样本
    # 这一步中negsample=0，不划分正负样本，只通过滑窗切分训练集（1 - n-1次点击作为正样本）和测试集（最后一次点击）
    train_set, test_set = gen_data_set(data, 0)
    
    # 整理输入数据
    # *_model_input：将样本中每个特征单独提取出来建立字典
    # *_label：对应的正负样本标签向量（1/0）
    # 补0至SEQ_LEN这么长
    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)
    
    # 确定Embedding的维度
    embedding_dim = 16
    
    # 将数据整理成模型可以直接输入的形式（Embedding层）
    # 用户塔特征列
    # user_id: 一个标量，做标准sparsefeature
    # hist_article_id: 一个可变长度的序列，做VarLenSparseFeat封装，内部embedding名称叫click_article_id，池化方式mean
    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
                            VarLenSparseFeat(SparseFeat('hist_article_id', feature_max_idx['click_article_id'], embedding_dim,
                                                        embedding_name="click_article_id"), SEQ_LEN, 'mean', 'hist_len'),]
    # 物品塔特征列
    # click_article_id: 一个标量，表示当前的候选文章，做标准sparsefeature
    item_feature_columns = [SparseFeat('click_article_id', feature_max_idx['click_article_id'], embedding_dim)]
    

    '''
    SparseFeature：任何单一值的离散特征都可以用sparsefeat表示，比如用户id，物品id，性别，城市编码等
    以user为例：首先提供一个特征名（user_id）
    然后提供不同的取值个数（feature_max_idx['user_id']）
    最后提供embedding的维度（embedding_dim）
    随后，每个用户id都会被转换成一个embedding_dim维的向量
    
    VarLenSparseFeat：任何用户行为序列或多值列表类型的离散特征都可以用VarLenSparseFeat表示
    比如用户最近看过的文章id列表，用户喜欢的标签集合，浏览过的商品列表等等
    在内部，sparsefeature也是一个标准的sparsefeature，将每个取值转换成embedding_dim维的向量
    VarLenSparseFeat需要提供一个序列长度（SEQ_LEN），如果序列长度小于这个值，则会在末尾填充0
    如果序列长度大于这个值，则将超出的部分截断
    每个序列都会被转换成一个embedding_dim * SEQ_LEN的变长矩阵（如文章列表取最近30个，则包含30个embedding_dim的向量）
    对这一系列向量进行池化（这里取了平均值），得到一个embedding_dim维的固定向量
    
    hist_len：一个标量，表示用户点击序列的长度（DIN中用作注意力权重，DIEN中用作时间衰减因子，用于mask或注意力正则化等）

    '''


    # 双塔模型的定义 
    # num_sampled: 负采样时的样本数量
    # 这里sampled softmax损失里每个正样本对应5个负样本（这样就不用对全量的物品做完整的softmax运算）
    # TODO：user_dnn_hidden_units: 第一层隐藏单元数64，第二层输出层单元数为embedding_dim（此处保持物品塔和用户塔的维度一致，方便后续使用点积计算相似度）
    # 隐藏层之间一般使用ReLu激活（默认）
    model = YoutubeDNN(user_feature_columns, item_feature_columns, num_sampled=5, user_dnn_hidden_units=(64, embedding_dim))
    # 模型编译
    # TODO：使用adam优化器自适应优化学习率，自动调整每条参数的学习速率，加速收敛并提高鲁棒性（啥意思）
    # sampledsoftmaxloss：按照题设比例正负样本1:5参与softmax运算，把正样本得分拉高
    model.compile(optimizer="adam", loss=sampledsoftmaxloss)  
    
    # 模型训练，这里可以定义验证集的比例validation_split，如果设置为0的话就是全量数据直接进行训练
    # batch_size：批量大小，每次训练时抽取的样本数量
    # epochs：训练的轮数，一个epoch表示使用训练集中的所有样本训练一次
    # verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
    history = model.fit(train_model_input, train_label, batch_size=256, epochs=1, verbose=1, validation_split=0.0)
    
    # TODO：这部分看不懂
    # 训练完模型之后,提取训练的Embedding，包括user端和item端
    test_user_model_input = test_model_input # 用测试集的用户计算用户的embedding
    all_item_model_input = {"click_article_id": item_profile['click_article_id'].values} # 用所有item计算item的embedding
    # 从训练好的keras模型中提取user和item的embedding
    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)
    # 保存当前的item_embedding 和 user_embedding 排序的时候可能能够用到，但是需要注意保存的时候需要和原始的id对应
    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)
    
    # embedding保存之前归一化一下（L2范数保证长度为1，方便后续使用相似度直接检索）
    user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
    item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)
    
    # 将编码前user和item的原始id映射到embedding向量
    raw_user_id_emb_dict = {user_index_2_rawid[k]: \
                                v for k, v in zip(user_profile['user_id'], user_embs)}
    raw_item_id_emb_dict = {item_index_2_rawid[k]: \
                                v for k, v in zip(item_profile['click_article_id'], item_embs)}
    # 将Embedding保存到本地，后续在排序过程中可以直接加载使用
    pickle.dump(raw_user_id_emb_dict, open(save_path + 'user_youtube_emb.pkl', 'wb'))
    pickle.dump(raw_item_id_emb_dict, open(save_path + 'item_youtube_emb.pkl', 'wb'))
    
    # faiss紧邻搜索，通过user_embedding 搜索与其相似性最高的topk个item
    index = faiss.IndexFlatIP(embedding_dim) # 使用内积作为相似度度量
    # 上面已经进行了归一化，这里可以不进行归一化了
    # faiss.normalize_L2(user_embs)
    # faiss.normalize_L2(item_embs)
    index.add(item_embs) # 将item向量构建索引
    # 使用所有item的embedding向量，通过user的embedding向量，检索出topk个相似的item
    # 返回sim：相似度得分矩阵，idx：相似item在item_embs中的索引
    sim, idx = index.search(np.ascontiguousarray(user_embs), topk) # 通过user去查询最相似的topk个item
    
    # 遍历每个用户
    user_recall_items_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(test_user_model_input['user_id'], sim, idx)):
        target_raw_id = user_index_2_rawid[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]): 
            rele_raw_id = item_index_2_rawid[rele_idx]
            user_recall_items_dict[target_raw_id][rele_raw_id] = user_recall_items_dict.get(target_raw_id, {})\
                                                                    .get(rele_raw_id, 0) + sim_value
        # 建立字典：{user_id: {item_id: score, item_id: score, ...}, user_id: {item_id: score, item_id: score, ...}, ...}
            
    user_recall_items_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True) for k, v in user_recall_items_dict.items()}
    # 按照降序排列召回列表
    
    # 保存召回的结果
    # 这里是直接通过向量的方式得到了召回结果，相比于上面的召回方法，上面的只是得到了i2i及u2u的相似性矩阵，还需要进行协同过滤召回才能得到召回结果
    # 可以直接对这个召回结果进行评估，为了方便可以统一写一个评估函数对所有的召回结果进行评估
    pickle.dump(user_recall_items_dict, open(save_path + 'youtube_u2i_dict.pkl', 'wb'))
    return user_recall_items_dict
    # 保存召回结果，后续可以直接使用pickle.load()加载，不用重新跑模型

# 根据metric_recall的值，决定是否进行召回效果评估
# 线上和线下模式复用一套代码
# 不做评估（线上模式）：直接使用全量点击日志召回
if not metric_recall:
    user_multi_recall_dict['youtubednn_recall'] = youtubednn_u2i_dict(all_click_df, topk=20)
# 进行召回效果评估（线下模式）：
# train_hist_click_df：训练集中的历史点击 -> 调用youtubednn_u2i_dict()得到只基于历史的召回
# train_last_click_df：训练集中的最后一次点击
else:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
    user_multi_recall_dict['youtubednn_recall'] = youtubednn_u2i_dict(trn_hist_click_df, topk=20)
    # 召回效果评估, 调用上面的函数，根据真实点击的文章输出命中率
    metrics_recall(user_multi_recall_dict['youtubednn_recall'], trn_last_click_df, topk=20)



'''
Item CF召回
'''
# 基于商品的召回i2i
# 关联规则：比较相似文章与历史点击文章的：
# 1. 点击顺序
# 2. 创建时间差
# 3. 内容相似性（使用embedding计算，但embedding时没计算所有商品两两的相似度（相似文章与历史点击文章不存在相似度），需要特殊处理）
def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列 {user1: {item1: time1, item2: time2..}...}
        -> 可以计算位置权重和创建时间差
        :param i2i_sim: 字典，文章相似性矩阵
        :param sim_item_topk: 整数， 选择与当前文章最相似的前k篇文章
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        :param emb_i2i_sim: 字典基于内容embedding算的文章相似矩阵
        
        return: 召回的文章列表 {item1: score1, item2: score2...}
        
    """
    # 获取用户历史交互的文章
    # 准备历史点击日志和输出容器
    user_hist_items = user_item_time_dict[user_id]
    item_rank = {}

    # 遍历用户历史点击文章
    for loc, (i, click_time) in enumerate(user_hist_items):
        # 遍历与当前文章相似的topk文章（wij：前面函数计算的协同过滤相似度）
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            # 跳过相似文章也被点击点击的情况
            if j in user_hist_items:
                continue
            
            # 各类权重因子
            # 1. 文章创建时间差权重（创建时间越近，权重越大）
            created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
            # 2. 相似文章和历史点击文章序列中历史文章所在的位置权重（位置越靠后，权重越高）
            loc_weight = (0.9 ** (len(user_hist_items) - loc))
            # 3. 内容相似性权重
            # 如果emb_i2i_sim中存在i和j的相似度，则将相似度加到content_weight中
            # TODO：如果没有相似度，则content_weight为1.0
            content_weight = 1.0
            if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                content_weight += emb_i2i_sim[i][j]
            if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                content_weight += emb_i2i_sim[j][i]
            
            # 计算相似度得分
            # 一个历史文章可能对同一个候选贡献多次（同一对相似物品在不同用户中出现多次）
            # 这里针对单用户，每对只计算一次
            item_rank.setdefault(j, 0)
            item_rank[j] += created_time_weight * loc_weight * content_weight * wij
    
    # 个数不足recall_item_num，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items(): # 填充的item应该不在原来的列表中
                continue
            item_rank[item] = - i - 100 # 随便给个负数就行
            if len(item_rank) == recall_item_num:
                break
    
    # 召回最终名单按得分排序
    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
        
    return item_rank

# ItemCF召回 - 使用协同过滤相似度
# 先进行itemcf召回, 为了召回评估，所以提取最后一次点击
if metric_recall: # 线下评估模式 - 提取最后一次点击
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else: # 线上模式 - 全量数据
    trn_hist_click_df = all_click_df

# 准备用户历史点击日志和输出容器
user_recall_items_dict = collections.defaultdict(dict) # 每个用户召回到的物品和分数 {user1: {item1: score1, item2: score2...}, ...}
user_item_time_dict = get_user_item_time(trn_hist_click_df) # 点击序列日志 {user1: [(item1, time1), (item2, time2), ...], ...}

# 加载协同过滤相似性矩阵
i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb'))
# 加载基于embedding内容的相似性矩阵
emb_i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl', 'rb'))

# 设置召回参数
sim_item_topk = 20 # 每个历史物品取的相似文章数量
recall_item_num = 10 # 每个用户最终召回的物品数量
item_topk_click = get_item_topk_click(trn_hist_click_df, k=50) # 热门物品

# 遍历每个用户
for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, \
                                                        i2i_sim, sim_item_topk, recall_item_num, \
                                                        item_topk_click, item_created_time_dict, emb_i2i_sim)

# 保存召回结果
user_multi_recall_dict['itemcf_sim_itemcf_recall'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['itemcf_sim_itemcf_recall'], open(save_path + 'itemcf_recall_dict.pkl', 'wb'))

if metric_recall:
    # 召回效果评估
    metrics_recall(user_multi_recall_dict['itemcf_sim_itemcf_recall'], trn_last_click_df, topk=recall_item_num)

# Item召回 - 使用Embedding相似度
# 下面的步骤与上面一致，只更改了相似度计算方式
if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df
user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)
i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl','rb'))
sim_item_topk = 20
recall_item_num = 10
item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)
for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, sim_item_topk, 
                                                        recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim)
user_multi_recall_dict['embedding_sim_item_recall'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['embedding_sim_item_recall'], open(save_path + 'embedding_sim_item_recall.pkl', 'wb'))
if metric_recall:
    metrics_recall(user_multi_recall_dict['embedding_sim_item_recall'], trn_last_click_df, topk=recall_item_num)



'''
UserCF召回
'''
# 基于用户的召回 u2u2i
# 关联规则：被推荐用户历史点击文章 与 相似用户历史点击文章
# 1. 文章相似度
# 2. 文章创建时间差
# 3. 文章位置权重（累加求和）
def user_based_recommend(user_id, user_item_time_dict, u2u_sim, sim_user_topk, recall_item_num, 
                         item_topk_click, item_created_time_dict, emb_i2i_sim):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: {item1: time1, item2: time2..}...}
        :param u2u_sim: 字典，文章相似性矩阵
        :param sim_user_topk: 整数， 选择与当前用户最相似的前k个用户
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        :param item_created_time_dict: 文章创建时间列表
        :param emb_i2i_sim: 字典基于内容embedding算的文章相似矩阵
        
        return: 召回的文章列表 {item1:score1, item2: score2...}
    """
    # 历史交互
    user_item_time_list = user_item_time_dict[user_id] # {item1: time1, item2: time2...}
    user_hist_items = set([i for i, t in user_item_time_list]) # 存在一个用户与某篇文章的多次交互， 这里得去重（不召回用户已经看过的文章）
    
    items_rank = {}
    # 遍历与当前用户最相似的k个用户
    for sim_u, wuv in sorted(u2u_sim[user_id].items(), key=lambda x: x[1], reverse=True)[:sim_user_topk]:
        # 遍历相似用户的历史点击文章
        for i, click_time in user_item_time_dict[sim_u]:
            if i in user_hist_items: # 跳过用户已经看过的文章
                continue
            items_rank.setdefault(i, 0)
            
            loc_weight = 1.0
            content_weight = 1.0
            created_time_weight = 1.0
            
            # 权重因子
            for loc, (j, click_time) in enumerate(user_item_time_list):
                # 1. 位置权重：目标用户点击序列中，越靠后的历史文章权重越高
                loc_weight += 0.9 ** (len(user_item_time_list) - loc)
                # 2. 内容相似性权重 - 使用embedding计算
                if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                    content_weight += emb_i2i_sim[i][j]
                if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                    content_weight += emb_i2i_sim[j][i]
                # 3. 创建时间差权重，创建时间越近，权重越大
                created_time_weight += np.exp(0.8 * np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
            
            # 累乘各个权重计算相似度得分
            items_rank[i] += loc_weight * content_weight * created_time_weight * wuv
        
    # 热度补全
    if len(items_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in items_rank.items(): # 填充的item应该不在原来的列表中
                continue
            items_rank[item] = - i - 100 # 填充负数保证热门物品在最后
            if len(items_rank) == recall_item_num:
                break
    # 召回最终名单按得分排序
    items_rank = sorted(items_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]    
    return items_rank

# UserCF召回 - 使用协同过滤相似度
# 召回评估提取最后一次点击
# 由于usercf中计算user之间的相似度的过程太费内存了，全量数据这里就没有跑，跑了一个采样之后的数据
if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df
    
user_recall_items_dict = collections.defaultdict(dict) # 每个用户的物品点击日志
user_item_time_dict = get_user_item_time(trn_hist_click_df) # 点击序列日志
u2u_sim = pickle.load(open(save_path + 'usercf_u2u_sim.pkl', 'rb')) # 用户相似性矩阵

sim_user_topk = 20 # 每个用户召回的相似用户数量
recall_item_num = 10 # 每个用户召回的物品数量
item_topk_click = get_item_topk_click(trn_hist_click_df, k=50) # 热门物品

# 遍历每个用户
for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = user_based_recommend(user, user_item_time_dict, u2u_sim, sim_user_topk, \
                                                        recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim)    
# 保存召回结果
pickle.dump(user_recall_items_dict, open(save_path + 'usercf_u2u2i_recall.pkl', 'wb'))

if metric_recall:
    # 召回效果评估
    metrics_recall(user_recall_items_dict, trn_last_click_df, topk=recall_item_num)

# User召回 - 使用Embedding相似度（YoutubeDNN）
# topk指的是每个user, faiss搜索后返回最相似的topk个user
def u2u_embdding_sim(click_df, user_emb_dict, save_path, topk):
    """
        使用Embedding的方式获取u2u的相似性矩阵
        :param click_df: 数据集
        :param user_emb_dict: 字典, 用户embedding字典
        :param save_path: 字符串, 保存路径
        :param topk: 整数, 每个用户召回的相似用户数量
    """
    # 这部分使用上面YouTubeDNN得到的用户向量，通过Faiss做近邻检索
    # 进而构建一个基于embedding的u2u的相似性矩阵
    # 然后再用user_based_recommend函数进行召回
    user_list = []
    user_emb_list = []
    for user_id, user_emb in user_emb_dict.items():
        user_list.append(user_id)
        user_emb_list.append(user_emb)
    user_index_2_rawid_dict = {k: v for k, v in zip(range(len(user_list)), user_list)}    
    user_emb_np = np.array(user_emb_list, dtype=np.float32)
    # 建立faiss索引
    user_index = faiss.IndexFlatIP(user_emb_np.shape[1])
    user_index.add(user_emb_np)
    # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
    sim, idx = user_index.search(user_emb_np, topk) # 返回的是列表

   
    # 将向量检索的结果保存成原始id的对应关系
    user_sim_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(user_emb_np)), sim, idx)):
        target_raw_id = user_index_2_rawid_dict[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]): 
            rele_raw_id = user_index_2_rawid_dict[rele_idx]
            user_sim_dict[target_raw_id][rele_raw_id] = user_sim_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value
    # 保存i2i相似度矩阵
    pickle.dump(user_sim_dict, open(save_path + 'youtube_u2u_sim.pkl', 'wb'))   
    return user_sim_dict

# 读取YoutubeDNN过程中产生的user embedding, 然后使用faiss计算用户之间的相似度
# 这里需要注意，这里得到的user_embedding其实并不是很好，因为YoutubeDNN中使用的是用户点击序列来训练的user_embedding,
# 如果序列普遍都比较短的话，其实效果并不是很好
user_emb_dict = pickle.load(open(save_path + 'user_youtube_emb.pkl', 'rb'))
u2u_sim = u2u_embdding_sim(all_click_df, user_emb_dict, save_path, topk=10)


# 使用召回评估函数验证当前召回方式的效果
if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)
u2u_sim = pickle.load(open(save_path + 'youtube_u2u_sim.pkl', 'rb'))

sim_user_topk = 20
recall_item_num = 10

item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)
for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = user_based_recommend(user, user_item_time_dict, u2u_sim, sim_user_topk, \
                                                        recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim)
    
user_multi_recall_dict['youtubednn_usercf_recall'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['youtubednn_usercf_recall'], open(save_path + 'youtubednn_usercf_recall.pkl', 'wb'))

if metric_recall:
    # 召回效果评估
    metrics_recall(user_multi_recall_dict['youtubednn_usercf_recall'], trn_last_click_df, topk=recall_item_num)



'''
冷启动问题
'''
# 分类：
# 1. 用户冷启动：平台新加入的用户无交互记录，如何给用户推荐文章（测试集的用户是否在测试集对应的log数据出现过）
# 也可以通过设定某些指标判别冷启动用户（使用时长，点击率，留存率等）
# 2. 物品冷启动：新加入的文章无交互记录，如何推荐给用户（可看作点击日志中没有出现过的文章）
# 3. 系统冷启动：1和2的综合

'''
1. 用户冷启动：
分析数据发现越20%用户只有一次点击，点击较少的用户召回可以单独做一些策略的补充（比如热门物品召回）
或者在排序后直接基于业务规则加入一些文章
2. 物品冷启动：
一种最简单的方案是随机选择一小部分商品，作为初始召回
另一种方案是：首先基于embedding大规模召回一批候选商品，然后使用场景规则过滤掉商品，最后使用模型进行召回排序
如：与历史点击文章主题相似的文章，或字数相差不大的文章，留下与测试集用户最后一次点击时间更接近的文章（如当天）
 - 这里基于embedding计算item之间相似度一致，但选取的候选文章数量要多一些，否则筛选后可能没有候选商品
 - 这里只考虑对物品冷启动的召回策略
'''

# 先进行itemcf召回，这里不需要做召回评估，这里只是一种策略
trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)
i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl','rb'))

sim_item_topk = 150
recall_item_num = 100 # 稍微召回多一点文章，便于后续的规则筛选

item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)
for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, sim_item_topk, 
                                                        recall_item_num, item_topk_click,item_created_time_dict, emb_i2i_sim)
pickle.dump(user_recall_items_dict, open(save_path + 'cold_start_items_raw_dict.pkl', 'wb'))


# 基于规则进行文章过滤
# 保留文章主题与用户历史浏览主题相似的文章
# 保留文章字数与用户历史浏览文章字数相差不大的文章
# 保留最后一次点击当天的文章
# 按照相似度返回最终的结果

def get_click_article_ids_set(all_click_df): # 获取用户历史点击日志（用于后续去重）
    return set(all_click_df.click_article_id.values)

def cold_start_items(user_recall_items_dict, user_hist_item_typs_dict, user_hist_item_words_dict, \
                     user_last_item_created_time_dict, item_type_dict, item_words_dict, 
                     item_created_time_dict, click_article_ids_set, recall_item_num):
    """
        冷启动的情况下召回一些文章
        :param user_recall_items_dict: 基于内容embedding相似性召回来的很多文章， 字典， {user1: [item1, item2, ..], }
        :param user_hist_item_typs_dict: 字典， 用户点击的文章的主题映射
        :param user_hist_item_words_dict: 字典， 用户点击的历史文章的字数映射
        :param user_last_item_created_time_idct: 字典，用户点击的历史文章创建时间映射
        :param item_tpye_idct: 字典，文章主题映射
        :param item_words_dict: 字典，文章字数映射
        :param item_created_time_dict: 字典， 文章创建时间映射
        :param click_article_ids_set: 集合，用户点击过得文章, 也就是日志里面出现过的文章
        :param recall_item_num: 召回文章的数量， 这个指的是没有出现在日志里面的文章数量
    """
    
    cold_start_user_items_dict = {}
    for user, item_list in tqdm(user_recall_items_dict.items()):
        cold_start_user_items_dict.setdefault(user, [])
        for item, score in item_list:
            # 获取用户侧的历史文章信息
            hist_item_type_set = user_hist_item_typs_dict[user] # 主题
            hist_mean_words = user_hist_item_words_dict[user] # 字数
            hist_last_item_created_time = user_last_item_created_time_dict[user] # 创建时间
            hist_last_item_created_time = datetime.fromtimestamp(hist_last_item_created_time)
            
            # 获取召回的大量候选物品属性
            curr_item_type = item_type_dict[item] # 主题
            curr_item_words = item_words_dict[item] # 字数
            curr_item_created_time = item_created_time_dict[item] # 创建时间
            curr_item_created_time = datetime.fromtimestamp(curr_item_created_time)

            # 基于规则进行文章过滤
            # 首先，文章不能出现在用户的历史点击中， 然后根据文章主题，文章单词数差，文章创建时间差进行筛选
            if curr_item_type not in hist_item_type_set or \
                item in click_article_ids_set or \
                abs(curr_item_words - hist_mean_words) > 200 or \
                abs((curr_item_created_time - hist_last_item_created_time).days) > 90: 
                continue
            # 如果通过了所有规则，则将该物品加入到召回列表中
            cold_start_user_items_dict[user].append((item, score)) # {user1: [(item1, score1), (item2, score2)..]...}
    
    # 需要控制一下冷启动召回的数量
    cold_start_user_items_dict = {k: sorted(v, key=lambda x:x[1], reverse=True)[:recall_item_num] \
                                  for k, v in cold_start_user_items_dict.items()}
    # 保存召回结果
    pickle.dump(cold_start_user_items_dict, open(save_path + 'cold_start_user_items_dict.pkl', 'wb'))
    return cold_start_user_items_dict


all_click_df_ = all_click_df.copy()
all_click_df_ = all_click_df_.merge(item_info_df, how='left', on='click_article_id')
user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict = get_user_hist_item_info_dict(all_click_df_)
click_article_ids_set = get_click_article_ids_set(all_click_df)
# 需要注意的是
# 这里使用了很多规则来筛选冷启动的文章，所以前面再召回的阶段就应该尽可能的多召回一些文章，否则很容易被删掉
cold_start_user_items_dict = cold_start_items(user_recall_items_dict, user_hist_item_typs_dict, user_hist_item_words_dict, \
                                              user_last_item_created_time_dict, item_type_dict, item_words_dict, \
                                              item_created_time_dict, click_article_ids_set, recall_item_num)

user_multi_recall_dict['cold_start_recall'] = cold_start_user_items_dict


'''
将多路召回合并
'''
def combine_recall_results(user_multi_recall_dict, weight_dict=None, topk=25):
    final_recall_items_dict = {}
    
    # 定义一个归一化函数，把每种召回方法给用户打出的分数归一化到[0, 1]之间，避免不同方法得分尺度差异过大
    def norm_user_recall_items_sim(sorted_item_list):
        # 如果冷启动中没有文章或者只有一篇文章，直接返回，出现这种情况的原因可能是冷启动召回的文章数量太少了，
        # 基于规则筛选之后就没有文章了, 这里还可以做一些其他的策略性的筛选
        if len(sorted_item_list) < 2:
            return sorted_item_list
        
        min_sim = sorted_item_list[-1][1]
        max_sim = sorted_item_list[0][1]
        
        norm_sorted_item_list = []
        # 归一化函数的计算公式：依据该路召回列表中的最大最小值比例
        # norm_score = (score - min_sim) / (max_sim - min_sim)
        for item, score in sorted_item_list:
            if max_sim > 0:
                norm_score = 1.0 * (score - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 1.0
            else:
                norm_score = 0.0
            norm_sorted_item_list.append((item, norm_score))
            
        return norm_sorted_item_list
    
    print('多路召回合并...')
    # 遍历每一种召回方法，逐用户归一化，原地修改item的得分
    for method, user_recall_items in tqdm(user_multi_recall_dict.items()):
        print(method + '...')
        # 在计算最终召回结果的时候，也可以为每一种通道的召回结果设置一个权重
        if weight_dict == None:
            recall_method_weight = 1
        else:
            recall_method_weight = weight_dict[method]
        
        for user_id, sorted_item_list in user_recall_items.items(): # 进行归一化
            user_recall_items[user_id] = norm_user_recall_items_sim(sorted_item_list)
        
        # 逐用户，逐通道累加得分
        for user_id, sorted_item_list in user_recall_items.items():
            # print('user_id')
            final_recall_items_dict.setdefault(user_id, {})
            for item, score in sorted_item_list:
                final_recall_items_dict[user_id].setdefault(item, 0)
                # 不同通道召回同一个物品，则累加得分
                final_recall_items_dict[user_id][item] += recall_method_weight * score  
    
    final_recall_items_dict_rank = {}
    # 多路召回时也可以控制最终的召回数量，这里只召回topk个
    for user, recall_item_dict in final_recall_items_dict.items():
        final_recall_items_dict_rank[user] = sorted(recall_item_dict.items(), key=lambda x: x[1], reverse=True)[:topk]

    # 将多路召回后的最终结果字典保存到本地
    pickle.dump(final_recall_items_dict, open(os.path.join(save_path, 'final_recall_items_dict.pkl'),'wb'))

    return final_recall_items_dict_rank


# 这里可以设置不同召回通道的权重
# 根据召回评估的结果，有些通道召回效果好，另一些很差，可以调整不同召回通道的权重
weight_dict = {'itemcf_sim_itemcf_recall': 1.0,
               'embedding_sim_item_recall': 1.0,
               'youtubednn_recall': 1.0,
               'youtubednn_usercf_recall': 1.0, 
               'cold_start_recall': 1.0}


# 最终合并之后每个用户召回150个商品进行排序
final_recall_items_dict_rank = combine_recall_results(user_multi_recall_dict, weight_dict, topk=150)