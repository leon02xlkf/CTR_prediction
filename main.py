# import packages
import time, math, os
from tqdm import tqdm
import gc
import pickle
import random
from datetime import datetime
from operator import itemgetter
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

data_path = './data_raw/'
save_path = './tmp_results/'

# 节约内存的一个标配函数
def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                           100*(start_mem-end_mem)/start_mem,
                                                                                                           (time.time()-starttime)/60))
    return df


# 在调试模式下，从训练数据中随机抽取指定数量的用户（默认1万），生成一个小规模的点击日志数据集
def get_all_click_sample(data_path, sample_nums=10000):
    """
        训练集中采样一部分数据调试
        data_path: 原数据的存储路径
        sample_nums: 采样数目（这里由于机器的内存限制，可以采样用户做）
    """
    all_click = pd.read_csv(data_path + 'train_click_log.csv') # 读取训练集的点击日志数据
    all_user_ids = all_click.user_id.unique() # 获取所有（不重复的）用户ID

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False) # 抽样
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)] # 筛选出抽样用户的点击日志

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp'])) # 移除重复的点击记录（某一用户在同一时刻点击了同一篇文章）
    return all_click


# 读取点击数据，这里使用offline分成线上和线下
# 线下：只使用训练集（验证模型或特征的有效性，测试模型对已知用户行为的拟合效果）
# 线上：合并训练集和测试集（生成最终的提交结果给真实用户推荐，覆盖更多的用户行为。更多的数据能提高召回的覆盖率和推荐质量）
def get_all_click_df(data_path='./data_raw/', offline=True):
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')

        all_click = trn_click.append(tst_click)

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click

# 在线训练 - 使用全量训练集
all_click_df = get_all_click_df(offline=False)

# 根据点击时间获取用户的点击文章序列   {user1: {item1: time1, item2: time2..}...}
def get_user_item_time(click_df):
    click_df = click_df.sort_values('click_timestamp') # 将用户的点击行为按点击时间排序
    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp'])) # 将点击文章ID和点击时间配对成一个元组，然后装进一个list
    user_item_time_df = click_df.groupby('user_id')['click_article_id', 'click_timestamp'].apply(lambda x: make_item_time_pair(x))\
                                                            .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list'])) # 将用户ID和点击文章list配对成一个字典
    return user_item_time_dict

# 获取近期点击最多的TopK文章
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click

def itemcf_sim(df):
    """
        ItemCF的相似性矩阵计算
        :param df: 数据表
        :item_created_time_dict:  文章创建时间的字典
        return : 文章与文章的相似性矩阵
        思路: 基于物品的协同过滤(详细请参考上一期推荐系统基础的组队学习)， 在多路召回部分会加上关联规则的召回策略
    """
    user_item_time_dict = get_user_item_time(df) # 返回上面生成的字典 - 用户:[(文章,点击时间),...]
    
    # 计算物品相似度
    i2i_sim = {} # [i][j] = 文章i和文章j的相似性
    item_cnt = defaultdict(int) # 储存文章i被点击的次数
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        # 计算加权共现频次
        for i, i_click_time in item_time_list:
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for j, j_click_time in item_time_list:
                if(i == j):
                    continue
                i2i_sim[i].setdefault(j, 0)
                # 使用对数为共现频次加权（惩罚高活跃用户，因为他们点击太多物品，点击记录会变得不可靠）
                # U1: A B
                # U2: A B C D E F
                # A,B 在U1权重较高，在U2权重较低
                i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)
                
    i2i_sim_ = i2i_sim.copy()
    # 计算Item之间的余弦相似度
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
    # 将得到的相似性矩阵保存到本地
    pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))
    return i2i_sim_

i2i_sim = itemcf_sim(all_click_df)

# 基于商品的召回i2i
def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: {item1: time1, item2: time2..}...}
        :param i2i_sim: 字典，文章相似性矩阵
        :param sim_item_topk: 整数， 选择与当前文章最相似的前k篇文章
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全        
        return: 召回的文章列表 {item1:score1, item2: score2...}
        注意: 基于物品的协同过滤(详细请参考上一期推荐系统基础的组队学习)， 在多路召回部分会加上关联规则的召回策略
    """
    
    # 通过字典获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]
    
    item_rank = {}
    # 遍历用户历史交互的每篇文章i
    for loc, (i, click_time) in enumerate(user_hist_items):
        # 查找与文章i最相似的sim_item_topk篇文章
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            # 如果文章j已经存在于用户的历史交互中，则跳过
            if j in user_hist_items:
                continue
            # 将文章j的得分初始化为0
            item_rank.setdefault(j, 0)
            # 将文章j的得分加上文章i和文章j的相似度
            item_rank[j] +=  wij
    
    # 如果召回数量不足，则用热门商品补全（如个别冷启动用户点击次数太少）
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items(): # 填充的热门item已经在借助相似度召回的item中，则跳过
                continue
            item_rank[item] = - i - 100 # 随便给这些补充的物品赋负数，确保排序靠后
            if len(item_rank) == recall_item_num:
                break
    
    # 将召回列表按相似度得分排序
    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
    # 返回最终的召回结果[(item1, score1), (item2, score2)...]
    return item_rank

user_recall_items_dict = collections.defaultdict(dict) # 初始化用户-文章-点击率字典
user_item_time_dict = get_user_item_time(all_click_df) # 填充字典
i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb')) # 导入预保存的相似度矩阵
sim_item_topk = 10 # 对每个用户使用相似度召回前10篇文章
recall_item_num = 10 # 对每个用户最终召回文章10篇作为候选列表
item_topk_click = get_item_topk_click(all_click_df, k=50) # 获取全局点击率最高的50篇文章，作为用户召回补全

for user in tqdm(all_click_df['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, 
                                                        sim_item_topk, recall_item_num, item_topk_click)

# 将字典的形式转换成df
# [User, Item, PredictionScore]
user_item_score_list = []
for user, items in tqdm(user_recall_items_dict.items()):
    for item, score in items:
        user_item_score_list.append([user, item, score])

recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])

# 生成提交文件（根据预测得分排序，取前topk=5篇）
def submit(recall_df, topk=5, model_name=None):
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    # 根据用户id分组，对预测得分进行降序排序
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')
    
    # 判断是不是每个用户都有topk篇文章及以上，否则报错中止
    tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
    assert tmp.min() >= topk
    # 删除预测得分列
    del recall_df['pred_score']
    submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()
    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]
    # 按照提交格式定义列名
    submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2', 
                                                  3: 'article_3', 4: 'article_4', 5: 'article_5'})
    
    save_name = save_path + model_name + '_' + datetime.today().strftime('%m-%d') + '.csv'
    submit.to_csv(save_name, index=False, header=True)

# 获取测试集
tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
tst_users = tst_click['user_id'].unique()

# 从所有的召回数据中将测试集中的用户选出来
tst_recall = recall_df[recall_df['user_id'].isin(tst_users)]

# 只保留召回数据中属于测试集用户的部分，生成提交文件
submit(tst_recall, topk=5, model_name='itemcf_baseline')

