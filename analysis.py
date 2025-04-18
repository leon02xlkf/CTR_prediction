'''
数据分析，熟悉了解整个数据集的基本情况
包括：每个文件里有哪些数据，具体的文件中的每个字段表示什么实际含义，以及数据集中特征之间的相关性
分析用户本身的基本属性，文章基本属性，以及用户和文章交互的一些分布
'''
# %matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='SimHei', size=13)
import os,gc,re,warnings,sys
warnings.filterwarnings("ignore")
path = './data_raw/'

'''
数据读取
'''
#train
trn_click = pd.read_csv(path+'train_click_log.csv')
item_df = pd.read_csv(path+'articles.csv')
item_df = item_df.rename(columns={'article_id': 'click_article_id'})  #重命名，方便后续match
item_emb_df = pd.read_csv(path+'articles_emb.csv')

#test
tst_click = pd.read_csv(path+'testA_click_log.csv')

# 对训练集、测试集中每个用户的点击时间戳倒序排序（从晚到早）
trn_click['rank'] = trn_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)
tst_click['rank'] = tst_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)
#计算用户点击文章的次数，并添加新的一列count（统计所有用户的点击总次数）
trn_click['click_cnts'] = trn_click.groupby(['user_id'])['click_timestamp'].transform('count')
tst_click['click_cnts'] = tst_click.groupby(['user_id'])['click_timestamp'].transform('count')


'''
数据浏览
'''
# 查看训练集用户点击日志文件中字段的信息
trn_click = trn_click.merge(item_df, how='left', on=['click_article_id'])
trn_click.head()
trn_click.info()
trn_click.describe()

# user_id: 用户的唯一标识
# click_article_id: 用户点击的文章唯一标识
# click_timestamp: 用户点击文章时的时间戳
# click_environment: 用户点击文章的环境
# click_deviceGroup: 用户点击文章的设备组
# click_os: 用户点击文章时的操作系统
# click_country: 用户点击文章时的所在的国家
# click_region: 用户点击文章时所在的区域
# click_referrer_type: 用户点击文章时，文章的来源


trn_click.user_id.nunique()
# = 200000（训练集中的用户数量为20w）
trn_click.groupby('user_id')['click_article_id'].count().min()
# = 2 （训练集里面每个用户至少点击了两篇文章）

# 绘制用户点击文章的次数的直方图
plt.figure()
plt.figure(figsize=(15, 20))
i = 1
for col in ['click_article_id', 'click_timestamp', 'click_environment', 'click_deviceGroup', 'click_os', 'click_country', 
            'click_region', 'click_referrer_type', 'rank', 'click_cnts']:
    plot_envs = plt.subplot(5, 2, i)
    i += 1
    v = trn_click[col].value_counts().reset_index()[:10]
    fig = sns.barplot(x=v['index'], y=v[col])
    for item in fig.get_xticklabels():
        item.set_rotation(90)
    plt.title(col)
plt.tight_layout()
plt.show()
# 点击时间clik_timestamp:
# 分布较为平均，可不做特殊处理。由于时间戳是13位的，后续将时间格式转换成10位方便计算
# 点击环境click_environment:
# 仅有1922次（占0.1%）点击环境为1；仅有24617次（占2.3%）点击环境为2；剩余（占97.6%）点击环境为4
# 点击设备组click_deviceGroup:
# 设备1占大部分（60.4%），设备3占36%


# 查看测试集用户点击日志文件中字段的信息
tst_click = tst_click.merge(item_df, how='left', on=['click_article_id'])
tst_click.head()
tst_click.describe()
# id不同:
# 训练集的用户ID由0 ~ 199999，而测试集A的用户ID由200000 ~ 249999


tst_click.user_id.nunique()
# = 50000 （测试集中的用户数量为5w）
tst_click.groupby('user_id')['click_article_id'].count().min() # 注意测试集里面有只点击过一次文章的用户
# = 1 （测试集里面每个用户至少点击过一篇文章）


# 新闻文章数据集浏览
item_df.head().append(item_df.tail())
item_df['words_count'].value_counts()
print(item_df['category_id'].nunique()) # 461个文章主题
item_df['category_id'].hist()
item_df.shape # 364047篇文章

# 新闻文章embedding向量表示
item_emb_df.head()
item_emb_df.shape # 364047行，251列


'''
数据分析
'''
# 用户点击文章的次数
#####merge
user_click_merge = trn_click.append(tst_click)

#用户重复点击
user_click_count = user_click_merge.groupby(['user_id', 'click_article_id'])['click_timestamp'].agg({'count'}).reset_index()
user_click_count[:10]

user_click_count[user_click_count['count']>7] # 显示用户重复点击次数大于7次的新闻
user_click_count['count'].unique() # 显示用户重复点击次数的去重值

# 显示用户点击新闻次数
user_click_count.loc[:,'count'].value_counts() 
# 有1605541（约占99.2%）的用户未重复阅读过文章，仅有极少数用户重复点击过某篇文章。这个也可以单独制作成特征


# 用户点击环境变化分析
# 画出若干列的直方图，以观察用户在点击行为中各个属性的分布是否稳定（城市、设备、操作系统、国家、区域、来源）
def plot_envs(df, cols, r, c):
    plt.figure()
    plt.figure(figsize=(10, 5))
    i = 1
    for col in cols:
        plt.subplot(r, c, i)
        i += 1
        v = df[col].value_counts().reset_index()
        fig = sns.barplot(x=v['index'], y=v[col])
        for item in fig.get_xticklabels():
            item.set_rotation(90)
        plt.title(col)
    plt.tight_layout()
    plt.show()

# 分析用户点击环境变化是否明显，这里不重复抽取5个用户分析这些用户的点击环境分布
sample_user_ids = np.random.choice(tst_click['user_id'].unique(), size=5, replace=False)
sample_users = user_click_merge[user_click_merge['user_id'].isin(sample_user_ids)]
cols = ['click_environment','click_deviceGroup', 'click_os', 'click_country', 'click_region','click_referrer_type']
for _, user_df in sample_users.groupby('user_id'):
    plot_envs(user_df, cols, 2, 3)
# 结果：绝大多数用户的点击环境比较固定，可以基于这些环境的统计特征代表该用户本身的属性
# 如果用户在不同点击中环境变化较大，则可能存在异常点击行为（共享设备，共享账号网络等）


# 绘制用户点击新闻次数的分布，可以直观反映用户的活跃度
user_click_item_count = sorted(user_click_merge.groupby('user_id')['click_article_id'].count(), reverse=True)
plt.plot(user_click_item_count)
plt.plot(user_click_item_count[:50]) # 精细绘制点击次数在前50的用户
plt.plot(user_click_item_count[25000:50000]) # 点击次数排名在[25000:50000]之间
# 点击次数排在前50的用户点击次数都在100次以上，点击次数不超过2次的用户非常多
# 因此，判断用户活跃度的一种简单方式是：定义点击次数大于等于100次的用户为活跃用户，小于2次的为非活跃用户
# 更加全面的方法是结合点击时间判断


# 绘制新闻被点击的次数
item_click_count = sorted(user_click_merge.groupby('click_article_id')['user_id'].count(), reverse=True)
plt.plot(item_click_count)
plt.plot(item_click_count[:100])
plt.plot(item_click_count[:20])
plt.plot(item_click_count[3500:])
# 点击次数最多的前100篇新闻超过了1000次
# 点击次数最多的前20篇新闻超过了2500次
# 绝大多数新闻只被点击过一两次，可以定义为冷门新闻
# 因此，可以直接根据点击次数划分热门新闻，更全面的方法也可以加入点击时间


# 新闻共现频次：两篇新闻连续出现的次数
# 找出哪些新闻是经常被一起点击的，可以用于构建文章之间的关联特征
tmp = user_click_merge.sort_values('click_timestamp') # 先把所有文章按照点击时间排序
tmp['next_item'] = tmp.groupby(['user_id'])['click_article_id'].transform(lambda x:x.shift(-1))
# 统计set(item1, item2)出现的次数
union_item = tmp.groupby(['click_article_id','next_item'])['click_timestamp'].agg({'count'}).reset_index().sort_values('count', ascending=False)
union_item[['count']].describe()

# 画图，横坐标是新闻1，纵坐标是它和新闻2一起出现的频次
# 可以观察出哪些文章更容易带出其他文章
x = union_item['click_article_id']
y = union_item['count']
plt.scatter(x, y)
# 画出共现频次较低的新闻对（从第40000个开始）查看低频共现的尾部分布
plt.plot(union_item['count'].values[40000:])
# 1. 如果大部分新闻对只共现1-2次（长尾），说明用户行为分散 -> 不同用户点击的文章组合差异很大，没有特别明确的群体兴趣
# 可能表示协同过滤效果不佳（需要更多依赖文章相似度或用户画像进行推荐）
# 可能是偶然点击（识别在建图和训练模型时权重太小，容易引入噪音）
# 2. 如果有一部分新闻对的共现频次特别高，可以考虑把它们作为“强相关文章”进行协同推荐或召回

# 大约70000对新闻至少共现了一次



# 新闻文章信息
# 每种类型的新闻出现的次数
plt.plot(user_click_merge['category_id'].value_counts().values)
plt.plot(user_click_merge['category_id'].value_counts().values[150:]) 
# 出现次数少于150次的新闻类型, 有些新闻类型，基本上就出现过几次

# 新闻字数的描述性统计（cnt， mean， std， min， max）
user_click_merge['words_count'].describe()
plt.plot(user_click_merge['words_count'].values)


# 用户点击的新闻类型偏好（可以用于度量用户的兴趣是否广泛）
# 一小部分用户的阅读类型及其广泛，大部分人都处在20个新闻类型以下
plt.plot(sorted(user_click_merge.groupby('user_id')['category_id'].nunique(), reverse=True))
user_click_merge.groupby('user_id')['category_id'].nunique().reset_index().describe()


# 用户查看文章的长度分布
# 通过统计不同用户点击新闻的平均字数，可以反映用户对长文更感兴趣还是对短文更感兴趣
# 一小部分人平均词数非常高，也有一小部分人平均词数非常低，大多数人偏好阅读200-400字的新闻
plt.plot(sorted(user_click_merge.groupby('user_id')['words_count'].mean(), reverse=True))
# 仔细查看1000-45000的区间，发现大多数人都是看250字以下的文章
plt.plot(sorted(user_click_merge.groupby('user_id')['words_count'].mean(), reverse=True)[1000:45000])
# 更加详细的参数（cnt， mean， std， min， max）
user_click_merge.groupby('user_id')['words_count'].mean().reset_index().describe()
user_click_merge.head()

# 用户点击新闻的时间分析
# 时间戳以秒为单位不好分析
# 为了更好的可视化，这里把时间进行归一化操作，转化在[0, 1]区间
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
user_click_merge['click_timestamp'] = mm.fit_transform(user_click_merge[['click_timestamp']])
user_click_merge['created_at_ts'] = mm.fit_transform(user_click_merge[['created_at_ts']])
# 排序
user_click_merge = user_click_merge.sort_values('click_timestamp')

# 定义函数计算“平均时间差”
# 给定一个用户，对点击行为的时间计算差值
def mean_diff_time_func(df, col):
    df = pd.DataFrame(df, columns={col})
    df['time_shift1'] = df[col].shift(1).fillna(0)
    df['diff_time'] = abs(df[col] - df['time_shift1'])
    return df['diff_time'].mean()
# 对这个差值取平均，能反映用户在多长时间会点击一篇新文章

# 使用函数分析两种时间间隔：
# 1. 点击时间差的平均值（时间间隔越小，说明用户更喜欢快速连续点击新闻）
mean_diff_click_time = user_click_merge.groupby('user_id')['click_timestamp', 'created_at_ts'].apply(lambda x: mean_diff_time_func(x, 'click_timestamp'))
plt.plot(sorted(mean_diff_click_time.values, reverse=True))
# 2. 前后点击前后两篇文章“创建时间”差的平均值（可以判断用户是否喜欢连续点击“同一时期的新闻”）
mean_diff_created_time = user_click_merge.groupby('user_id')['click_timestamp', 'created_at_ts'].apply(lambda x: mean_diff_time_func(x, 'created_at_ts'))
plt.plot(sorted(mean_diff_created_time.values, reverse=True))

# 分析用户前后点击文章的相似性
# 将article_id映射为文章的embedding向量行索引
item_idx_2_rawid_dict = dict(zip(item_emb_df['article_id'], item_emb_df.index))
del item_emb_df['article_id']
item_emb_np = np.ascontiguousarray(item_emb_df.values, dtype=np.float32)
# 随机不放回抽取15个用户
sub_user_ids = np.random.choice(user_click_merge.user_id.unique(), size=15, replace=False)
sub_user_info = user_click_merge[user_click_merge['user_id'].isin(sub_user_ids)]

sub_user_info.head()
# 计算前后两篇文章的余弦相似度
def get_item_sim_list(df):
    sim_list = []
    item_list = df['click_article_id'].values
    for i in range(0, len(item_list)-1):
        emb1 = item_emb_np[item_idx_2_rawid_dict[item_list[i]]]
        emb2 = item_emb_np[item_idx_2_rawid_dict[item_list[i+1]]]
        sim_list.append(np.dot(emb1,emb2)/(np.linalg.norm(emb1)*(np.linalg.norm(emb2))))
    sim_list.append(0)
    return sim_list
# 对每个用户，计算前后两篇文章的余弦相似度（判断用户是否喜欢连续点击“同一内容类型”的新闻）
for _, user_df in sub_user_info.groupby('user_id'):
    item_sim_list = get_item_sim_list(user_df)
    plt.plot(item_sim_list)