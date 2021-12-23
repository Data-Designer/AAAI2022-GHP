import torch
import numpy as np
import pandas as pd
import random
import dgl
from config.settings import *
from sklearn.preprocessing import StandardScaler,OneHotEncoder,PolynomialFeatures,LabelEncoder,MinMaxScaler
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

# Tensorboard
writer = SummaryWriter('../log/runs26/healthy_analysis') # 修改label
# 10 是无percep 11是ND,12是NP,13是NA,14是RA
# 8-15-16-17 hyper-encoder,15最佳
# 18-19-20 GCN
# 21,22,23,24,25,26 single 1-6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# 处理得到的是一天的帮会profile，one-hot处理
def preprocessingpro(data_guild_profile):
    '''
    func: 预处理profile
    '''

    # 这两个每次读取数据都会扩展，所以必须设置成局部变量
    CONT_COLUMNS = ['cai_qi_value', 'today_online_cnt', 'total_count', 'scale', 'build_info', 'xuetu_num', 'member_num',
                    'zi_cai',
                    'rq_value', 'fund', 'prosperity', 'guild_member_cnt', 'max_guild_member_cnt', 'max_xue_tu_member',
                    'full_degree',
                    'liansairank', 'joinmem', 'leavemem', 'chatnum', 'chuangongnum']

    CATG_COLUMNS = ['guild_type', 'low_maintain_state', 'g_force']

    # 补0
    data_guild_profile = data_guild_profile.fillna(0)
    # 类别变量处理
    tmp_columns = []
    for i in CATG_COLUMNS:
        tmp = pd.get_dummies(data_guild_profile[i].astype(int), prefix=i)  # low maintain本身就是0，1，不需要
        tmp_columns.extend(list(tmp.columns))
        data_guild_profile = pd.concat([data_guild_profile, tmp], axis=1)
        data_guild_profile.pop(i)
        CATG_COLUMNS.remove(i)

    CATG_COLUMNS.extend(i for i in tmp_columns)

    # 建筑比较特殊,连续变量
    tmp_build_info = buildprocess(data_guild_profile)
    data_guild_profile = pd.concat([data_guild_profile, tmp_build_info], axis=1)
    data_guild_profile.pop("build_info")
    CONT_COLUMNS.extend(list(tmp_build_info.columns))
    CONT_COLUMNS.remove('build_info')

    # 返回归一化后的连续变量和离散变量
    scaler = StandardScaler()
    x = data_guild_profile
    x_catg = np.array(data_guild_profile[CATG_COLUMNS], dtype='float32')  # 与cont一致即可
    x_cont = np.array(data_guild_profile[CONT_COLUMNS], dtype='float32')
    x_cont = scaler.fit_transform(data_guild_profile[CONT_COLUMNS])

    return [x, x_catg, x_cont]


def buildprocess(data_guild_profile):
    '''
    func: 建筑分割
    '''
    temp = defaultdict(list)
    for row in range(len(data_guild_profile)):
        for index,value in enumerate(data_guild_profile['build_info'][row].split('-')):
            _,value = value.split(':')
            temp[index].append(value)
    build_info = pd.DataFrame(temp)
    build_info.columns =['build_1','build_2','build_3','build_4','build_5',
                              'build_6','build_7','build_8','build_9','build_10','build_11','build_12','build_13']
    build_info = build_info.apply(lambda x:x.astype(int)) # 原来为字符串
    return build_info


def guildprofile(data_guild_profile):
    '''
    func:返回预处理完的帮会profile
    '''
    x,x_catg,x_cont= preprocessingpro(data_guild_profile) # 预处理帮会profile
    poly = PolynomialFeatures(degree=2,interaction_only=True,include_bias=False) # 不要那个为全1列
    x_catg_poly =poly.fit_transform(x_catg) # 分类特征交叉 cross项
    x_catg = torch.from_numpy(x_catg_poly.astype('float32'))
    # 处理连续特征
    x_cont = torch.from_numpy(x_cont.astype('float32'))
    # 参数维度定义
    wide_dim = x_catg_poly.shape[1] # 这地方能简化
    dense_dim = x_cont.shape[1] # dense层传的是交叉特征emebedding和cont维度
    embed_dim = 8 # 一维变8维度，所以下面加上6*8,这里忽略
    return x_catg,x_cont,wide_dim,dense_dim,embed_dim


# def labelprocess(label):
#     '''极端值会影响'''
#     scaler = MinMaxScaler()
#     return scaler.fit_transform(label)



def create_lookup_tables(node_id):
    node_id = set(node_id)  # 顺序不重要
    raw_to_new = {}
    for v_i, v in enumerate(node_id):
        raw_to_new[v] = v_i  # key=raw value=new
    new_to_raw = {v_i: v for v, v_i in raw_to_new.items()}  # key=new value=raw
    return raw_to_new, new_to_raw


def preprocessnewid(vocab_to_int, src_raw_id, dst_raw_id):
    # new id
    src_new_id = [vocab_to_int[i] for i in src_raw_id]
    dst_new_id = [vocab_to_int[i] for i in dst_raw_id]
    return src_new_id, dst_new_id


# def preprocessrawid(new):
#     # raw id
#     raw = [int_to_vocab[i] for i in new]
#     return raw


def build_karate_club_graph(src, dst):
    # 双有向=无向
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # Construct a DGLGraph
    return dgl.graph((u, v))


def idconvert(src_raw_id, dst_raw_id):
    '''
    func:接收原始id对，返回新id对和对照表
    '''
    data_id_all = np.concatenate([src_raw_id, dst_raw_id])  # 原节点id
    vocab_to_int, int_to_vocab = create_lookup_tables(data_id_all[:, 0])  # 提取词典
    src_new_id, dst_new_id = preprocessnewid(vocab_to_int, src_raw_id['role_id_src'],
                                             dst_raw_id['role_id_dst'])  # 转换为新id
    return data_id_all, src_new_id, dst_new_id, vocab_to_int, int_to_vocab


def nodeembed(data_src_embedding, data_dst_embedding, columns, int_to_vocab):
    '''
    func:接收原始embedding，返回去重、排序、归一化的embedding
    '''
    data_src_embedding.columns = columns
    data_dst_embedding.columns = columns  # 统一列名，方便concat
    data_embed_all = pd.concat([data_src_embedding, data_dst_embedding], axis=0, ignore_index=True)
    data_embed_all = data_embed_all.drop_duplicates(['role_id'])  # 去重，每个role的embedding
    list_custom = list(int_to_vocab.values())  # 0-7919对照的id顺序排序
    data_embed_all['role_id'] = data_embed_all['role_id'].astype('category')
    data_embed_all['role_id'].cat.reorder_categories(list_custom, inplace=True)
    data_embed_all.sort_values('role_id', inplace=True)
    data_embed_all = data_embed_all[
        ['role_level', 'vip_level', 'role_skill', 'role_practice', 'role_equip', 'role_total']]
    data_embed_all = data_embed_all.fillna(0)
    scaler = StandardScaler()  # 归一化
    data_embed_all = torch.from_numpy(scaler.fit_transform(data_embed_all).astype('float32'))
    return data_embed_all


def graphbuild(src_new_id, dst_new_id, embed, edge_weights):
    '''
    func:返回构造的G和输入，方便下面构建GNN
    '''
    G = build_karate_club_graph(src_new_id, dst_new_id)
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    G.ndata['feat'] = embed  # embedding.weight 这里是NP,可以直接隐藏掉。
    G.edata['w'] = edge_weights
    G = dgl.add_self_loop(G)
    inputs = embed
    return G, inputs



def graphall(relation_df):
    '''
    func: 接收一天的某种关系df，返回图所需要的5种变量
    '''
    src_raw_id = relation_df[['role_id_src', 'src_guild']]  # data_chat，变量命名要统一格式
    dst_raw_id = relation_df[['role_id_dst', 'dst_guild']]
    data_id_all, src_new_id, dst_new_id, vocab_to_int, int_to_vocab = idconvert(src_raw_id, dst_raw_id)  # 获取节点id
    data_src_embedding = relation_df[SRC_ROLE]  # 获取node embedding
    data_dst_embedding = relation_df[DST_ROLE]
    columns = ['role_id', 'role_level', 'vip_level', 'role_skill', 'role_practice', 'role_equip', 'role_total']
    data_embed_all = nodeembed(data_src_embedding, data_dst_embedding, columns,
                               int_to_vocab)  # 这里隐藏bug，存储int_to_vocab指向不明，需要加参数

    # 根据node embedding，边的权重构造图
    embed = data_embed_all  # 这里传入上面定义的node embedding，id from 0-7917
    edge_weights = torch.from_numpy(
        np.concatenate([relation_df['weight'], relation_df['weight']]).astype('float32'))  # 边没有重复
    G, inputs = graphbuild(src_new_id, dst_new_id, embed, edge_weights)  # 获取图结构和输入
    return data_id_all, src_new_id, dst_new_id, vocab_to_int, int_to_vocab, G, inputs,data_embed_all


# 对label进行一系列的处理
def cutbox(x):
    '''
    func:进行分箱，x为list，label为列表，num为箱分位点或者箱子数量,返回离散化的类别
    '''
    num = 10
    labels = [0, 1,2, 3, 4, 5,6,7,8,9]
    cats = np.array(pd.qcut(x,num,labels))
    return cats


# 针对rank要特殊做处理
def rankpro(rank):
    '''
    0单独一桶，其他的进行分桶
    '''
    labels = [0,1]
    rank = np.array(pd.cut(rank, bins=[-0.1, 0.9, 1000], labels=labels))  # 12个桶
    return rank  #

def cutstab(stab):
    return np.array(list(map(lambda x: 1 if x!=0 else x,stab))) # 对应改aux和main文件


def abvalue(column):
    '''
    需要对异常值进行替换,用均值进行替换
    :return:
    '''
    up = column.mean()+column.std()*2
    down = column.mean()-column.std()*2
    c = column
    c[(c>=up)|(c<=down)]=np.nan
    c.fillna(c.median(),inplace=True)
    return column




