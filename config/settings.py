# 帮会Profile
COLUMNS = ['cai_qi_value', 'today_online_cnt', 'total_count', 'scale', 'build_info', 'xuetu_num',
           'member_num', 'zi_cai', 'rq_value', 'fund', 'prosperity', 'guild_member_cnt', 'max_guild_member_cnt',
           'max_xue_tu_member', 'full_degree', 'g_force', 'liansairank', 'joinmem', 'leavemem', 'chatnum',
           'chuangongnum'
    , 'guild_type', 'low_maintain_state']


# 标杆-恢复
COLUMNS2 = ['cai_qi_value', 'today_online_cnt', 'total_count', 'scale', 'build_info', 'xuetu_num',
           'member_num', 'zi_cai', 'rq_value', 'fund', 'prosperity', 'guild_member_cnt', 'max_guild_member_cnt',
           'max_xue_tu_member', 'full_degree', 'g_force', 'liansairank', 'joinmem', 'leavemem', 'chatnum', 'chuangongnum'
           ,'guild_type', 'low_maintain_state']

CONT_COLUMNS2 = ['cai_qi_value', 'today_online_cnt', 'total_count', 'scale', 'build_info', 'xuetu_num', 'member_num', 'zi_cai',
                'rq_value', 'fund', 'prosperity', 'guild_member_cnt', 'max_guild_member_cnt', 'max_xue_tu_member', 'full_degree',
                 'liansairank', 'joinmem', 'leavemem', 'chatnum', 'chuangongnum']

CATG_COLUMNS2 = ['guild_type', 'low_maintain_state','g_force']



# 玩家Embedding列
SRC_ROLE = ['role_id_src','src_role_level','src_vip_level','src_role_skill','src_role_practice','src_role_equip', 'src_total_score'] # 暂时只看这个
DST_ROLE = ['role_id_dst','dst_role_level','dst_vip_level', 'dst_role_skill','dst_role_practice', 'dst_role_equip', 'dst_total_score']

DAYS = 7 # 设置聚合的天数
RELATION_NUM =4 # 暂时考虑三种关系

# 定义图的输入参数
input_fec_embed = 6
hidden_size = 8
out_embedding = 8

# 批处理大小
BATCHSIZE = 32


# 训练相关参数
EPOCHS = 5 # main1 为20,过大的epoch会导致在局部数据上过拟合
train_losses = []  # 记录每epoch训练的平均loss
eval_losses = []  # 测试的
test_flag = False  # 为True时候加载保存好的model进行测试
log_dir = '../log/checkpoint'


# # 以下是为main2单独准备的一套,但是好像用不上
train_losses_epoch = []
loss1_traines_epoch = []
loss2_traines_epoch= []
loss3_traines_epoch= []
loss4_traines_epoch= []
loss5_traines_epoch= []
loss6_traines_epoch= []
eval_losses_epoch = []
loss1_evals_epoch = []
loss2_evals_epoch=[]
loss3_evals_epoch=[]
loss4_evals_epoch=[]
loss5_evals_epoch=[]
loss6_evals_epoch=[]
# #######

