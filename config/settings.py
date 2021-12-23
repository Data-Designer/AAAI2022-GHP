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


# 查询某天的guildinfo ds ds_end ds_end，需要过滤
sql_guildinfo = """
        select * from up_nsh_ads.ads_nsh_guildhealth_guildinfo_add_d
        where ds = '{}' and guild_id in (
                -- 按最后一天id进行过滤
                select guild_id
                from up_nsh_ads.ads_nsh_guildhealth_guildinfo_add_d
                where ds = '{}' and DATEDIFF(ds,create_date)>=14
        )
        order by guild_id
    """


# 查询某天的friend-relation，ds
sql_friend = """
        select * 
        from up_nsh_ads.ads_nsh_guildhealth_relation_friend_add_d
        where ds = '{}'
    """

# 查询某天的chat relation,ds
sql_chat= """
        select * 
        from up_nsh_ads.ads_nsh_guildhealth_relation_add_d
        where ds = '{}'
    """

# 查询某天的trade relation,ds
sql_trade = """
        select * 
        from up_nsh_ads.ads_nsh_guildhealth_relation_trade_add_d
        where ds = '{}'
"""

# 查询某天的team relation ds
sql_team = """
        select * 
        from up_nsh_ads.ads_nsh_guildhealth_team_add_d
        where ds = '{}'
"""


# label,后七天的label,同样需要过滤 ds_end;avg((joinmem+leavemem)/total_count) 原方案 且这里换成full_degree,fund量级太大了，这两者之间呈现正相关
# sign(avg((joinmem-leavemem))) as stab
# sql_label1 = '''
#         select guild_id,log2(abs(sum(chatnum))+1) as commu,max(liansairank) as repu,log2(abs(sum(today_online_cnt))+1) as act,log2(abs(avg(ability))+1) as ability,log2(abs(sum(full_degree))+1) as reso, log2(abs(sum((joinmem+leavemem)/total_count))+1) as stab
#         from up_nsh_ads.ads_nsh_guildhealth_guildinfo_add_d
#         where ds between date_sub('{}',7) and '{}' and guild_id in (
#                 -- 按最后一天id进行过滤
#                 select guild_id
#                 from up_nsh_ads.ads_nsh_guildhealth_guildinfo_add_d
#                 where ds = '{}' and DATEDIFF(ds,create_date)>=14
#         )
#         group by guild_id
#         order by guild_id
# '''


sql_label ='''
        select guild_id,commu,repu,act,ability,reso,stab,leadership from
        (
        --加入abs和1防止有脏数据, correlation
            select max(leader_id) as leader_id,guild_id,log2(abs(avg(nvl(chatnum,0)))+1) as commu,nvl(max(liansairank),0) as repu,log2(abs(avg(nvl(today_online_cnt,0)))+1) as act,log2(abs(avg(nvl(ability,0)))+1) as ability,log2(abs(sum(nvl(fund,0)))+1) as reso, log2(abs(sum((nvl(joinmem,0)-nvl(leavemem,0))+total_count))+1) as stab
            from up_nsh_ads.ads_nsh_guildhealth_guildinfo_add_d
            where ds between date_sub('{}',7) and '{}' and guild_id in (
                    -- 按最后一天id进行过滤
                    select guild_id
                    from up_nsh_ads.ads_nsh_guildhealth_guildinfo_add_d
                    where ds = '{}' and DATEDIFF(ds,create_date)>=14
            )
        group by guild_id
        ) as label
        left join
        (
            select role_id,log2(abs(avg(nvl(role_total_score,0)))+1) as leadership from 
            luoge_nsh_mid.mid_role_portrait_all_d
            where ds between date_sub('{}',7) and '{}'
            group by role_id
        ) as score
        on label.leader_id = score.role_id
        -- where repu !=0
        -- 比！=快多了
        -- where ability>=13 and stab<=1 -- 过滤掉一部分偏差大的数据
        order by guild_id
'''
