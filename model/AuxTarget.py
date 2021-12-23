from EmbeddingReturn import *
from SelfAttentionLabel import *


class AuxTarget(torch.nn.Module):
    '''
    func:concat不共享
    '''
    def __init__(self,wide_dim,dense_dim,embed_dim,input_fec_embed, hidden_size, out_embedding):
        super(AuxTarget,self).__init__()
        self.embeds = EmbedReturn(wide_dim,dense_dim,embed_dim,input_fec_embed, hidden_size, out_embedding)
        self.self_attention = SelfAttentionLabel()
        self.repu_target = torch.nn.Linear(64,1)
        self.act_target = torch.nn.Linear(64,1) # 这里要设计一下，不然不同的y的范围不一样，会不会出问题。
        self.commu_target = torch.nn.Linear(64,1)
        self.reso_target = torch.nn.Linear(64,1)
        self.ability_target = torch.nn.Linear(64,1)
        self.combine = torch.nn.Linear(320+256,64) # 5*64+256 aggdim
        self.stab = torch.nn.Linear(64,1)
        self.dropout = torch.nn.Dropout(0.5)

        # self.classifier = nn.LogSoftmax(dim=1)
    def forward(self,x_catgs,x_conts,Gs,inputss,data_id_alls,vocab_to_ints,STEP,data_guild):
        out,repu_embed,act_embed,commu_embed,reso_embed,ability_embed = self.embeds(x_catgs,x_conts,Gs,inputss,data_id_alls,vocab_to_ints,STEP,data_guild)
        repu_target = self.repu_target(repu_embed) # 使用领导力替换
        act_target = self.act_target(act_embed)
        commu_target = self.commu_target(commu_embed)
        reso_target = self.reso_target(reso_embed)
        ability_target  = self.ability_target(ability_embed)
        dim_atten = self.self_attention(repu_embed,act_embed,commu_embed,reso_embed,ability_embed)
        all_embed = torch.cat((dim_atten, out), axis=1)
        stab_target = self.stab(self.dropout(self.combine(all_embed)))
        # print('sigmmoid,',repu_target)
        return stab_target,repu_target,act_target,commu_target,reso_target,ability_target



class AuxTarget2(torch.nn.Module):
    '''
    func: 六个子任务
    '''
    def __init__(self,wide_dim,dense_dim,embed_dim,input_fec_embed, hidden_size, out_embedding):
        super(AuxTarget2,self).__init__()
        self.embeds = EmbedReturn(wide_dim,dense_dim,embed_dim,input_fec_embed, hidden_size, out_embedding)
        self.self_attention = SelfAttentionLabel()
        self.repu_target = torch.nn.Linear(64,1)
        self.act_target = torch.nn.Linear(64,1) # 这里要设计一下，不然不同的y的范围不一样，会不会出问题。
        self.commu_target = torch.nn.Linear(64,1)
        self.reso_target = torch.nn.Linear(64,1)
        self.ability_target = torch.nn.Linear(64,1)
        self.combine = torch.nn.Linear(256,64) # 5*64+256 aggdim
        self.stab = torch.nn.Linear(64,1)
        # self.dropout = torch.nn.Dropout(0.5)

        # self.classifier = nn.LogSoftmax(dim=1)
    def forward(self,x_catgs,x_conts,Gs,inputss,data_id_alls,vocab_to_ints,STEP,data_guild):
        out,repu_embed,act_embed,commu_embed,reso_embed,ability_embed = self.embeds(x_catgs,x_conts,Gs,inputss,data_id_alls,vocab_to_ints,STEP,data_guild)
        repu_target = self.repu_target(repu_embed) # 使用领导力替换
        act_target = self.act_target(act_embed)
        commu_target = self.commu_target(commu_embed)
        reso_target = self.reso_target(reso_embed)
        ability_target  = self.ability_target(ability_embed)
        dim_atten = self.combine(out)
        stab_target = self.stab(dim_atten)
        # print('sigmmoid,',repu_target)
        return stab_target,repu_target,act_target,commu_target,reso_target,ability_target


class AuxTarget3(torch.nn.Module):
    '''
    func:高层共享
    '''

    def __init__(self, wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding):
        super(AuxTarget3, self).__init__()
        self.embeds = EmbedReturn(wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding)
        self.self_attention = SelfAttentionLabel()
        self.share_weight = torch.nn.Linear(64, 1)
        self.repu_target = self.share_weight
        self.act_target = self.share_weight
        self.commu_target = self.share_weight
        self.reso_target = self.share_weight
        self.ability_target = self.share_weight
        self.combine = torch.nn.Linear(320, 64)  # 5*64+256 aggdim
        self.stab = torch.nn.Linear(64, 1)
        self.dropout = torch.nn.Dropout(0.5)

        # self.classifier = nn.LogSoftmax(dim=1)

    def forward(self, x_catgs, x_conts, Gs, inputss, data_id_alls, vocab_to_ints, STEP, data_guild):
        out, repu_embed, act_embed, commu_embed, reso_embed, ability_embed = self.embeds(x_catgs, x_conts, Gs, inputss,
                                                                                         data_id_alls, vocab_to_ints,
                                                                                         STEP, data_guild)
        repu_target = self.repu_target(repu_embed)
        act_target = self.act_target(act_embed)
        commu_target = self.commu_target(commu_embed)
        reso_target = self.reso_target(reso_embed)
        ability_target = self.ability_target(ability_embed)
        dim_atten = self.self_attention(repu_embed, act_embed, commu_embed, reso_embed, ability_embed)
        # all_embed = torch.cat((dim_atten, out), axis=1)
        stab_target = self.stab(self.dropout(self.combine(dim_atten)))
        return stab_target, repu_target, act_target, commu_target, reso_target, ability_target


class AuxTarget4(torch.nn.Module):
    '''
    func:直接concat
    '''

    def __init__(self, wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding):
        super(AuxTarget4, self).__init__()
        self.embeds = EmbedReturn(wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding)
        self.self_attention = SelfAttentionLabel()
        self.repu_target = torch.nn.Linear(64, 1)
        self.act_target = torch.nn.Linear(64, 1)  # 这里要设计一下，不然不同的y的范围不一样，会不会出问题。
        self.commu_target = torch.nn.Linear(64, 1)
        self.reso_target = torch.nn.Linear(64, 1)
        self.ability_target = torch.nn.Linear(64, 1)
        self.combine = torch.nn.Linear(320, 64)  # 5*64+256 aggdim
        self.stab = torch.nn.Linear(64, 1)
        self.dropout = torch.nn.Dropout(0.5)

        # self.classifier = nn.LogSoftmax(dim=1)

    def forward(self, x_catgs, x_conts, Gs, inputss, data_id_alls, vocab_to_ints, STEP, data_guild):
        out, repu_embed, act_embed, commu_embed, reso_embed, ability_embed = self.embeds(x_catgs, x_conts, Gs, inputss,
                                                                                         data_id_alls, vocab_to_ints,
                                                                                         STEP, data_guild)
        repu_target = self.repu_target(repu_embed)  # 使用领导力替换
        act_target = self.act_target(act_embed)
        commu_target = self.commu_target(commu_embed)
        reso_target = self.reso_target(reso_embed)
        ability_target = self.ability_target(ability_embed)
        dim_atten = self.self_attention(repu_embed, act_embed, commu_embed, reso_embed, ability_embed)
        stab_target = self.stab(self.dropout(self.combine(dim_atten)))
        # print('sigmmoid,',repu_target)
        return stab_target, repu_target, act_target, commu_target, reso_target, ability_target

################################################################################


class AuxTarget5(torch.nn.Module):
    '''
    func:高层共享+concat
    '''

    def __init__(self, wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding):
        super(AuxTarget5, self).__init__()
        self.embeds = EmbedReturn(wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding)
        self.self_attention = SelfAttentionLabel()
        self.share_weight = torch.nn.Linear(64, 1)
        self.repu_target = self.share_weight
        self.act_target = self.share_weight
        self.commu_target = self.share_weight
        self.reso_target = self.share_weight
        self.ability_target = self.share_weight
        self.combine = torch.nn.Linear(320+256, 64)  # 5*64+256 aggdim
        self.stab = torch.nn.Linear(64, 1)
        self.dropout = torch.nn.Dropout(0.5)

        # self.classifier = nn.LogSoftmax(dim=1)

    def forward(self, x_catgs, x_conts, Gs, inputss, data_id_alls, vocab_to_ints, STEP, data_guild):
        out, repu_embed, act_embed, commu_embed, reso_embed, ability_embed = self.embeds(x_catgs, x_conts, Gs, inputss,
                                                                                         data_id_alls, vocab_to_ints,
                                                                                         STEP, data_guild)
        repu_target = self.repu_target(repu_embed)
        act_target = self.act_target(act_embed)
        commu_target = self.commu_target(commu_embed)
        reso_target = self.reso_target(reso_embed)
        ability_target = self.ability_target(ability_embed)
        dim_atten = self.self_attention(repu_embed, act_embed, commu_embed, reso_embed, ability_embed)
        all_embed = torch.cat((dim_atten, out), axis=1)
        stab_target = self.stab(self.dropout(self.combine(all_embed)))
        return stab_target, repu_target, act_target, commu_target, reso_target, ability_target






class AuxTarget6(torch.nn.Module):
    '''
    func:多加入一层共享层，然后concat
    '''

    def __init__(self, wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding):
        super(AuxTarget6, self).__init__()
        self.embeds = EmbedReturn(wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding)
        self.self_attention = SelfAttentionLabel()
        self.share_weight = torch.nn.Linear(64, 64)
        self.repu_target = torch.nn.Linear(64, 1)
        self.act_target = torch.nn.Linear(64, 1)
        self.commu_target = torch.nn.Linear(64, 1)
        self.reso_target = torch.nn.Linear(64, 1)
        self.ability_target = torch.nn.Linear(64, 1)
        self.combine = torch.nn.Linear(320+256, 64)  # 5*64+256 aggdim
        self.stab = torch.nn.Linear(64, 1)
        self.dropout = torch.nn.Dropout(0.5)

        # self.classifier = nn.LogSoftmax(dim=1)

    def forward(self, x_catgs, x_conts, Gs, inputss, data_id_alls, vocab_to_ints, STEP, data_guild):
        out, repu_embed, act_embed, commu_embed, reso_embed, ability_embed = self.embeds(x_catgs, x_conts, Gs, inputss,
                                                                                         data_id_alls, vocab_to_ints,
                                                                                         STEP, data_guild)
        repu_target = self.repu_target(self.share_weight(repu_embed))
        act_target = self.act_target(self.share_weight(act_embed))
        commu_target = self.commu_target(self.share_weight(commu_embed))
        reso_target = self.reso_target(self.share_weight(reso_embed))
        ability_target = self.ability_target(self.share_weight(ability_embed))
        dim_atten = self.self_attention(repu_embed, act_embed, commu_embed, reso_embed, ability_embed)
        all_embed = torch.cat((dim_atten, out), axis=1)
        stab_target = self.stab(self.dropout(self.combine(all_embed)))
        return stab_target, repu_target, act_target, commu_target, reso_target, ability_target



class AuxTarget7(torch.nn.Module):
    '''
    func:多加入一层共享层，然后concat,不过这里是shareweightconcat
    '''

    def __init__(self, wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding):
        super(AuxTarget7, self).__init__()
        self.embeds = EmbedReturn(wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding)
        self.self_attention = SelfAttentionLabel()
        self.share_weight = torch.nn.Linear(64, 64)
        self.repu_target = torch.nn.Linear(64, 1)
        self.act_target = torch.nn.Linear(64, 1)
        self.commu_target = torch.nn.Linear(64, 1)
        self.reso_target = torch.nn.Linear(64, 1)
        self.ability_target = torch.nn.Linear(64, 1)
        self.combine = torch.nn.Linear(320+256, 64)  # 5*64+256 aggdim
        self.stab = torch.nn.Linear(64, 1)
        self.dropout = torch.nn.Dropout(0.5)

        # self.classifier = nn.LogSoftmax(dim=1)

    def forward(self, x_catgs, x_conts, Gs, inputss, data_id_alls, vocab_to_ints, STEP, data_guild):
        out, repu_embed, act_embed, commu_embed, reso_embed, ability_embed = self.embeds(x_catgs, x_conts, Gs, inputss,
                                                                                         data_id_alls, vocab_to_ints,
                                                                                         STEP, data_guild)
        tmp1 = self.share_weight(repu_embed)
        repu_target = self.repu_target(tmp1)
        tmp2 = self.share_weight(act_embed)
        act_target = self.act_target(tmp2)
        tmp3 = self.share_weight(commu_embed)
        commu_target = self.commu_target(tmp3)
        tmp4 = self.share_weight(reso_embed)
        reso_target = self.reso_target(tmp4)
        tmp5 = self.share_weight(ability_embed)
        ability_target = self.ability_target(tmp5)
        dim_atten = self.self_attention(tmp1,tmp2,tmp3,tmp4,tmp5)
        all_embed = torch.cat((dim_atten, out), axis=1)
        stab_target = self.stab(self.dropout(self.combine(all_embed)))
        return stab_target, repu_target, act_target, commu_target, reso_target, ability_target



class AuxTarget8(torch.nn.Module):
    '''
    func:多加入一层共享层，然后concat,detach
    '''

    def __init__(self, wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding):
        super(AuxTarget8, self).__init__()
        self.embeds = EmbedReturn(wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding)
        self.self_attention = SelfAttentionLabel()
        self.share_weight = torch.nn.Linear(64, 64)
        self.repu_target = torch.nn.Linear(64, 1)
        self.act_target = torch.nn.Linear(64, 1)
        self.commu_target = torch.nn.Linear(64, 1)
        self.reso_target = torch.nn.Linear(64, 1)
        self.ability_target = torch.nn.Linear(64, 1)
        self.combine = torch.nn.Linear(320+256, 64)  # 5*64+256 aggdim
        self.stab = torch.nn.Linear(64, 1)
        self.dropout = torch.nn.Dropout(0.5)

        # self.classifier = nn.LogSoftmax(dim=1)

    def forward(self, x_catgs, x_conts, Gs, inputss, data_id_alls, vocab_to_ints, STEP, data_guild):
        out, repu_embed, act_embed, commu_embed, reso_embed, ability_embed = self.embeds(x_catgs, x_conts, Gs, inputss,
                                                                                         data_id_alls, vocab_to_ints,
                                                                                         STEP, data_guild)
        tmp1 = self.share_weight(repu_embed)
        repu_target = self.repu_target(tmp1)
        tmp2 = self.share_weight(act_embed)
        act_target = self.act_target(tmp2)
        tmp3 = self.share_weight(commu_embed)
        commu_target = self.commu_target(tmp3)
        tmp4 = self.share_weight(reso_embed)
        reso_target = self.reso_target(tmp4)
        tmp5 = self.share_weight(ability_embed)
        ability_target = self.ability_target(tmp5)
        dim_atten = self.self_attention(tmp1,tmp2,tmp3,tmp4,tmp5).detach()
        all_embed = torch.cat((dim_atten, out), axis=1)
        stab_target = self.stab(self.dropout(self.combine(all_embed)))
        return stab_target, repu_target, act_target, commu_target, reso_target, ability_target



class AuxTarget9(torch.nn.Module):
    '''
    func:GCN benchmark
    '''
    def __init__(self,wide_dim,dense_dim,embed_dim,input_fec_embed, hidden_size, out_embedding):
        super(AuxTarget9,self).__init__()
        self.embeds = EmbedReturn2(wide_dim,dense_dim,embed_dim,input_fec_embed, hidden_size, out_embedding)
        self.lin1 = torch.nn.Linear(384,1)
        self.lin2 = torch.nn.Linear(64, 1)
        self.lin3 = torch.nn.Linear(64, 1)
        self.lin4 = torch.nn.Linear(64, 1)
        self.lin5 = torch.nn.Linear(64, 1)
        self.lin6 = torch.nn.Linear(64,1)

    def forward(self,x_catgs,x_conts,Gs,inputss,data_id_alls,vocab_to_ints,STEP,data_guild):
        out,repu_embed,act_embed,commu_embed,reso_embed,ability_embed = self.embeds(x_catgs,x_conts,Gs,inputss,data_id_alls,vocab_to_ints,STEP,data_guild)
        repu_target = self.lin6(repu_embed)
        act_target = self.lin5(act_embed)
        commu_target = self.lin4(commu_embed)
        reso_target = self.lin3(reso_embed)
        ability_target  = self.lin2(ability_embed)
        stab_target = self.lin1(out)
        # print('sigmmoid,',repu_target)
        return stab_target,repu_target,act_target,commu_target,reso_target,ability_target


