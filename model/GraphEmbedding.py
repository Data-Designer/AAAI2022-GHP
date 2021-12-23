from GCN import *
from config.utils import *


class SingleGraphNodeEmbedding(torch.nn.Module):
    '''
    func:一天的Batchsize个帮会某种的关系表征，比如32个帮会的1天的聊天关系表征
    '''

    def __init__(self, input_fec_embed, hidden_size, out_embedding):
        super(SingleGraphNodeEmbedding, self).__init__()
        self.net = GCN(input_fec_embed, hidden_size, out_embedding)
        self.lin1 = torch.nn.Linear(8, 64) # 改

    def forward(self, G, inputs, data_id_all, vocab_to_int, STEP,data_guild): # 问题出在data_guild上，这里没有替换
        logits = self.net(G, inputs)
        # print(data_guild.shape) # 测试是不是在testing时候还是那个data_guild而不是data_guild_valid
        guild_role_embeds = torch.zeros(data_guild['guild_id'][BATCHSIZE * STEP:BATCHSIZE * (STEP + 1)].shape[0], 8)
        for i, guild_id in enumerate(data_guild['guild_id'][BATCHSIZE * STEP:BATCHSIZE * (STEP + 1)]):
            role_lis, _ = preprocessnewid(vocab_to_int, set(data_id_all[data_id_all[:, 1] == guild_id][:, 0]),
                                          [])  # 获取到一个帮会所有成员lis
            if not torch.isnan(torch.mean(logits[role_lis], dim=0)[0]):  # 池化，取平均出现nan值,填0
                guild_role_embeds[i] = torch.mean(logits[role_lis], dim=0)  # 第i个帮会的某种关系的embedding
            else:
                continue
        out = self.lin1(guild_role_embeds.to(device)) # mean过需要重新device
        return out  # [Batchsize,embedding_size],[32,64]



class GraphNodeEmbedding(torch.nn.Module):
    '''
    func:返回一天Batchsize个帮会的4种关系表征【4，Batchsize,embeddingsize】
    '''

    def __init__(self, input_fec_embed, hidden_size, out_embedding):
        super(GraphNodeEmbedding, self).__init__()
        self.nets = torch.nn.ModuleList(
            [SingleGraphNodeEmbedding(input_fec_embed, hidden_size, out_embedding) for i in range(RELATION_NUM)])

    def forward(self, Gs, inputss, data_id_alls, vocab_to_ints, STEP,data_guild):
        # 注意这里输入存为列表会方便一点，且这里没有问题，这里的Gs已经是Gs[i]了
        chatnet_emb = self.nets[0](Gs[0], inputss[0], data_id_alls[0], vocab_to_ints[0], STEP,data_guild)
        tradenet_emb = self.nets[1](Gs[1], inputss[1], data_id_alls[1], vocab_to_ints[1], STEP,data_guild)
        friendnet_emb = self.nets[2](Gs[2], inputss[2], data_id_alls[2], vocab_to_ints[2], STEP,data_guild)
        team_emb = self.nets[3](Gs[3], inputss[3], data_id_alls[3], vocab_to_ints[3], STEP,data_guild)
        all_relation = torch.stack((chatnet_emb, tradenet_emb, friendnet_emb, team_emb), dim=0)
        return all_relation  # 【4，Batchsize,embeddingsize】


