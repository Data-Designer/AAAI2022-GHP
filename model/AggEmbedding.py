from WideAndDeep import *
from GraphEmbedding import *
from Transformer import *

class AggEmbedding(torch.nn.Module):
    '''
    func:按天进行图和Profile的聚合,两步，聚合一天的profile和关系，返回七天的不同的向量
    '''
    def __init__(self,wide_dim,dense_dim,embed_dim,input_fec_embed, hidden_size, out_embedding):
        super(AggEmbedding,self).__init__()
        self.wide_and_deep = torch.nn.ModuleList([WideAndDeep(wide_dim,dense_dim,embed_dim) for i in range(DAYS)])
        self.graph_node = torch.nn.ModuleList([GraphNodeEmbedding(input_fec_embed, hidden_size, out_embedding) for i in range(DAYS)])
        self.attention = MultiHeadAttention(64, 8, 0.5) # 增设att层，64是embedding_size，8是head,0.5是dropout 改

    def forward(self,x_catgs,x_conts,Gs,inputss,data_id_alls,vocab_to_ints,STEP,data_guild):
        agg_embeds = torch.zeros(data_guild['guild_id'][BATCHSIZE*STEP:BATCHSIZE*(STEP+1)].shape[0],DAYS,384)
        for i in range(0,DAYS):
            profile = self.wide_and_deep[i](x_catgs[i],x_conts[i])
            graph = self.graph_node[i](Gs[i],inputss[i],data_id_alls[i],vocab_to_ints[i],STEP,data_guild) #【4，64】，attention问题直接view这个graph,加上一个layer转换大小！
            tmp1 = torch.cat([profile.unsqueeze(dim=0),graph],dim=0) # [5,embedding_size=64]
            graphattn = self.attention(tmp1) # [5,32,64] # attention4个的话，变更这里
            graph = graphattn.view(graphattn.size(1), -1) # [32,320] 按照batch维度展开=5*64 = 320
            tmp = (torch.cat((profile,graph),axis = 1)) # 一天的【32，1，384】
            agg_embeds[:,i,:] = tmp # 存入七天的，准备送入Trans

        return agg_embeds # [32,7,384] ，160+256


class AggEmbedding1(torch.nn.Module):
    '''
    func:图自己玩，需要对照修改Transformer
    '''
    def __init__(self,wide_dim,dense_dim,embed_dim,input_fec_embed, hidden_size, out_embedding):
        super(AggEmbedding1,self).__init__()
        self.wide_and_deep = torch.nn.ModuleList([WideAndDeep(wide_dim,dense_dim,embed_dim) for i in range(DAYS)])
        self.graph_node = torch.nn.ModuleList([GraphNodeEmbedding(input_fec_embed, hidden_size, out_embedding) for i in range(DAYS)])
        self.attention = MultiHeadAttention(64, 8, 0.5) # 增设att层，64是embedding_size，8是head,0.5是dropout 改

    def forward(self,x_catgs,x_conts,Gs,inputss,data_id_alls,vocab_to_ints,STEP,data_guild):
        agg_embeds = torch.zeros(data_guild['guild_id'][BATCHSIZE*STEP:BATCHSIZE*(STEP+1)].shape[0],DAYS,320)
        for i in range(0,DAYS):
            profile = self.wide_and_deep[i](x_catgs[i],x_conts[i])
            graph = self.graph_node[i](Gs[i],inputss[i],data_id_alls[i],vocab_to_ints[i],STEP,data_guild) #【4，64】，attention问题直接view这个graph,加上一个layer转换大小！
            tmp1 = graph
            graphattn = self.attention(tmp1) # [4,32,64] # attention4个的话，变更这里
            graph = graphattn.view(graphattn.size(1), -1) # [32,256] 按照batch维度展开=5*64 = 320
            tmp = (torch.cat((profile,graph),axis = 1)) # 一天的【32，1，320】
            agg_embeds[:,i,:] = tmp # 存入七天的，准备送入Trans
        return agg_embeds # [32,7,320] ，160+256




class AggEmbedding2(torch.nn.Module):
    '''
    func:GCN bencmark
    '''
    def __init__(self,wide_dim,dense_dim,embed_dim,input_fec_embed, hidden_size, out_embedding):
        super(AggEmbedding2,self).__init__()
        self.wide_and_deep = torch.nn.ModuleList([WideAndDeep(wide_dim,dense_dim,embed_dim) for i in range(DAYS)])
        self.graph_node = torch.nn.ModuleList([GraphNodeEmbedding(input_fec_embed, hidden_size, out_embedding) for i in range(DAYS)])
        self.attention = MultiHeadAttention(64, 8, 0.5) # 增设att层，64是embedding_size，8是head,0.5是dropout 改

    def forward(self,x_catgs,x_conts,Gs,inputss,data_id_alls,vocab_to_ints,STEP,data_guild):
        agg_embeds = torch.zeros(data_guild['guild_id'][BATCHSIZE*STEP:BATCHSIZE*(STEP+1)].shape[0],DAYS,384)
        for i in range(0,DAYS):
            profile = self.wide_and_deep[i](x_catgs[i],x_conts[i])
            graph = self.graph_node[i](Gs[i],inputss[i],data_id_alls[i],vocab_to_ints[i],STEP,data_guild) #【4，64】，attention问题直接view这个graph,加上一个layer转换大小！
            tmp1 = torch.cat([profile.unsqueeze(dim=0),graph],dim=0) # [5,embedding_size=64]
            graphattn = self.attention(tmp1) # [5,32,64] # attention4个的话，变更这里
            graph = graphattn.view(graphattn.size(1), -1) # [32,320] 按照batch维度展开=5*64 = 320
            tmp = (torch.cat((profile,graph),axis = 1)) # 一天的【32，1，384】
            agg_embeds[:,i,:] = tmp # 存入七天的，准备送入Trans
        agg_embeds = torch.mean(agg_embeds,dim=1).cuda()
        return agg_embeds # [32,384] ，160+256

