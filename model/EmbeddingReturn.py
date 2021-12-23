from AggEmbedding import *

# class EmbedReturn(torch.nn.Module):
#     '''
#     func:生成各部分emebeding
#     '''
#
#     def __init__(self, wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding):
#         super(EmbedReturn, self).__init__()
#         self.agg_embed = AggEmbedding(wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding)
#         self.transform = Transformer()
#         self.reputation = torch.nn.Linear(256, 64)  # 这里是reputation，这个和agg_dim一样大 参数共享是不是该在这里设置啊
#         self.activity = torch.nn.Linear(256, 64)  # Activity
#         self.congnitive = torch.nn.Linear(256, 64)  # Congnitive
#         self.resource = torch.nn.Linear(256, 64)  # Resource
#         self.diverse = torch.nn.Linear(256, 64)  # diverse
#
#     def forward(self, x_catgs, x_conts, Gs, inputss, data_id_alls, vocab_to_ints, STEP,data_guild):
#         out = self.agg_embed(x_catgs, x_conts, Gs, inputss, data_id_alls, vocab_to_ints,
#                              STEP,data_guild)  # [32, 7, 384]，【bs，seq_len,feature_dim】
#         out = self.transform(out)  # [32, 256]
#         repu_embed = self.reputation(out)  # [32,64]
#         act_embed = self.activity(out)
#         cong_embed = self.congnitive(out)
#         reso_embed = self.resource(out)
#         div_embed = self.diverse(out)
#         return out, repu_embed, act_embed, cong_embed, reso_embed, div_embed


class EmbedReturn(torch.nn.Module):
    '''
    func:曾经低sharing weights
    '''

    def __init__(self, wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding):
        super(EmbedReturn, self).__init__()
        self.agg_embed = AggEmbedding(wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding)
        self.transform = Transformer()
        self.reputation = torch.nn.Linear(256, 64)  # 这里是reputation，这个和agg_dim一样大 参数共享是不是该在这里设置啊
        self.activity = torch.nn.Linear(256, 64)  # Activity
        self.congnitive = torch.nn.Linear(256, 64)  # Congnitive
        self.resource = torch.nn.Linear(256, 64)  # Resource
        self.diverse = torch.nn.Linear(256, 64)  # diverse

    def forward(self, x_catgs, x_conts, Gs, inputss, data_id_alls, vocab_to_ints, STEP,data_guild):
        out = self.agg_embed(x_catgs, x_conts, Gs, inputss, data_id_alls, vocab_to_ints,
                             STEP,data_guild)  # [32, 7, 384]，【bs，seq_len,feature_dim】
        out = self.transform(out)  # [32, 256]，直接取torch.mean()在这里取
        repu_embed = self.reputation(out)  # [32,64]
        act_embed = self.activity(out)
        cong_embed = self.congnitive(out)
        reso_embed = self.resource(out)
        div_embed = self.diverse(out)
        return out, repu_embed, act_embed, cong_embed, reso_embed, div_embed



class EmbedReturn2(torch.nn.Module):
    '''
    func:GCN benmark
    '''

    def __init__(self, wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding):
        super(EmbedReturn2, self).__init__()
        self.agg_embed = AggEmbedding2(wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding)
        self.reputation = torch.nn.Linear(384, 64)  # 这里是reputation，这个和agg_dim一样大 参数共享是不是该在这里设置啊
        self.activity = torch.nn.Linear(384, 64)  # Activity
        self.congnitive = torch.nn.Linear(384, 64)  # Congnitive
        self.resource = torch.nn.Linear(384, 64)  # Resource
        self.diverse = torch.nn.Linear(384, 64)  # diverse

    def forward(self, x_catgs, x_conts, Gs, inputss, data_id_alls, vocab_to_ints, STEP,data_guild):
        out = self.agg_embed(x_catgs, x_conts, Gs, inputss, data_id_alls, vocab_to_ints,
                             STEP,data_guild)  # [32, 384]，【bs，seq_len,feature_dim】
        repu_embed = self.reputation(out)  # [32,64]
        act_embed = self.activity(out)
        cong_embed = self.congnitive(out)
        reso_embed = self.resource(out)
        div_embed = self.diverse(out)
        return out, repu_embed, act_embed, cong_embed, reso_embed, div_embed
