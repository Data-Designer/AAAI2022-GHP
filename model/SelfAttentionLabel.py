import torch
class SelfAttentionLabel(torch.nn.Module):
    '''
    func:聚合特定含义的embedding层
    '''

    def __init__(self):
        super(SelfAttentionLabel, self).__init__()

    def forward(self, repu_embed, act_embed, cong_embed, reso_embed, div_embed):
        return torch.cat((repu_embed, act_embed, cong_embed, reso_embed, div_embed), axis=1)
