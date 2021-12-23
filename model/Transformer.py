import torch.nn as nn
import torch.nn.functional as F
import copy
from config.utils import *


# Transformer Encoder + CNN / LSTM /GRU
class PositionalEncoding(nn.Module):
    '''
    func:位置编码
    pad_size: seq_len,我们这里都是7天
    embed:day_embed
    '''
    def __init__(self, embed, pad_size, dropout):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])   # 偶数sin
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])   # 奇数cos
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 加上位置信息
        out = x + nn.Parameter(self.pe, requires_grad=False).to(device) # [bs,seq_len,all_head_dim]
        out = self.dropout(out)
        return out

class ScaledDotProductAttention(nn.Module):
    '''
    func:点积注意力
    '''
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # [bs*heads,seq_len,singel_head_dim] * [bs*heads,singel_head_dim,seq_len]
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context ## [bs*heads,seq_len,singel_head_dim]


class MultiHeadAttention(nn.Module):
    '''
    func: 多头注意力
    dim_model:day embedding
    '''
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0    # head数必须能够整除隐层大小
        self.dim_head = dim_model // self.num_head   # 单头维度
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)  # 放缩回原来大小
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model) # 定义都无视bs
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)   # 自带的LayerNorm方法

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x) # 转换为Q,K，V
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)  # [bs*heads,seq_len,singel_head_dim]
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5  # 根号dk分之一，对应Scaled操作
        context = self.attention(Q, K, V, scale) # [bs*heads,seq_len,singel_head_dim]
        context = context.view(batch_size, -1, self.dim_head * self.num_head) # [bs,seq_len,all_head_dim]
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x     # 残差连接
        out = self.layer_norm(out)  # [bs,seq_len,all_head_dim]
        return out


class PositionwiseFeedForward(nn.Module):
    '''
    func: FeedForward层
    '''
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)   #  [bs,seq_len,all_head_dim]
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(dim_model, num_head, dropout)
        self.feed_forward = PositionwiseFeedForward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x) # [bs,seq_len,all_head_dim]
        out = self.feed_forward(out) #  [bs,seq_len,all_head_dim]
        return out


class ConfigTrans(object):
    '''
    func:配置模型参数
    '''
    def __init__(self):
        self.model_name = 'Transformer'
        self.dropout = 0.5
        self.agg_dim = 256                      # 总的aggdim,改
        # self.batch_size = 128             # mini-batch大小
        self.pad_size = 7                     # 每句话处理成的长度(短填长切)，我们是固定的7天
        self.embed = 384         # 字向量维度，用于位置编码 64+64*5，改graph的时候这里换成320
        self.dim_model = 384      # 字向量维度需要与embed一样,只不过用于Encoder，这里换成320
        self.hidden = 256
        # self.last_hidden = 512
        self.num_head = 8       # 多头注意力，注意需要整除
        self.num_encoder = 2    # 使用两个Encoder，数据量较少我怕过拟合
config = ConfigTrans()


class Transformer(nn.Module):
    '''
    fun:返回七天的聚合
    '''
    def __init__(self):
        super(Transformer, self).__init__()
        self.postion_embedding = PositionalEncoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])   # 多次Encoder
        self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.agg_dim)

    def forward(self, x):
        out = self.postion_embedding(x.to(device))
        for encoder in self.encoders:
            out = encoder(out) # [bs,seq_len,all_head_dim]
        out = out.view(out.size(0), -1)  # [bs,seq_len*all_head_dim]
        # out = torch.mean(out, 1)    # 也可用池化来做
        out = self.fc1(out)
        return out # agg_dim
