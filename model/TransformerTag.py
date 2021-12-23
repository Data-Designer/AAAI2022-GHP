# 1. 导入相关的包
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from random import shuffle
from torch.utils.data import TensorDataset
from data.Data_copy import getdatabench
from data.SQL import get_date_lista
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('../log/runsCNN/healthy_analysis')

def dataprepare():
    # 数据准备
    ds_ends = get_date_lista(begin_date='2020-11-01', end_date='2021-02-05')  # 后期可以直接transplit
    shuffle(ds_ends)  # 不打乱会出现20次为周期的下降
    ds_ends.remove('2020-12-20')  # 本来在error里的，减少IO
    ds_ends.remove('2020-12-06')  # 本来在error里的，减少IO
    ds_ends.remove('2020-12-13')  # 本来在error里的，减少IO
    ds_ends.remove('2020-12-27')  # 本来在error里的，减少IO
    ds_ends.remove('2020-11-22')  # 本来在error里的，减少IO
    ds_ends.remove('2020-11-01')  # 本来在error里的，减少IO
    # 将切片数据整合返回
    label_stabs = torch.Tensor([])
    label_repus = torch.Tensor([])
    label_acts  = torch.Tensor([])
    label_commus = torch.Tensor([])
    label_resos = torch.Tensor([])
    label_abilitys = torch.Tensor([])
    xs = torch.Tensor([])
    for ds_end in ds_ends:
        print(ds_end)
        data_guild, x_catgs,x_conts,wide_dim,dense_dim,embed_dim,\
                   label_stab,label_repu,label_act,label_commu,label_reso,label_ability  = getdatabench(ds_end)
        x_catgs = torch.stack([i for i in x_catgs],dim = 0) # 七天直接concat[7,36]
        x_conts = torch.stack([i for i in x_conts],dim = 0)
        # print(x_catgs.shape)
        x = torch.cat([x_catgs,x_conts],dim = 2).cpu() # 总特征[7,数据量,总特征维度数]
        xs = torch.cat([xs,x],dim = 1) # 合并为总的
        label_stabs = torch.cat([label_stabs,label_stab.cpu()],dim = 0)
        label_repus = torch.cat([label_repus,label_repu.cpu()],dim = 0)
        label_acts = torch.cat([label_acts,label_act.cpu()],dim = 0)
        label_commus = torch.cat([label_commus,label_commu.cpu()],dim = 0)
        label_resos = torch.cat([label_resos,label_reso.cpu()],dim = 0)
        label_abilitys = torch.cat([label_abilitys,label_ability.cpu()],dim = 0)
    # 转换为numpy
    xs = xs.permute(1,0,2).numpy() # [num,7,68]
    label_stabs = label_stabs.view(-1,).numpy()
    label_repus = label_repus.view(-1,).numpy()
    label_acts = label_acts.view(-1,).numpy()
    label_commus = label_commus.view(-1,).numpy()
    label_resos =label_resos.view(-1,).numpy()
    label_abilitys = label_abilitys.view(-1,).numpy()
    return xs,label_stabs,label_repus,label_acts,label_commus,label_resos,label_abilitys



class Positional_Encoding(nn.Module):
    '''
    params: embed-->word embedding dim      pad_size-->max_sequence_lenght
    Input: x
    Output: x + position_encoder
    '''

    def __init__(self, embed, pad_size, dropout):
        super(Positional_Encoding, self).__init__()
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])  # 偶数sin
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])  # 奇数cos
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 单词embedding与位置编码相加，这两个张量的shape一致
        out = x + nn.Parameter(self.pe, requires_grad=False).cuda()
        out = self.dropout(out)
        return out


class Multi_Head_Attention(nn.Module):
    '''
    params: dim_model-->hidden dim      num_head
    '''

    def __init__(self, dim_model, num_head, dropout=0.5):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0  # head数必须能够整除隐层大小
        self.dim_head = dim_model // self.num_head  # 按照head数量进行张量均分
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)  # Q，通过Linear实现张量之间的乘法，等同手动定义参数W与之相乘
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)  # 自带的LayerNorm方法

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1,
                   self.dim_head)  # reshape to batch*head*sequence_length*(embedding_dim//head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # 无需mask
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 根号dk分之一，对应Scaled操作
        context = self.attention(Q, K, V, scale)  # Scaled_Dot_Product_Attention计算
        context = context.view(batch_size, -1, self.dim_head * self.num_head)  # reshape 回原来的形状
        out = self.fc(context)  # 全连接
        out = self.dropout(out)
        out = out + x  # 残差连接,ADD
        out = self.layer_norm(out)  # 对应Norm
        return out


class Multi_Head_Attention(nn.Module):
    '''
    params: dim_model-->hidden dim *num_head
    '''

    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0  # head数必须能够整除隐层大小
        self.dim_head = dim_model // self.num_head  # 按照head数量进行张量均分
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)  # Q，通过Linear实现张量之间的乘法，等同手动定义参数W与之相乘
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)  # 自带的LayerNorm方法

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1,
                   self.dim_head)  # reshape to batch*head*sequence_length*(embedding_dim//head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 根号dk分之一，对应Scaled操作
        context = self.attention(Q, K, V, scale)  # Scaled_Dot_Product_Attention计算
        context = context.view(batch_size, -1, self.dim_head * self.num_head)  # reshape 回原来的形状
        out = self.fc(context)  # 全连接
        out = self.dropout(out)
        out = out + x  # 残差连接,ADD
        out = self.layer_norm(out)  # 对应Norm
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product'''

    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # Q*K^T
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)  # 两层全连接
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])  # 多次Encoder

        self.glob = nn.Linear(config.pad_size * config.dim_model, config.num_classes_g)
    def forward(self, x):
        out = self.postion_embedding(x)  # batch,seqlen
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)  # 将三维张量reshape成二维，然后直接通过全连接层将高维数据映射为classes
        # out = torch.mean(out, 1)    # 也可用池化来做，但是效果并不是很好
        glob = self.glob(out)
        return glob


class ConfigTrans(object):
    """配置参数"""

    def __init__(self):
        self.model_name = 'Transformer'
        self.dropout = 0.5
        self.num_classes_g = 1  # 类别数
        self.num_epochs = 30  # epoch数
        self.batch_size = 32  # mini-batch大小
        self.pad_size = 7  # 每句话处理成的长度(短填长切)，这个根据自己的数据集而定
        self.learning_rate = 1e-5  # 学习率
        self.embed = 68  # 字向量维度
        self.dim_model = 68  # 需要与embed一样
        self.hidden = 64
        self.last_hidden = 64
        self.num_head = 4  # 多头注意力，需要能够被embed整除！，不然会出现问题！
        self.num_encoder = 2  # 使用两个Encoder，尝试6个encoder发现存在过拟合，毕竟数据集量比较少（10000左右），可能性能还是比不过LSTM

config = ConfigTrans()



# 模型训练
def train(model, train_loader, epoch):
    model.train()
    train_loss = 0
    for i, data in enumerate(train_loader, 0):  # 遍历一个epoch
        x, y = data
        x = x.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss
    loss_mean = train_loss / (i + 1)
    train_losses.append(loss_mean)
    if epoch==20:
        print('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, loss_mean.item()))
    writer.add_scalar('train loss', loss_mean,epoch)


# 模型测试
def test(model, test_loader,epoch):
    model.eval()
    test_loss = 0
    mae_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            x, y = data
            x = x.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            out = model(x)
            test_loss += criterion(out, y).item()
            mae_loss += criterion2(out,y).item()
        test_loss /= (i + 1)*BATCHSIZE
        mae_loss /= (i+1)*BATCHSIZE
        eval_losses.append(test_loss)
        mae_losses.append(mae_loss)
        if epoch==20:
            print('Test set: Average loss: {:.4f}\n'.format(test_loss))
            print('Test set: MAE loss: {:.4f}\n'.format(mae_loss))
        writer.add_scalar('test loss', test_loss, epoch)
        writer.add_scalar('mae_loss',mae_loss,epoch)


def main():
    start_epoch = 0
    for epoch in range(start_epoch + 1, EPOCHS):
        train(model, train_load, epoch)
        test(model, test_load,epoch)
        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch
                 }
        torch.save(state,log_dir) # 一个epoch存储一次
    print('Finish Training')

# 目标划分
def trainlabel(x,label,test_size):
    X_train, X_test, y_train, y_test = train_test_split(x, label, test_size=test_size, random_state=0)
    return X_train,X_test,y_train,y_test

if __name__ =='__main__':
    # 读取数据
    xs, label_stabs, label_repus, label_acts, label_commus, label_resos, label_abilitys = dataprepare()
    labels = [ label_commus,label_abilitys, label_resos,label_acts, label_repus ,label_stabs]
    for i in range(len(labels)):
        print('第{}个标签\n'.format(i + 1))
        X_train, X_test, y_train, y_test = trainlabel(xs, labels[i], test_size=0.1)
        train_data = TensorDataset(torch.from_numpy(X_train.astype(np.float32)),
                                   torch.from_numpy(y_train.astype(np.float32)))
        test_data = TensorDataset(torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test.astype(np.float32)))
        # 切分数据
        BATCHSIZE = 32
        WORKERS = 2
        train_load = torch.utils.data.DataLoader(train_data, batch_size=BATCHSIZE, shuffle=True, num_workers=WORKERS,drop_last=True)
        test_load = torch.utils.data.DataLoader(test_data, batch_size=BATCHSIZE, shuffle=False, num_workers=WORKERS,drop_last=True)
        # 定义模型
        # embed_size, kernel_sizes, nums_channels = 68, [3, 4, 5], [128, 128, 128]
        model = Transformer().cuda()
        EPOCHS = 21
        train_losses = []  # 记 录每epoch训练的平均loss
        eval_losses = []  # 测试的
        mae_losses = []
        test_flag = False  # 为True时候加载保存好的model进行测试
        log_dir = '../log/checkpoint3'
        # 定义loss和optimizer
        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        # 运行model
        main()
        writer.close()