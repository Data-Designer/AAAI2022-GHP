# 1. 导入相关的包
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import weight_norm
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





class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[2])


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs=68, num_channels=[16, 16], kernel_size=5, dropout=0.5):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.classifier_g = nn.Linear(112,1) # 这个地方是2个


    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        x = x.permute(0, 2, 1)
        out = self.network(x).view(32, -1)
        glob = self.classifier_g(out)
        return glob



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
    if epoch==39:
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
        if epoch==39:
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
        model = TemporalConvNet().cuda()
        EPOCHS = 40
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