# 1. 导入相关的包
import numpy as np
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





class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[2])


class CNNTag(nn.Module):
    def __init__(self, embed_size, kernel_sizes, num_channels):
        super(CNNTag, self).__init__()
        # 不参与训练的嵌入层
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(sum(num_channels), 64)
        self.dense = nn.Linear(64,32)
        self.classifiers = nn.Linear(32, 1)
        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels = embed_size,
                                        out_channels = c,
                                        kernel_size = k)) # 卷积就是不断聚合的过程


    def forward(self, inputs):
        # 将两个形状是(批量大小, 词数, 词向量维度)的嵌入层的输出按词向量连结
        embeddings = inputs # (batch, seq_len, embed_size)
        # 根据Conv1D要求的输入格式，将词向量维，即一维卷积层的通道维(即词向量那一维)，变换到前一维
        embeddings = embeddings.permute(0, 2, 1)
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的
        # Tensor。使用flatten函数去掉最后一维，然后在通道维上连结
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)
        # 应用丢弃法后使用全连接层得到输出
        outputs = self.dense(self.dropout(self.classifier(self.dropout(encoding))))
        outputs = self.classifiers(outputs)
        return outputs



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
    if epoch==29:
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
        if epoch==29:
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
    labels = [label_commus, label_abilitys, label_resos,label_acts, label_repus ,label_stabs]
    for i in range(len(labels)):
        print('第{}个标签\n'.format(i + 1))
        X_train, X_test, y_train, y_test = trainlabel(xs, labels[i], test_size=0.1)
        train_data = TensorDataset(torch.from_numpy(X_train.astype(np.float32)),
                                   torch.from_numpy(y_train.astype(np.float32)))
        test_data = TensorDataset(torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test.astype(np.float32)))
        # 切分数据
        BATCHSIZE = 32
        WORKERS = 2
        train_load = torch.utils.data.DataLoader(train_data, batch_size=BATCHSIZE, shuffle=True, num_workers=WORKERS)
        test_load = torch.utils.data.DataLoader(test_data, batch_size=BATCHSIZE, shuffle=False, num_workers=WORKERS)
        # 定义模型
        embed_size, kernel_sizes, nums_channels = 68, [3, 4, 5], [128, 128, 128]
        model = CNNTag(embed_size, kernel_sizes, nums_channels).cuda()
        EPOCHS = 30
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