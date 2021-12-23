import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from data.Data_copy import getdatabench
from data.SQL import get_date_lista
from random import shuffle
from torch.utils.tensorboard import SummaryWriter

# 全局变量定义
writer = SummaryWriter('../log/runslstm/healthy_analysis')
EPOCHS = 30
train_losses = []  # 记 录每epoch训练的平均loss
eval_losses = []  # 测试的
mae_losses = []
test_flag = False  # 为True时候加载保存好的model进行测试
log_dir = '../log/checkpoint2'

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
        x_catgs = torch.stack([i for i in x_catgs],dim = 0) # 七天直接concat[7,1149,36]
        x_conts = torch.stack([i for i in x_conts],dim = 0)

        x = torch.cat([x_catgs,x_conts],dim = 2).cpu() # 总特征[7,数据量,总特征维度数]
        xs = torch.cat([xs,x],dim = 1) # 合并为总的
        label_stabs = torch.cat([label_stabs,label_stab.cpu()],dim = 0)
        label_repus = torch.cat([label_repus,label_repu.cpu()],dim = 0)
        label_acts = torch.cat([label_acts,label_act.cpu()],dim = 0)
        label_commus = torch.cat([label_commus,label_commu.cpu()],dim = 0)
        label_resos = torch.cat([label_resos,label_reso.cpu()],dim = 0)
        label_abilitys = torch.cat([label_abilitys,label_ability.cpu()],dim = 0)
    # 转换为numpy
    xs = xs.permute(1,0,2).numpy()
    label_stabs = label_stabs.view(-1,).numpy()
    label_repus = label_repus.view(-1,).numpy()
    label_acts = label_acts.view(-1,).numpy()
    label_commus = label_commus.view(-1,).numpy()
    label_resos =label_resos.view(-1,).numpy()
    label_abilitys = label_abilitys.view(-1,).numpy()
    return xs,label_stabs,label_repus,label_acts,label_commus,label_resos,label_abilitys



class LSTMTag(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTMTag, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim  # 隐藏层大小
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer,
                            batch_first=True,dropout=0.5) #
        self.dense = nn.Linear(hidden_dim,hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_class)

    #         self.hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size):
        # 开始时刻, 没有隐状态
        # 关于维度设置的详情,请参考 Pytorch 文档
        # 各个维度的含义是 (layer_num, minibatch_size, hidden_dim)
        return (torch.zeros(2, batch_size, self.hidden_dim).cuda(),
                torch.zeros(2, batch_size, self.hidden_dim).cuda())

    def forward(self, x, batch_size):
        # hidden_weight = self.init_hidden(batch_size)
        # out, _ = self.lstm(x, hidden_weight)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 只需要最后一个隐藏层状态
        out = self.classifier(self.dense(out))
        return out

def get_net(in_dim, hidden_dim, n_layer, n_class):
    net = LSTMTag(in_dim, hidden_dim, n_layer, n_class) # 修改变种
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net


# 模型训练
def train(model, train_loader, epoch):
    model.train()
    train_loss = 0
    for i, data in enumerate(train_loader, 0):  # 遍历一个epoch
        x, y = data
        x = x.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        out = model(x, len(x))
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss
    loss_mean = train_loss / (i + 1) # 这里怎么能除以trainload的大小呢！！！！这个i不是数量！！靠
    train_losses.append(loss_mean)
    writer.add_scalar('training loss', loss_mean, epoch)
    if i==29:
        print('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, loss_mean.item()))


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
            out = model(x, len(x))  # 这里存疑
            test_loss += criterion(out, y).item()
            mae_loss += criterion2(out,y).item()
        test_loss /= (i + 1)*BATCHSIZE # 这里不该除以i+1，该除以test size的数量
        mae_loss /= (i+1)*BATCHSIZE
        eval_losses.append(test_loss)
        mae_losses.append(mae_loss)
        if epoch==29:
            print('Test set: Average loss: {:.4f}\n '.format(test_loss))
            print('Test set: MAE loss: {:.4f}\n '.format(mae_loss))
        writer.add_scalar('test loss', test_loss, epoch)
        writer.add_scalar('mae loss', mae_loss, epoch)


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


if __name__ == '__main__':

    # 获取数据
    xs, label_stabs, label_repus, label_acts, label_commus, label_resos, label_abilitys = dataprepare()
    labels = [label_stabs, label_repus,label_acts, label_commus, label_resos, label_abilitys]
    for i in range(len(labels)):
        print('第{}个标签\n'.format(i + 1))
        X_train, X_test, y_train, y_test = trainlabel(xs, labels[i], test_size=0.1)
        # 分类任务记得long,float64!=float32
        train_data = TensorDataset(torch.from_numpy(X_train.astype(np.float32)),
                                   torch.from_numpy(y_train.astype(np.float32)))
        test_data = TensorDataset(torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test.astype(np.float32)))
        BATCHSIZE = 32
        WORKERS = 2
        # 数据切分
        train_load = torch.utils.data.DataLoader(train_data, batch_size=BATCHSIZE, shuffle=True, num_workers=WORKERS)
        test_load = torch.utils.data.DataLoader(test_data, batch_size=BATCHSIZE, shuffle=False, num_workers=WORKERS)
        # 定义model
        model = get_net(xs.shape[2], 64, 2, 1)  # 单词embedding维度，hidden 维度，两层隐藏层叠加,回归任务改为1
        use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
        if use_gpu:
            model = model.cuda()
        # 定义loss和optimizer
        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        main()
        # 关闭tensorboard
        writer.close()