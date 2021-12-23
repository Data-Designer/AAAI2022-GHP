# 1. 导入相关的包
import pandas as pd
import numpy as np
import random
import os
from collections import defaultdict
from sklearn.preprocessing import StandardScaler,OneHotEncoder,PolynomialFeatures
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,f1_score
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import  torch.nn.functional as F


# 2.数据预处理
def preprocessingpro(data_guild_profile):
    '''
    func: 预处理profile
    '''
    # 补0
    data_guild_profile = data_guild_profile.fillna(0)
    # 类别变量处理
    tmp_columns = []
    for i in CATG_COLUMNS:
        tmp = pd.get_dummies(data_guild_profile[i].astype(int), prefix=i)  # low maintain本身就是0，1，不需要
        tmp_columns.extend(list(tmp.columns))
        data_guild_profile = pd.concat([data_guild_profile, tmp], axis=1)
        data_guild_profile.pop(i)
        CATG_COLUMNS.remove(i)

    CATG_COLUMNS.extend(i for i in tmp_columns)

    # 建筑比较特殊,连续变量
    tmp_build_info = buildprocess()
    data_guild_profile = pd.concat([data_guild_profile, tmp_build_info], axis=1)
    data_guild_profile.pop("build_info")
    CONT_COLUMNS.extend(list(tmp_build_info.columns))
    CONT_COLUMNS.remove('build_info')

    # 返回归一化后的连续变量和离散变量
    scaler = StandardScaler()
    x = data_guild_profile
    x_catg = np.array(data_guild_profile[CATG_COLUMNS], dtype='float32')  # 与cont一致即可
    x_cont = np.array(data_guild_profile[CONT_COLUMNS], dtype='float32')
    x_cont = scaler.fit_transform(data_guild_profile[CONT_COLUMNS])

    return [x, x_catg, x_cont]


def buildprocess():
    '''
    func: 建筑分割
    '''
    temp = defaultdict(list)
    for row in range(len(data_guild_profile)):
        for index, value in enumerate(data_guild_profile['build_info'][row].split('-')):
            _, value = value.split(':')
            temp[index].append(value)
    build_info = pd.DataFrame(temp)
    build_info.columns = ['build_1', 'build_2', 'build_3', 'build_4', 'build_5',
                          'build_6', 'build_7', 'build_8', 'build_9', 'build_10', 'build_11', 'build_12', 'build_13']
    build_info = build_info.apply(lambda x: x.astype(int))  # 原来为字符串
    return build_info


def guildprofile(data_guild_profile):
    '''
    func:返回预处理完的帮会profile
    '''
    x, x_catg, x_cont = preprocessingpro(data_guild_profile)  # 预处理帮会profile
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)  # 不要那个为全1列
    x_catg_poly = poly.fit_transform(x_catg)  # 分类特征交叉 cross项
    x_catg = x_catg_poly  # 这里不需要转换
    return x_catg, x_cont


# 3. 读取数据，按理说是7个
COLUMNS = ['cai_qi_value', 'today_online_cnt', 'total_count', 'scale', 'build_info', 'xuetu_num',
           'member_num', 'zi_cai', 'rq_value', 'fund', 'prosperity', 'guild_member_cnt', 'max_guild_member_cnt',
           'max_xue_tu_member', 'full_degree', 'g_force', 'liansairank', 'joinmem', 'leavemem', 'chatnum', 'chuangongnum'
           ,'guild_type', 'low_maintain_state']

CONT_COLUMNS = ['cai_qi_value', 'today_online_cnt', 'total_count', 'scale', 'build_info', 'xuetu_num', 'member_num', 'zi_cai',
                'rq_value', 'fund', 'prosperity', 'guild_member_cnt', 'max_guild_member_cnt', 'max_xue_tu_member', 'full_degree',
                 'liansairank', 'joinmem', 'leavemem', 'chatnum', 'chuangongnum']

CATG_COLUMNS = ['guild_type', 'low_maintain_state','g_force']

data_guild = pd.read_csv('../data/guild.csv')
data_guild_profile = data_guild[COLUMNS] # profile
x_catg,x_cont= guildprofile(data_guild_profile)




# LSTM重新读取数据
# feature
x_all = np.concatenate((x_catg,x_cont),axis = 1) # 所有用于训练的x特征
# label这里针对每个目标都要做一次！！！
y_all_1 = np.random.randint(5,size = (1205,)) # 注意是一维而不是二维


X_train,X_test,y_train,y_test =  train_test_split(x_all,y_all_1,test_size =0.3,random_state = 0)

# y_train = y_train[:,np.newaxis]
# y_test = y_test[:,np.newaxis] # 增加新维度

y_train = y_train[:]
y_test = y_test[:]


X_train = np.stack((X_train,X_train,X_train,X_train,X_train,X_train,X_train),axis = 1) # LSTM是处理序列！ 7=sequence_len,68=embedding size
X_test = np.stack((X_test,X_test,X_test,X_test,X_test,X_test,X_test),axis = 1)
train_data = TensorDataset(torch.from_numpy(X_train.astype(np.float32)),torch.from_numpy(y_train.astype(np.float32)).long())
test_data = TensorDataset(torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test.astype(np.float32)).long())
BATCHSIZE = 32
WORKERS = 2
train_load = torch.utils.data.DataLoader(train_data, batch_size=BATCHSIZE, shuffle=True, num_workers=WORKERS)
test_load = torch.utils.data.DataLoader(test_data, batch_size=BATCHSIZE, shuffle=False, num_workers=WORKERS)


class LSTMTag(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTMTag, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim  # 隐藏层大小
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer,
                            batch_first=True,dropout=0.5) #
        self.classifier = nn.Linear(hidden_dim, n_class)

    #         self.hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size):
        # 开始时刻, 没有隐状态
        # 关于维度设置的详情,请参考 Pytorch 文档
        # 各个维度的含义是 (layer_num, minibatch_size, hidden_dim)
        return (torch.zeros(2, batch_size, self.hidden_dim).cuda(),
                torch.zeros(2, batch_size, self.hidden_dim).cuda())

    def forward(self, x, batch_size):
        hidden_weight = self.init_hidden(batch_size)
        out, _ = self.lstm(x, hidden_weight)
        out = out[:, -1, :]  # 只需要最后一个隐藏层状态
        out = self.classifier(out)
        return out



model = LSTMTag(x_all.shape[1], 64, 2, 5)  # 单词embedding维度，hidden 维度，两层隐藏层叠加，输出分类数是5
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()
# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.2)

EPOCHS = 10
train_losses = []  # 记 录每epoch训练的平均loss
eval_losses = []  # 测试的
test_flag = False  # 为True时候加载保存好的model进行测试
log_dir = './healthy_model_lstm.pth'


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
    loss_mean = train_loss / (i + 1)
    train_losses.append(loss_mean)
    print('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, loss_mean.item()))


# 模型测试

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            x, y = data
            x = x.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            out = model(x, len(x))  # 这里存疑
            test_loss += criterion(out, y).item()
            pred = out.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()  # 相等的才累加，最后是所有分类正确的数目
        test_loss /= (i + 1)
        eval_losses.append(test_loss)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_data), 100. * correct / len(test_data)))


def main():
    if test_flag:
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        test(model, test_load)
        return

    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('加载epoch {} 成功'.format(start_epoch))
    else:
        start_epoch = 0
        print('无保存model，将从头开始训练！')

    for epoch in range(start_epoch + 1, EPOCHS):
        train(model, train_load, epoch)
        test(model, test_load)
        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch
                 }
    #         torch.save(state,log_dir) # 一个epoch存储一次

    print('Finish Training')


main()