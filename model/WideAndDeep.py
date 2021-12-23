import torch
import torch.nn as nn

class WideAndDeep(nn.Module):
    '''
    func:一天所有帮会的profile embedding
    '''

    def __init__(self, wide_dim, dense_dim, embed_dim):
        super(WideAndDeep, self).__init__()
        # define some layers
        self.wide = nn.Sequential(
            nn.Linear(wide_dim, 128)  #
        )
        self.dense = nn.Sequential(
            nn.Linear(dense_dim, 128),  # 这里要加上embedding的量，这里忽略
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.lin1 = nn.Linear(160, 64)  # 32+128,和输入的维度没有关系,改成和关系一样的大小

    def forward(self, x_catg, x_cont):
        x_dense = x_cont  # dense层将交叉特征的embedding和连续数据进行concat，这里忽略
        x_wide = x_catg  # wide层放入交叉特征数据
        out1 = self.wide(x_wide)
        out2 = self.dense(x_dense)  # 需要转为tensor和float32
        out = torch.cat((out1, out2), dim=1)  # 128+32
        out = self.lin1(out)
        return out
