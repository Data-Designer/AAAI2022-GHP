import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv

# 构造GCN进行聚合,一定记住不要inplace，不要单独构造！！！
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)
        self.conv3 = GraphConv(hidden_size,hidden_size)
        self.conv4 = GraphConv(hidden_size,hidden_size)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.nn.ReLU()(h)
        h = self.conv3(g, h)
        h = torch.nn.ReLU()(h)
        h = self.conv4(g,h)
        h = torch.nn.ReLU()(h)
        h = self.conv2(g, h)
#         print(h._version)
        return h

