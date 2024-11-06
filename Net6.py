import torch
from torch.nn import BatchNorm1d
from torch_geometric.nn import global_mean_pool, GCNConv, global_max_pool, GraphNorm
import torch.nn.functional as F


#  使用了prelu和bn层


embed_dim = 32
channels = 64
inter_channels = 32

class Net6(torch.nn.Module):
    def __init__(self):
        super(Net6, self).__init__()

        # prelu的科学系参量：
        self.weight1 = torch.nn.Parameter(torch.Tensor(channels).fill_(0.25))
        self.weight2 = torch.nn.Parameter(torch.Tensor(inter_channels).fill_(0.25))

        self.conv1 = GCNConv(embed_dim, channels)
        self.conv2 = GCNConv(channels, channels)

        # self.lin_edge = torch.nn.Linear(1, 128)

        self.lin1 = torch.nn.Linear(channels*2, inter_channels)
        # self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(inter_channels, 1)

        # bn层
        self.bn1 = GraphNorm(channels)
        self.bn2 = GraphNorm(channels)

    def forward(self, data):
        # x：节点特征矩阵，edge_index：边索引矩阵，edge_attr：边特征矩阵，batch：batch信息
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 第一层GCNConv
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.prelu(x, self.weight1)
        x = F.dropout(x, p=0.5, training=self.training)

        # 边特征的线性变换
        # edge_attr = F.relu(self.lin_edge(edge_attr))

        # 第二层GCNConv
        x = self.conv2(x, edge_index, edge_attr)
        x = F.prelu(x, self.weight1)
        x = self.bn2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # 池化层
        x1 = global_mean_pool(x, batch=batch)
        x2 = global_max_pool(x, batch=batch)
        x = torch.cat([x1, x2], dim=1)

        # 线性层
        x = self.lin1(x)
        # x = F.prelu(x, self.weight1)
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin2(x)
        x = F.prelu(x, self.weight2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)

        return x.squeeze()
