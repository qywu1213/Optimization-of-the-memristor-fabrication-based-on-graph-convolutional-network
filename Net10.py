import torch
from torch.nn import BatchNorm1d
from torch_geometric.nn import global_mean_pool, GCNConv, global_max_pool, GraphNorm
import torch.nn.functional as F


#  Net4的基础上调整了通道数，同时第一层卷积就使用了边的信息
# 修正了Net9中第二层卷积层中Prelu激活和BN层顺序反了的问题。


# embed_dim = 38
embed_dim = 42
# p_list_1 = [0.5, 0.5, 0.5, 0.5]
# p_list_1 = [0.8, 0.5, 0.5, 0.5]
p_list_1 = [0.5, 0.5, 0.5, 0.5]


class Net10(torch.nn.Module):
    def __init__(self):
        super(Net10, self).__init__()

        # prelu的可学习参量：
        self.weight1 = torch.nn.Parameter(torch.Tensor(256).fill_(0.25))
        self.weight0 = torch.nn.Parameter(torch.Tensor(128).fill_(0.25))
        self.weight2 = torch.nn.Parameter(torch.Tensor(64).fill_(0.25))

        self.conv1 = GCNConv(embed_dim, 256)
        self.conv2 = GCNConv(256, 128)

        self.lin1 = torch.nn.Linear(128*2, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)

        # bn层
        self.bn1 = GraphNorm(256)
        self.bn2 = GraphNorm(128)

    def forward(self, data):
        # x：节点特征矩阵，edge_index：边索引矩阵，edge_attr：边特征矩阵，batch：batch信息
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 第一层GCNConv
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.prelu(x, self.weight1)
        x = F.dropout(x, p=p_list_1[0], training=self.training)

        # 边特征的线性变换
        # edge_attr = F.relu(self.lin_edge(edge_attr))

        # 第二层GCNConv
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.prelu(x, self.weight0)
        x = F.dropout(x, p=p_list_1[1], training=self.training)

        # 池化层
        x1 = global_mean_pool(x, batch=batch)
        x2 = global_max_pool(x, batch=batch)
        x = torch.cat([x1, x2], dim=1)

        # 线性层
        x = self.lin1(x)
        x = F.prelu(x, self.weight0)
        x = F.dropout(x, p=p_list_1[2], training=self.training)
        x = self.lin2(x)
        x = F.prelu(x, self.weight2)
        x = F.dropout(x, p=p_list_1[3], training=self.training)
        x = self.lin3(x)

        return x.squeeze()
