import torch
from torch_geometric.nn import global_mean_pool, GCNConv, global_max_pool
import torch.nn.functional as F


embed_dim = 32


class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()

        self.conv1 = GCNConv(embed_dim, 128)
        self.conv2 = GCNConv(128, 128)

        self.lin_edge = torch.nn.Linear(1, 128)

        self.lin1 = torch.nn.Linear(128*2, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)

    def forward(self, data):
        # x：节点特征矩阵，edge_index：边索引矩阵，edge_attr：边特征矩阵，batch：batch信息
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 第一层GCNConv
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        # 边特征的线性变换
        # edge_attr = F.relu(self.lin_edge(edge_attr))

        # 第二层GCNConv
        # print(x.shape, edge_index.shape, edge_attr.shape)
        # x = self.conv2(x, edge_index, edge_attr)
        # print(x.shape, edge_index.shape, edge_attr.shape)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)

        # 池化层
        x1 = global_mean_pool(x, batch=batch)
        x2 = global_max_pool(x, batch=batch)
        x = torch.cat([x1, x2], dim=1)

        # 线性层
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)

        return x.squeeze()
