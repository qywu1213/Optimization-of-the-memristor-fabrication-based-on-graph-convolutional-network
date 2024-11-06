import torch
import torch.nn as nn
import torch.nn.functional as F

# num_heads = 4
# in_features = 38
# head_dim = 64
# out_features = num_heads * head_dim


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, edge_dim, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.edge_dim = edge_dim

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features + edge_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(self, input_h, edge_index, edge_features):

        # e 边的特征
        h = torch.mm(input_h, self.W)
        N = h.size(0)  # N代表图的节点数

        # Concatenate the node features for self-attention calculation
        # print(h.shape)
        # print(edge_features.shape)

        # 把adj扩展成邻接矩阵的形式
        adj = torch.zeros(N, N)
        edge_matrix = torch.zeros(N * N, self.edge_dim)
        for index in range(0, edge_index.shape[1]):
            i = edge_index[0][index]
            j = edge_index[1][index]
            # print(i, j)
            adj[i][j] = 1
            edge_matrix[i * N + j] = edge_features[index]
        # print(adj.shape)
        # print(edge_matrix.shape)
        # print(adj)
        # print(edge_matrix)
        # print(edge_matrix)
        input_concat = torch.cat([h.repeat(1, N).view(N * N, -1),
                                  h.repeat(N, 1),
                                  edge_matrix.repeat(1, 1)], dim=1).view(N, -1, 2 * self.out_features + self.edge_dim)

        # 注意力系数
        e = self.leaky_relu(torch.matmul(input_concat, self.a).squeeze(2))  # Attention coefficients

        zero_vec = -1e12 * torch.ones_like(e)  # 这个向量用来将没有连接的边设置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # Masked softmax. 当边存在时设置为注意力系数，否则使之为负无穷。
        attention = torch.softmax(attention, dim=1)  # 得到形状仍然为[N,N]的且归一化的注意力权重

        # Apply dropout to attention coefficients
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        output_h = torch.matmul(attention, h)
        return output_h


class GATRegression(nn.Module):
    def __init__(self, in_feature_size, output_size, num_heads, edge_dim, dropout, alpha):
        super(GATRegression, self).__init__()
        self.dropout = dropout
        # 创建num_heads个注意力头并且把这些注意力头聚合在这个列表里。
        self.attentions = [GATLayer(in_feature_size, output_size, edge_dim,  dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(num_heads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(output_size * num_heads, 32, edge_dim, dropout=dropout, alpha=alpha, concat=False)
        self.fc = torch.nn.Linear(32, 1)

    def forward(self, data):
        node_features = data.x
        adj = data.edge_index
        edge_features = data.edge_attr

        x = F.dropout(node_features, self.dropout, training=self.training)
        x = torch.cat([att(node_features, adj, edge_features) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj, edge_features)).mean(dim=0)
        # print("输出的形状为：{}".format(x.shape))
        x = self.fc(x)
        # print("最终输出的形状为：{}".format(x.shape))
        return x
