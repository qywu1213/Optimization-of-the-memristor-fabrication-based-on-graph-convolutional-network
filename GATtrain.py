import random

import torch.nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from GAT1 import *  # model10
from GetDataSet import *


def weighted_crit(output, target, weight):
    dis = abs(output - target)
    weight = torch.tensor(weight)
    dis = dis.cpu()
    weighted_dis = torch.mul(dis, weight)
    ref = torch.zeros_like(weighted_dis)
    weighted_loss = crit(weighted_dis, ref)
    return weighted_loss


# -----训练所需要定义的变量-----
model_index = 2
in_feature_size = 38
out_feature_size = 128
num_heads = 4
edge_dim = 1
drop = 0.5
alpha = 1.0

model = GATRegression(in_feature_size, out_feature_size, num_heads, edge_dim, drop, alpha)
# model = model.cuda()
dataset = MyOwnDataset("DataSet_OneHot_inverse_expanded_weighted_reduced_edge")

# -----划分训练集、测试集-----
random.seed(191)
dataset_size = len(dataset)
indices = list(range(dataset_size))
random.shuffle(indices)
train_ratio, test_ratio = 0.9, 0.1
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size
train_indices, test_indices = indices[:train_size], indices[train_size:]
train_dataset = dataset[train_indices]
test_dataset = dataset[test_indices]


#  -----训练参数-----
learning_rate = 3e-4
num_of_epochs = 1000
train_batch_size = 1
test_batch_size = 1
weight_decay = 0.02

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size)
crit = torch.nn.L1Loss()
# crit = crit.cuda()

# tensorboard:
writer = SummaryWriter("../logs")


# -----训练函数-----
def train():
    model.train()

    lost_all = 0
    # lost_all_origin = 0
    num_of_high = 0
    for data in train_dataloader:
        # print(len(train_dataloader))
        # print(len(data))
        # data = data.cuda()
        optimizer.zero_grad()
        output = model(data)
        label = data.y
        label = label.squeeze()
        label1 = label.tolist()
        weight = []
        if label1 >= 3.0:
            weight.append(2.0)
            num_of_high = num_of_high + 1
        else:
            weight.append(1.0)
        # weight = [2.0 if yi >= 3.0 else 1.0 for yi in label1]
        loss = weighted_crit(output, label, weight)
        # loss_origin = crit(output, label)
        loss.backward()
        lost_all = lost_all + data.num_graphs * loss.item()
        # lost_all_origin = lost_all_origin + data.num_graphs * loss_origin.item()
        optimizer.step()
        # print("one batch")

    return lost_all / (len(train_dataset) + num_of_high)


def test():
    model.eval()
    with torch.no_grad():
        test_loss_all = 0
        for data in test_dataloader:
            # data = data.cuda()
            label = data.y
            label = label.squeeze()
            output = model(data)
            loss = crit(output, label)
            test_loss_all = test_loss_all + data.num_graphs * loss.item()

        return test_loss_all / len(test_dataset)


# 正式训练和测试：

for epoch in range(num_of_epochs):
    print("-----第{}轮训练开始-----".format(epoch + 1))
    train_loss = train()
    if (epoch + 1) % 20 == 0:
        test_loss = test()
        print("测试的损失为：{}".format(test_loss))
        writer.add_scalar("test_loss", test_loss, epoch)
        writer.add_scalar("train_loss", train_loss, epoch)
    print("训练的损失为：{}".format(train_loss))

writer.close()

torch.save(model, "../GATmodels/GAT{}_lr_{}_epoch_{}_batch_{}_decay_{}".format(model_index, learning_rate,
                                                                               num_of_epochs, train_batch_size,
                                                                               weight_decay))
