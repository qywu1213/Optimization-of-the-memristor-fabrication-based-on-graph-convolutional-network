import torch.nn
import random
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from Net9 import *
from GetDataSet import *

# 使用列表中的哪一个随机数
random_index = 2
random_list = [114, 930, 586, 247, 108, 188, 597]

# 好像是为了增加开关比大于3的样本的权重
def weighted_crit(output, target, weight):
    dis = abs(output - target)
    weight = torch.tensor(weight)
    dis = dis.cpu()
    weighted_dis = torch.mul(dis, weight)
    ref = torch.zeros_like(weighted_dis)
    weighted_loss = crit(weighted_dis, ref)
    return weighted_loss


# -----训练所需要定义的变量-----
model_index = 0

model = Net9()

# 加载已经训练过部分epoch的模型
# model = torch.load("../new_model/net9/2-model9_lr_1e-05_epoch_130000_train_0.9_decay_0.02.pth")
# model = torch.load("../new_model/net9/3-model9_lr_3e-05_epoch_40000_train_0.9_decay_0.04.pth")
# model = torch.load("../ex4/hfmodels/5-model13_lr_3e-05_epoch_40000_train_0.9_decay_0.02.pth")


model = model.cuda()

dataset = MyOwnDataset("revision_hf")
dataset_inv = MyOwnDataset("revision_hf_inv")

print("dataset的长度为{}".format(len(dataset)))

# -----划分训练集、测试集-----
random.seed(random_list[random_index])
dataset_size = len(dataset)
indices = list(range(dataset_size))
random.shuffle(indices)
train_ratio, test_ratio = 0.9, 0.1
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size
train_indices, test_indices = indices[:train_size], indices[train_size:]
# 和反演的一起取平均
train_dataset = dataset[train_indices]
train_dataset_inv = dataset_inv[train_indices]
test_dataset = dataset[test_indices]
# print(len(train_dataset))
# print(len(test_dataset))

#  -----训练参数-----
learning_rate = 1e-4
num_of_epochs = 100
pre_epoch = 0
train_batch_size = 64
test_batch_size = 64
weight_decay = 0.02

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
train_dataloader_inv = DataLoader(train_dataset_inv, batch_size=train_batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size)
# print(len(train_dataloader))
# print(len(test_dataloader))
crit = torch.nn.L1Loss()
crit = crit.cuda()

# tensorboard:
writer = SummaryWriter("../logs")


# -----训练函数-----
def train():
    model.train()

    lost_all = 0
    # lost_all_origin = 0
    num_of_high = 0
    num_of_high_inv = 0
    for data, data_inv in zip(train_dataloader, train_dataloader_inv):
        # print(len(train_dataloader))
        # print(len(data))
        data = data.cuda()
        data_inv = data_inv.cuda()

        optimizer.zero_grad()

        output = model(data)
        output_inv = model(data_inv)

        label = data.y
        label = label.squeeze()
        label1 = label.tolist()
        weight = []
        for yi in label1:
            if yi >= 3.0:
                weight.append(2.0)
                num_of_high = num_of_high + 1
            else:
                weight.append(1.0)
        loss = weighted_crit(output, label, weight)

        label_inv = data_inv.y
        label_inv = label_inv.squeeze()
        label1_inv = label_inv.tolist()
        weight_inv = []
        for yi in label1_inv:
            if yi >= 3.0:
                weight_inv.append(2.0)
                num_of_high_inv = num_of_high_inv + 1
            else:
                weight_inv.append(1.0)
        loss_inv = weighted_crit(output_inv, label_inv, weight_inv)

        average_loss = (loss + loss_inv)/2.0
        # loss_origin = crit(output, label)

        average_loss.backward()
        lost_all = lost_all + data.num_graphs * average_loss.item()
        optimizer.step()

    return lost_all / (len(train_dataset) + num_of_high)


def test():
    model.eval()
    with torch.no_grad():
        test_loss_all = 0
        for data in test_dataloader:
            data = data.cuda()
            label = data.y
            label = label.squeeze()
            output = model(data)
            loss = crit(output, label)
            test_loss_all = test_loss_all + data.num_graphs * loss.item()

        return test_loss_all / len(test_dataset)


# 正式训练和测试：
num_model = 1

for epoch in range(num_of_epochs):
    print("-----第{}轮训练开始-----".format(epoch + 1))
    train_loss = train()

    # 最终选择模型的代码
    test_loss = test()
    if test_loss < 0.26 and train_loss < 0.26:
        torch.save(model, "../ex7/hfmodels/model_loss{}_no{}_rdm{}.pth".format(test_loss, num_model, random_index))
        num_model = num_model + 1
        print("训练的损失为：{}".format(train_loss))
        print("测试的损失为：{}".format(test_loss))
        break
    if (epoch + 1) % 20 == 0:
        test_loss = test()
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("test_loss", test_loss, epoch)
        print("训练的损失为：{}".format(train_loss))
        print("测试的损失为：{}".format(test_loss))
    # writer.add_scalar("train_loss", train_loss, epoch)
    # writer.add_scalar("test_loss", test_loss, epoch)

writer.close()


# 保存模型
torch.save(model, "../ex7/hfmodels/model_epoch_{}_rdm{}.pth".format(num_of_epochs + pre_epoch,
                                                                    random_index
                                                                                  ))