#  该文件用于训练正常的没有反演的版本。

import torch.nn
import random
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from Net9 import *
from GetDataSet import *

# 使用列表中的哪一个随机数
random_index = 8
random_list = [114, 930, 586, 247, 108, 188, 597, 299, 905]

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
model = Net9()

model = torch.load("../ex7/hfmodels/2model_loss_0.302_0.345_epoch35000_rdm8_lr0.0001.pth")
# model = torch.load("../ex7/hfmodels/rdm4/2model_epoch_40000_rdm4_lr3e-05.pth")

model = model.cuda()

# dataset = MyOwnDataset("revision_hf")
dataset = MyOwnDataset("5_hf_oh_inv_mw_rdc_edg")    #使用以前的反演数据集

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
train_dataset = dataset[train_indices]
test_dataset = dataset[test_indices]
# print(len(train_dataset))
# print(len(test_dataset))
print("划分结束")

#  -----训练参数-----
learning_rate = 3e-5
num_of_epochs = 20000
pre_epoch = 35000
train_batch_size = 64
test_batch_size = 64
weight_decay = 0.02

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size)

crit = torch.nn.L1Loss()
crit = crit.cuda()

# tensorboard:
writer = SummaryWriter("../logs")


# -----训练函数-----
def train():
    model.train()

    lost_all = 0
    num_of_high = 0
    for data in train_dataloader:
        data = data.cuda()
        optimizer.zero_grad()
        output = model(data)
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
        loss.backward()
        lost_all = lost_all + data.num_graphs * loss.item()
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
print("开始训练")
for epoch in range(num_of_epochs):
    print("-----第{}轮训练开始-----".format(epoch + 1))
    train_loss = train()

    # 最终选择模型的代码
    test_loss = test()
    if test_loss < 0.26 and train_loss < 0.26 and num_model < 10:
        torch.save(model, "../ex7/hfmodels/{}model_loss{}_no{}_rdm{}_lr{}.pth".format(model_index, test_loss, num_model, random_index, learning_rate))
        num_model = num_model + 1
        print("训练的损失为：{}".format(train_loss))
        print("测试的损失为：{}".format(test_loss))
        if num_model >= 10 and epoch > 5000:
            break
    if (epoch + 1) % 20 == 0:
        test_loss = test()
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("test_loss", test_loss, epoch)
        print("训练的损失为：{}".format(train_loss))
        print("测试的损失为：{}".format(test_loss))
        if (epoch + 1) % 5000 == 0 and epoch > 1:
            torch.save(model, "../ex7/hfmodels/{}model_loss_{:.3f}_{:.3f}_epoch{}_rdm{}_lr{}.pth".format(model_index, train_loss, test_loss, epoch + pre_epoch + 1,
                                                                                   random_index, learning_rate))
    # writer.add_scalar("train_loss", train_loss, epoch)
    # writer.add_scalar("test_loss", test_loss, epoch)

writer.close()


# 保存模型
torch.save(model, "../ex7/hfmodels/{}model_epoch_{}_rdm{}_lr{}.pth".format(model_index,
                                                                           num_of_epochs + pre_epoch,
                                                                        random_index,
                                                                         learning_rate
                                                                                  ))