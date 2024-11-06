import torch.nn
import random
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from Net9 import *
from GetDataSet import *

random_index = 9
random_list = [114, 930, 586, 247, 108, 188, 597, 299, 905, 671]

def weighted_crit(output, target, weight):
    dis = abs(output - target)
    weight = torch.tensor(weight)
    dis = dis.cpu()
    weighted_dis = torch.mul(dis, weight)
    ref = torch.zeros_like(weighted_dis)
    weighted_loss = crit(weighted_dis, ref)
    return weighted_loss


# -----训练所需要定义的变量-----
# model_index = 9
# exp_index = 1
model = Net9()
model = model.cuda()
# dataset = MyOwnDataset("DataSet_inverse_expanded_weighted_reduced_edge")
hfset = MyOwnDataset("5_hf_oh_inv_mw_rdc_edg")
cuset = MyOwnDataset("5_cu_oh_inv_mw_rdc_edg")


# -----划分训练集、测试集-----
random.seed(random_list[random_index]) # 一开始是191
cuset_size = len(cuset)
indices = list(range(cuset_size))
random.shuffle(indices)
remove_ratio, test_ratio = 0.7, 0.3
remove_size = int(remove_ratio * cuset_size)
test_size = cuset_size - remove_size
remove_indices, test_indices = indices[:remove_size], indices[remove_size:]
test_dataset = cuset[test_indices]

total_dataset = hfset
total_size = len(total_dataset)
indices2 = list(range(total_size))
random.shuffle(indices2)
train_ratio, val_ratio = 0.9, 0.1
train_size = int(train_ratio * total_size)
val_size = total_size - train_size
train_indices, val_indices = indices2[:train_size], indices2[train_size:]

train_dataset = total_dataset[train_indices] + cuset[remove_indices]
val_dataset = total_dataset[val_indices] + cuset[remove_indices]


#  -----训练参数-----
learning_rate = 3e-4

num_of_epochs = 15000
pre_epoch = 0
train_batch_size = 64
val_batch_size = 64
test_batch_size = 64
weight_decay = 0.02

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size)
crit = torch.nn.L1Loss()
crit = crit.cuda()

# tensorboard:
writer = SummaryWriter("../ex7/logs")


# -----训练函数-----
def train():
    model.train()

    lost_all = 0
    # lost_all_origin = 0
    num_of_high = 0
    for data in train_dataloader:
        # print(len(train_dataloader))
        # print(len(data))
        data = data.cuda()
        optimizer.zero_grad()
        output = model(data)
        label = data.y
        #print (label)
        label = label.squeeze()
        label1 = label.tolist()
        weight = []
        for yi in label1:
            if yi >= 3.0:
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


def val():
    model.eval()
    with torch.no_grad():
        val_loss_all = 0
        for data in val_dataloader:
            data = data.cuda()
            label = data.y
            label = label.squeeze()
            output = model(data)
            loss = crit(output, label)
            val_loss_all = val_loss_all + data.num_graphs * loss.item()

        return val_loss_all / len(val_dataset)


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
num_model = 0
train_min = 0.8
val_min = 0.8
test_min = 0.8
min_index = -1

for epoch in range(num_of_epochs):
    print("-----第{}轮训练开始-----".format(epoch + 1))
    train_loss = train()
    # test_loss = test()
    # if test_loss < 0.34:
    #     torch.save(model, "../new_model/ef_model/4-model_decay0.02_91_loss{}_no{}.pth".format(test_loss, num_model))
    #     num_model = num_model + 1
    #     print("训练的损失为：{}".format(train_loss))
    #     print("测试的损失为：{}".format(test_loss))
    #     break
    if (epoch + 1) % 20 == 0:
        val_loss = val()
        test_loss = test()
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("test_loss", test_loss, epoch)
        print("训练的损失为：{}".format(train_loss))
        print("验证的损失为：{}".format(val_loss))
        print("测试的损失为：{}".format(test_loss))
        if test_loss < test_min:
            train_min = train_loss
            val_min = val_loss
            test_min = test_loss
            min_index = epoch
        if (epoch+1) % 3000 == 0:
            torch.save(model, "../ex7/transfer/rdm{}_loss_{:.3f}_{:.3f}_{:.3f}_model_lr_{}_epoch_{}_remove_{}.pth".format(random_index,
                                                                                                              train_loss, val_loss, test_loss,
                                                                                                learning_rate,
                                                                                                epoch + pre_epoch,
                                                                                                remove_ratio
                                                                                                ))
    # writer.add_scalar("train_loss", train_loss, epoch)
    # writer.add_scalar("test_loss", test_loss, epoch)

writer.close()


# 保存模型
torch.save(model, "../ex7/transfer/rdm{}_model_lr_{}_epoch_{}_remove_{}.pth".format(random_index,
                                                                                  learning_rate,
                                                                                  num_of_epochs + pre_epoch ,
                                                                                  remove_ratio
                                                                       ))

print("第{}轮训练得到最小的test:{}，train:{}, val:{}".format(min_index, test_min, train_min, val_min))
