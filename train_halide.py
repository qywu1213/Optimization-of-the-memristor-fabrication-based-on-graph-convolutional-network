import torch.nn
import random
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from Net9 import *
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
net_index = 9
exp_index = "2_1"
# model = Net9()
model = torch.load("../ex5/models/2model_net9_lr3e-05_epoch_20000_decay_0.02.pth")
model = model.cuda()
# dataset = MyOwnDataset("DataSet_inverse_expanded_weighted_reduced_edge")
hfset = MyOwnDataset("5_hf_oh_inv_mw_rdc_edg")
cuset = MyOwnDataset("5_cu_oh_inv_mw_rdc_edg")
halide_set = MyOwnDataset("5_halide_m1_inverse_edg_rdc")

# -----划分训练集、测试集-----
random.seed(114)
halide_set_size = len(halide_set)
indices = list(range(halide_set_size))
random.shuffle(indices)
remove_ratio, test_ratio = 0.9, 0.1
remove_size = int(remove_ratio * halide_set_size)
test_size = halide_set_size - remove_size
remove_indices, test_indices = indices[:remove_size], indices[remove_size:]
test_dataset = halide_set[test_indices]

hf_dataset = hfset
cu_dataset = cuset

train_dataset = hf_dataset + cu_dataset + halide_set[remove_indices]
# 用全部的铜基、Hf基和90%的卤化物忆阻器数据训练，10%卤化物忆阻器的数据做验证。


#  -----训练参数-----
learning_rate = 1e-5

num_of_epochs = 10000
pre_epoch = 20000
train_batch_size = 64
val_batch_size = 64
test_batch_size = 64
weight_decay = 0.02

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size)
crit = torch.nn.L1Loss()
crit = crit.cuda()

# tensorboard:
writer = SummaryWriter("../ex5/logs")


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


# def val():
#     model.eval()
#     with torch.no_grad():
#         val_loss_all = 0
#         for data in val_dataloader:
#             data = data.cuda()
#             label = data.y
#             label = label.squeeze()
#             output = model(data)
#             loss = crit(output, label)
#             val_loss_all = val_loss_all + data.num_graphs * loss.item()
#
#         return val_loss_all / len(val_dataset)


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
precise_flag = False

# 该参数手动设置，用来判断是否开始记录模型
final_flag = True

for epoch in range(num_of_epochs):
    print("-----第{}轮训练开始-----".format(epoch + 1))
    train_loss = train()

    if train_loss < 0.24 and final_flag:
        test_loss = test()
        if abs(train_loss - test_loss) <= 0.03 and test_loss < 0.24:
            torch.save(model, "../ex5/models/prepared_models/5_model_decay0.02_loss{}_{}_no{}_step{}.pth".format(train_loss,
                                                                                                 test_loss,
                                                                                                 num_model, epoch))
            num_model = num_model + 1
            print("训练的损失为：{}".format(train_loss))
            print("测试的损失为：{}".format(test_loss))
            if num_model > 100:
                break
    if (epoch + 1) % 20 == 0:
        test_loss = test()
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("test_loss", test_loss, epoch)
        print("训练的损失为：{}".format(train_loss))
        print("测试的损失为：{}".format(test_loss))
        # if test_loss < 0.24 and train_loss < 0.24 and abs(test_loss - train_loss) < 0.03 and not precise_flag:
        #     precise_flag = True
    # writer.add_scalar("train_loss", train_loss, epoch)
    # writer.add_scalar("test_loss", test_loss, epoch)

writer.close()

# 保存模型
torch.save(model, "../ex5/models/{}model_net{}_lr{}_epoch_{}_decay_{}.pth".format(exp_index,
                                                                             net_index,
                                                                             learning_rate,
                                                                             num_of_epochs + pre_epoch,
                                                                             weight_decay))
