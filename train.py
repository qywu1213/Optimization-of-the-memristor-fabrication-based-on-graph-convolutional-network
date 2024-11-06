import torch.nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from Net9 import *
from GetDataSet import *

# -----训练所需要定义的变量-----
model_index = 9
model = Net9()
# model = torch.load("../new_model_rs/final_model_decay0.1_1")
model = model.cuda()
dataset = MyOwnDataset("DataSet_OneHot_inverse_expanded_weighted_reduced_edge")

#  -----训练参数-----
learning_rate = 3e-4
num_of_epochs = 3500
train_batch_size = 64
test_batch_size = 64
weight_decay = 0.05

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
train_dataloader = DataLoader(dataset, batch_size=train_batch_size)
test_dataloader = DataLoader(dataset, batch_size=test_batch_size)
crit = torch.nn.L1Loss()
crit = crit.cuda()

# tensorboard:
writer = SummaryWriter("../logs")


# -----训练函数-----
def train():
    model.train()

    lost_all = 0
    for data in train_dataloader:
        # print(len(train_dataloader))
        # print(len(data))
        data = data.cuda()
        optimizer.zero_grad()
        output = model(data)
        label = data.y
        label = label.squeeze()
        loss = crit(output, label)
        loss.backward()
        lost_all = lost_all + data.num_graphs * loss.item()
        optimizer.step()
        # print("one batch")

    return lost_all / len(dataset)


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

        return test_loss_all / len(dataset)


# 正式训练和测试：

for epoch in range(num_of_epochs):
    print("-----第{}轮训练开始-----".format(epoch + 1))
    train_loss = train()
    # test_loss = test()
    # if test_loss < 0.47:
    #     torch.save(model, "../new_model_rs/final_model_decay{}_used".format(weight_decay))
    #     print("训练的损失为：{}".format(train_loss))
    #     print("测试的损失为：{}".format(test_loss))
    #     break
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
torch.save(model, "../new_model_rs/model{}_lr_{}_epoch_{}_batch_{}_decay_{}".format(model_index,
                                                                                  learning_rate,
                                                                                  num_of_epochs,
                                                                                  train_batch_size,
                                                                                  weight_decay))
# torch.save(model, "../new_model_rs/final_model_decay{}_3".format(weight_decay))
