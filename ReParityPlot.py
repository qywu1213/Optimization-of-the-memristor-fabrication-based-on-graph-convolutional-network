import pandas as pd
import torch.nn
import random
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from Net10 import *
from GetDataSet import *
import pandas

# load dataset
# hfset = MyOwnDataset("5_hf_oh_inv_mw_rdc_edg") #应该是ex4中的嵌入方式。当时还没有用卤化物数据。
cuset = MyOwnDataset("revision_cu_transfer")

# create models
model0 = Net10()
model3 = Net10()
model7 = Net10()
model9 = Net10()
# load models
model0 = torch.load("../ex4/models/1model10_lr_0.001_epoch_10000_remove_0_decay_0.02.pth")
model3 = torch.load("../ex4/models/1model10_lr_0.001_epoch_13000_remove_0.3_decay_0.02.pth")
model7 = torch.load("../ex4/models/1model10_lr_0.001_epoch_13000_remove_0.7_decay_0.02.pth")
model9 = torch.load("../ex4/models/1model10_lr_0.001_epoch_13000_remove_0.9_decay_0.02.pth")

# load dataset
test_dataloader = DataLoader(cuset, batch_size=64, shuffle=False)

real_value_list = []
predicted_value_list0 = []
predicted_value_list3 = []
predicted_value_list7 = []
predicted_value_list9 = []

for data in test_dataloader:
    print("start computing------")
    data = data.cuda()
    real_value = data.y
    real_value = real_value.squeeze()
    real_value = real_value.tolist()
    real_value_list.append(real_value)
    predicted_value0 = model0(data)
    predicted_value0 = predicted_value0.tolist()
    predicted_value3 = model3(data)
    predicted_value3 = predicted_value3.tolist()
    predicted_value7 = model7(data)
    predicted_value7 = predicted_value7.tolist()
    predicted_value9 = model9(data)
    predicted_value9 = predicted_value9.tolist()
    predicted_value_list0.append(predicted_value0)
    predicted_value_list3.append(predicted_value3)
    predicted_value_list7.append(predicted_value7)
    predicted_value_list9.append(predicted_value9)

real_value_df = pd.DataFrame(real_value_list)
real_value_df = pd.DataFrame(real_value_df.values.T)
predicted_value_df0 = pd.DataFrame(predicted_value_list0)
predicted_value_df0 = pd.DataFrame(predicted_value_df0.values.T)
predicted_value_df3 = pd.DataFrame(predicted_value_list3)
predicted_value_df3 = pd.DataFrame(predicted_value_df3.values.T)
predicted_value_df7 = pd.DataFrame(predicted_value_list7)
predicted_value_df7 = pd.DataFrame(predicted_value_df7.values.T)
predicted_value_df9 = pd.DataFrame(predicted_value_list9)
predicted_value_df9 = pd.DataFrame(predicted_value_df9.values.T)
df = pd.concat([real_value_df, predicted_value_df0, predicted_value_df3, predicted_value_df7, predicted_value_df9], axis=1)
df.to_csv("../ex7/transfer.csv")

print("-----over-----")

