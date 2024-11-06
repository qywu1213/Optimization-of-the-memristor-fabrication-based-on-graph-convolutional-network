import pandas as pd

# 读取没有处理之前的数据（开关比，包含三层阻变层以及ITO）
train_set_0 = pd.read_excel("../ex4/hfset.xlsx")
print(train_set_0.head())

# # 查找筛选要求的行
# drop_list = train_set_0.loc[
#     (train_set_0['Type'] == 'Trilayer') | (train_set_0['TE'] == "ITO") | (train_set_0['BE'] == 'ITO')]
# drop_index = drop_list.iloc[:, 0]
# drop_index = drop_index.to_list()
# # print(drop_index)


# 其他直接删除的方法
train_set_1 = train_set_0[train_set_0.Type.isin(['Trilayer']) == False]
# train_set_2 = train_set_1[train_set_1.TE.isin(['ITO', 'Pd', 'SnO2', 'Sn']) == False]
# train_set_3 = train_set_2[train_set_2.BE.isin(['ITO', 'Pd', 'SnO2', 'Sn']) == False]
print(train_set_1.shape[0] - train_set_0.shape[0])

train_set_1.to_excel("../ex4/hfset1.xlsx")