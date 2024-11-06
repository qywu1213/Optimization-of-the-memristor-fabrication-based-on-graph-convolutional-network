import pandas as pd
from torch_geometric.data import InMemoryDataset, download_url, Data
import torch
from tqdm import tqdm
import numpy as np


# 用于处理元素的embedding 由于图神经网络的要求，这些元素统一去除空格，不再区分金属和非金属。
elem_embedding_list = pd.read_excel("../TrainSet/elem_embedding_list_OneHot.xlsx")

# 非金属表，只有5个很少：
non_metal_list = ["C", "N", "O", "P", "S", "Occ"]

# 测试集
num_test_files = 1
root_path = "../ex2/test/"

# 厚度表
te_tk = [20, 50, 100]
len_te_tk = len(te_tk)
be_tk = [20, 50, 100]
len_be_tk = len(be_tk)
sl_tk = [20, 50, 100]
len_sl_tk = len(sl_tk)


# 输入元素名称返回编码
def return_vector(elem):
    print(elem)
    elem = elem.capitalize()
    elem_vector = elem_embedding_list.query("Symbol == @elem").copy()
    if elem_vector.shape[0] == 0:
        print("error, lost elem name is {}".format(elem))
    elem_vector = elem_vector.values[:, 2:]
    elem_vector = elem_vector.astype(np.float32)
    elem_vector = torch.from_numpy(elem_vector)
    elem_vector = elem_vector.squeeze()
    return elem_vector


# 构建自定义数据集
class TestSet(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # 返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    # 返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['test.dataset']

    # # 用于从网上下载数据集
    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
    #     ...

    # 生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        # 构建Data

        # 准备工作：
        te = 0
        m1 = 1
        m2 = 2
        nm1 = 3
        nm2 = 4
        dm = 5
        be = 6

        # 邻接矩阵：
        source_0 = torch.tensor(
            [0, 1, 0, 2, 0, 3, 1, 3, 1, 2, 2, 3, 2, 4, 2, 5, 3, 5, 3, 4, 4, 5, 4, 6, 4, 7, 5, 6, 5, 7, 6, 7],
            dtype=torch.long)
        target_0 = torch.tensor(
            [1, 0, 2, 0, 3, 0, 3, 1, 2, 1, 3, 2, 4, 2, 5, 2, 5, 3, 4, 3, 5, 4, 6, 4, 7, 4, 6, 5, 7, 5, 7, 6],
            dtype=torch.long)
        source_89 = torch.tensor([8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9], dtype=torch.long)
        target_89 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
        source_nodes = torch.cat((source_0, source_89, target_89), 0)
        target_nodes = torch.cat((target_0, target_89, source_89), 0)
        source_nodes = source_nodes.unsqueeze(0)
        target_nodes = target_nodes.unsqueeze(0)
        edge_index = torch.cat((source_nodes, target_nodes), 0)

        # 处理每一行的信息生成data
        for i in range(num_test_files):
            file_path = root_path + "test_statistics{}.xlsx".format(i + 1)
            test_file = pd.read_excel(file_path)
            num_items = test_file.shape[0]
            for index in tqdm(range(num_items)):
                te_name = test_file.iloc[index, te]
                be_name = test_file.iloc[index, be]
                m1_name = test_file.iloc[index, m1]
                m2_name = test_file.iloc[index, m2]
                nm1_name = test_file.iloc[index, nm1]
                nm2_name = test_file.iloc[index, nm2]
                dm_name = test_file.iloc[index, dm]
                print(test_file.iloc[index])

                #  备注：1-第一层阻变层； 2- 第二层阻变层； 3- 掺杂材料（没有的取两层阻变层的平均）4- TE； 5- BE
                vec41 = return_vector(te_name)
                vec42 = return_vector("Occ")
                vec51 = return_vector(be_name)
                vec52 = return_vector("Occ")
                vec11 = return_vector(m1_name)
                vec12 = return_vector(nm1_name)
                vec21 = return_vector(m2_name)
                vec22 = return_vector(nm2_name)
                if dm_name == "None":
                    vec31 = 0.5 * (vec11 + vec21)
                    vec32 = 0.5 * (vec12 + vec22)
                else:
                    vec31 = return_vector(dm_name)
                    vec32 = return_vector("Occ")

                node_features = torch.cat((vec41, vec42, vec11, vec12, vec21, vec22, vec51, vec52, vec31, vec32), 0)

                for te_tk_index in te_tk:
                    for be_tk_index in be_tk:
                        for sl_tk_index in sl_tk:
                            attr = [sl_tk_index] * 64
                            for j in range(0, 10):
                                attr[j] = te_tk_index
                            for j in range(22, 32):
                                attr[i] = be_tk_index
                            edge_attr = torch.tensor(attr, dtype=torch.float32)
                            edge_attr = edge_attr.unsqueeze(1)

                            # 存储data：
                            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=0)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


test_dataset = TestSet("TestDataSet_expanded")

