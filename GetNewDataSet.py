import pandas as pd
from torch_geometric.data import InMemoryDataset, download_url, Data
import torch
from tqdm import tqdm
import numpy as np

# 读取筛选后的样本表
#train_items = pd.read_excel("../ex5/hfset1.xlsx")
train_items = pd.read_excel("../ex5/cuset1.xlsx")

# 用于处理元素的embedding 由于图神经网络的要求，这些元素统一去除空格，不再区分金属和非金属。
elem_embedding_list = pd.read_excel("../ex5/elem_embed_list.xlsx")

# 非金属表，只有5个很少：
non_metal_list = ["C", "N", "O", "P", "S", "In", "Te", "Occ"]


# 用来移除字符串末尾的数字和x,训练的数据集里只有最多一位数字
def remove_digits(mat):
    if mat[-1].isdigit() or mat[-1] == 'x':
        output = mat[:-1]
    else:
        output = mat
    if output[-1].isdigit() or output[-1] == 'x':
        print("failure, mat is {}".format(output))
        # if output == 'O':
        #     print("big error! mistaken")
    return output


# 根据元素名称返回向量的函数：
def return_vector(elem):
    elem_vector = elem_embedding_list.query("Symbol == @elem").copy()
    if elem_vector.shape[0] == 0:
        print("error, lost elem name is {}".format(elem))
    elem_vector = elem_vector.values[:, 2:]
    elem_vector = elem_vector.astype(np.float32)
    elem_vector = torch.from_numpy(elem_vector)
    elem_vector = elem_vector.squeeze()
    return elem_vector


# 用来根据元素大小写分割材料中的元素，然后给出对应的向量的函数：
def mat_split(mat):
    # 将材料按照大小写顺序分成多组元素
    res = []
    start = 0
    for i in range(1, len(mat)):
        if mat[i].isupper():
            res.append(mat[start:i])
            start = i
    res.append(mat[start:])
    if len(res) >= 3:
        print("error, some mat have more than two elements, mat is {}".format(mat))

    # 这个地方对于只有一个元素应该也没有问题，这两个就是对应同一个向量。
    elem1 = res[0]
    elem2 = res[-1]

    # 判断元素是否是HfOx或者以小写的x结尾，这种情况需要去掉x
    # 这里还需要一步，如果材料分割元素后边有化学计量比的数字，要把数字去掉：
    # 这两步合并为一步，写了一个函数用来处理：
    elem1 = remove_digits(elem1)
    elem2 = remove_digits(elem2)

    # 判断一下，如果材料只含一种元素，那么大概率为金属元素，非金属部分用列表中的Occ补足（主要有一个N掺杂的例外）
    if elem1 == elem2:
        if elem1 not in non_metal_list:
            elem2 = 'Occ'
        else:
            elem1 = 'Occ'

    # 这里elem1 和 elem2 应该都处理好了；

    # 根据元素名称返回向量(构造函数)
    if elem1 == "O2" or elem2 == "O2":
        print("find o2 error, mat it {}".format(mat))
    elem1_vector = return_vector(elem1)
    elem2_vector = return_vector(elem2)
    # 这一步是为了让data的node特征为[节点数，特征数]的格式
    elem1_vector = elem1_vector.unsqueeze(0)
    elem2_vector = elem2_vector.unsqueeze(0)
    return elem1_vector, elem2_vector


# 用于分割name的函数
def name2mat(name):
    res = []
    for c in ['/', '-']:
        if c in name:
            res = name.split(c)
            break
    if not res:
        res.append(name)
    return res


def data_making(items):
    num_of_items = items.shape[0]
    # 给每个序号代表了什么编号
    mat_index = 2
    type_index = 3
    te_index = 4
    thick_te_index = 5
    be_index = 6
    thick_be_index = 7
    thick_mat_index = 8
    label = 9

    # 处理每一行的信息生成data

    for index in tqdm(range(0, num_of_items)):
        name = items.iloc[index, mat_index]
        mat = name2mat(name)
        # 判断有几个材料，如果只有一个，说明是HfO2
        if len(mat) == 1:
            mat1 = mat[0]
            mat2 = mat[0]
        else:
            mat1, mat2 = mat

        # 这一部分处理的是主体结构的信息。
        if items.iloc[index, type_index] == 'Doping':
            # 这里mat1是掺杂金属，mat2是主体材料
            vec31, vec32 = mat_split(mat1)
            vec11, vec12 = mat_split(mat2)
            vec21 = vec11
            vec22 = vec12

        # 这里是剩下情况，就是HFO2和双层都可以这样处理，只要没有掺杂就行：
        else:
            # print("the processed item name : {}".format(name))
            # 对于bilayer的忆阻器，返回双层结构的两层对应元素的向量，掺杂元素用两层的平均值
            vec11, vec12 = mat_split(mat1)
            vec21, vec22 = mat_split(mat2)
            vec31 = 0.5*(vec11 + vec21)
            vec32 = 0.5*(vec12 + vec22)

        # 处理电极信息
        # 准备好电极相关的信息，这一部分所有情况都是一样处理：
        te_name = items.iloc[index, te_index]
        te_thickness = items.iloc[index, thick_te_index]
        be_name = items.iloc[index, be_index]
        be_thickness = items.iloc[index, thick_be_index]
        vec41, vec42 = mat_split(te_name)
        vec51, vec52 = mat_split(be_name)

    # 备注：1-第一层阻变层； 2- 第二层阻变层； 3- 掺杂材料（没有的取两层阻变层的平均）4- TE； 5- BE
    #     return vec41, vec42, vec11, vec12, vec21, vec22, vec51, vec22, vec31, vec32


# 构建自定义数据集
class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.weights = []  # 存储每个数据的权重向量
        # for data in self.data_list:
        #     # 根据自己的需求设置权重向量
        #     weight = 1.0 if data.y >= 3.0 else 0.5
        #     self.weights.append(weight)
        #     self.weights.append(weight)

    # 返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    # 返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['MemDataset.dataset']

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
        items = train_items
        num_of_items = items.shape[0]
        # 给每个序号代表了什么编号
        mat_index = 2
        type_index = 3
        te_index = 4
        thick_te_index = 5
        be_index = 6
        thick_be_index = 7
        thick_mat_index = 8
        # 这个代表标签！！！
        # hf 数据集：开关比序号是9
        # cu 数据集：开关比序号是11，endurance是9
        label = 11

        # 处理每一行的信息生成data

        for index in tqdm(range(0, num_of_items)):

            # 邻接矩阵
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

            # 构建节点特征
            name = items.iloc[index, mat_index]
            mat = name2mat(name)
            # 判断有几个材料，如果只有一个，说明是HfO2
            if len(mat) == 1:
                mat1 = mat[0]
                mat2 = mat[0]
            else:
                mat1, mat2 = mat

            # 这一部分处理的是主体结构的信息。
            if items.iloc[index, type_index] == 'Doping' or items.iloc[index, type_index] == 'Doped':
                # 这里mat1是掺杂金属，mat2是主体材料
                vec31, vec32 = mat_split(mat1)
                vec11, vec12 = mat_split(mat2)
                vec21 = vec11
                vec22 = vec12

            # 这里是剩下情况，就是HFO2和双层都可以这样处理，只要没有掺杂就行：
            else:
                # print("the processed item name : {}".format(name))
                # 对于bilayer的忆阻器，返回双层结构的两层对应元素的向量，掺杂元素用两层的平均值
                vec11, vec12 = mat_split(mat1)
                vec21, vec22 = mat_split(mat2)
                vec31 = 0.5 * (vec11 + vec21)
                vec32 = 0.5 * (vec12 + vec22)

            # 处理电极信息
            # 准备好电极相关的信息，这一部分所有情况都是一样处理：
            te_name = items.iloc[index, te_index]
            te_thickness = items.iloc[index, thick_te_index]
            be_name = items.iloc[index, be_index]
            be_thickness = items.iloc[index, thick_be_index]
            rs_thickness = items.iloc[index, thick_mat_index]
            vec41, vec42 = mat_split(te_name)
            vec51, vec52 = mat_split(be_name)

            node_features = torch.cat((vec41, vec42, vec11, vec12, vec21, vec22, vec51, vec52, vec31, vec32), 0)
            # 备注：1-第一层阻变层； 2- 第二层阻变层； 3- 掺杂材料（没有的取两层阻变层的平均）4- TE； 5- BE
            node_features_inverse = torch.cat((vec51, vec52, vec21, vec22, vec11, vec22, vec41, vec42, vec31, vec32))

            # 获得target
            target = items.iloc[index, label]
            target = target.astype(np.float32)
            # 这一句可能存在问题，我不知道这样写符不符合格式规范（！！）
            target = torch.tensor([[target]], dtype=torch.float32)
            x = node_features
            y = target

            # 边的信息：
            attr = [rs_thickness] * 64
            for i in range(0, 10):
                attr[i] = te_thickness
            for i in range(22, 32):
                attr[i] = be_thickness
            edge_attr = torch.tensor(attr, dtype=torch.float32)
            edge_attr = edge_attr.unsqueeze(1)
            edge_attr = edge_attr / 10.0

            # 反演边的构造：
            attr_inverse = [rs_thickness] * 64
            for i in range(0, 10):
                attr_inverse[i] = be_thickness
            for i in range(22, 32):
                attr_inverse[i] = te_thickness
            edge_attr_inverse = torch.tensor(attr_inverse, dtype=torch.float32)
            edge_attr_inverse = edge_attr_inverse.unsqueeze(1)
            edge_attr_inverse = edge_attr_inverse / 10.0

            # 存储data：
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)
            # 反演数据加入：
            # data_inverse = Data(x=node_features_inverse, edge_index=edge_index, edge_attr=edge_attr_inverse, y=y)
            # data_list.append(data_inverse)
            # print("add a data")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


dataset = MyOwnDataset("revision_cu")