import pandas as pd
from Net9 import *
import torch
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data

MODE = -114

# 非金属表，只有5个很少：
non_metal_list = ["C", "N", "O", "P", "S", "In", "Te", "Occ"]
# A位元素列表：
a_elem_list = ["Ma", "Fa", "Cs", "Na"]
# B位元素列表：
b_elem_list = ["Bi", "Pb", "Cu", "Ag", "Sb", "Sn"]
# C位元素列表：
c_elem_list = ["Cl", "Br", "S", "I"]

# 测试集
test_index = 1
num_test_files = 34
root_path = "../ex6/test/"

# 准备工作：
item_index = 1
mat_index = 2
te_index = 4
be_index = 5
tte_index = 6
tbe_index = 7
tmt_index = 8

elem_embedding_list = pd.read_excel("../ex5/elem_embed_list.xlsx")


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


# 分割元素符号和后面计量比，如输入Ma4.5返回Ma和4.5
def elem_sto_split(elem_sto):
    pivot = 0
    # print(elem_sto)
    for i in range(0, len(elem_sto)):
        if elem_sto[i].isdigit():
            pivot = i
            break
    # print("pivot:{}".format(pivot))
    if pivot == 0:
        sto = 1.0
        output_elem = elem_sto
    else:
        sto = float(elem_sto[pivot:])
        output_elem = elem_sto[:pivot]
    # print(output_elem)
    # print(sto)
    return output_elem, sto


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


# 这个函数处理二元材料，返回一个金属向量和一个非金属向量：
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


# 该函数将钙钛矿类型的材料名称转化为ABC对应的加权平均向量
def get_halide_mat_vector(mat):
    # 将材料按照大小写顺序分成多组元素，元素后面跟随化学计量比，存储在res中
    res = []
    start = 0
    for i in range(1, len(mat)):
        if mat[i].isupper():
            res.append(mat[start:i])
            start = i
    res.append(mat[start:])

    # 将res中的每一个元素分成两部分，分别为元素名称和化学计量比。化学计量比缺失的说明是1，用1代替。elem中存储元素名称，sto存储计量比
    elem = []
    sto = []
    for item in res:
        # elem_name = remove_digits(item)
        elem_item, sto_item = elem_sto_split(item)
        elem.append(elem_item)
        sto.append(sto_item)

    # 将elem中的各个元素按照名称分为A位、B位和C位
    flag = []
    for index in range(0, len(elem)):
        if index == 0:
            flag.append("A")
        else:
            elem_name = elem[index]
            if elem_name in a_elem_list:
                flag.append("A")
            elif elem_name in b_elem_list:
                flag.append("B")
            elif elem_name in c_elem_list:
                flag.append("C")
            else:
                print("error, some elem lost in perov:{}".format(elem_name))
                flag.append("Error")

    # 设置A、B、C位的权重：
    weight_a = []
    elem_a = []
    weight_b = []
    elem_b = []
    weight_c = []
    elem_c = []
    for index in range(0, len(flag)):
        if flag[index] == "A":
            weight_a.append(sto[index])
            elem_a.append(elem[index])
        elif flag[index] == "B":
            weight_b.append(sto[index])
            elem_b.append(elem[index])
        elif flag[index] == "C":
            weight_c.append(sto[index])
            elem_c.append(elem[index])
    # 将权重归一化：
    total_sum_a = sum(weight_a)
    weight_a_norm = [num / total_sum_a for num in weight_a]
    total_sum_b = sum(weight_b)
    weight_b_norm = [num / total_sum_b for num in weight_b]
    total_sum_c = sum(weight_c)
    weight_c_norm = [num / total_sum_c for num in weight_c]

    # 根据权重和元素名称返回向量：
    vector_a = return_vector(elem_a[0])
    vector_a = vector_a * weight_a_norm[0]
    for index in range(1, len(elem_a)):
        next_elem_vector = return_vector(elem_a[index])
        next_elem_vector_norm = next_elem_vector * weight_a_norm[index]
        vector_a = vector_a + next_elem_vector_norm

    vector_b = return_vector(elem_b[0])
    vector_b = vector_b * weight_b_norm[0]
    for index in range(1, len(elem_b)):
        next_elem_vector = return_vector(elem_b[index])
        next_elem_vector_norm = next_elem_vector * weight_b_norm[index]
        vector_b = vector_b + next_elem_vector_norm

    vector_c = return_vector(elem_c[0])
    vector_c = vector_c * weight_c_norm[0]
    for index in range(1, len(elem_c)):
        next_elem_vector = return_vector(elem_c[index])
        next_elem_vector_norm = next_elem_vector * weight_c_norm[index]
        vector_c = vector_c + next_elem_vector_norm

    vector_a = vector_a.unsqueeze(0)
    vector_b = vector_b.unsqueeze(0)
    vector_c = vector_c.unsqueeze(0)
    return vector_a, vector_b, vector_c


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


def item2data(file, num):
    te = file.iloc[num, te_index]
    be = file.iloc[num, be_index]
    mat = file.iloc[num, mat_index]
    tte = file.iloc[num, tte_index]
    tbe = file.iloc[num, tbe_index]
    tmt = file.iloc[num, tmt_index]

    # 4- TE； 5- BE
    vector_a, vector_b, vector_c = get_halide_mat_vector(mat)
    vec41, vec42 = mat_split(te)
    vec51, vec52 = mat_split(be)

    node = torch.cat((vec41, vec42,
                      vector_a, vector_c, vector_a, vector_c,
                      vec51, vec52,
                      vector_b, vector_c), 0)

    return node, tte, tbe, tmt


# 加载测试模型
model = torch.load("../ex5/models/picked_model/used_model1.pth")
model = model.cuda()

# 输出文件：
# 每次预测5个生成5个
final_files_index = 26
final_file = pd.DataFrame()
num_processed_item = 0


model.eval()
with torch.no_grad():
    for i in range(final_files_index, final_files_index + 8):
        print("processed file index:{}".format(i + 1))
        file_path = root_path + "test_statistics{}.xlsx".format(i + 1)
        print(file_path)
        test_file = pd.read_excel(file_path)
        # test_file = test_file.iloc[:, 1:]
        if MODE == -1:
            num_items = 10
        else:
            num_items = test_file.shape[0]

        for index in tqdm(range(num_items)):
            node_features, tte, tbe, tmt = item2data(test_file, index)

            # 边的信息
            attr = [tmt] * 64
            for j in range(0, 10):
                attr[j] = tte
            for j in range(22, 32):
                attr[j] = tbe
            edge_attr = torch.tensor(attr, dtype=torch.float32)
            edge_attr = edge_attr.unsqueeze(1)
            # 约化后的厚度
            edge_attr = edge_attr / 10.0
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=0)
            data = data.cuda()
            output = model(data)
            output_cpu = output.tolist()
            item = pd.DataFrame(test_file.iloc[index].values)
            item = item.T
            output_df = pd.DataFrame([output_cpu])
            output_df = output_df.T
            row_df = pd.concat((item, output_df), axis=1)
            final_file = pd.concat((final_file, row_df))

            num_processed_item = num_processed_item + 1
            # if num_processed_item % 1000 == 0:
            #     print("file{}, the index of the item processed is {}".format(i, index))
            if num_processed_item % 125000 == 0:
                final_file.to_csv(
                    "../ex6/out/prediction{}.csv".format(final_files_index))
                final_files_index = final_files_index + 1
                final_file = pd.DataFrame()
final_file.to_csv("../ex6/out/prediction{}.csv".format(final_files_index))

print("-----over-----")
