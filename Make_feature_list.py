# 该文件用来处理embedding表，让留下来的信息更多，只保留需要的元素。

import pandas as pd
import numpy as np
metal_needed = ['Li', 'Mg', 'Al', 'Si', 'Ti', 'Co', 'Cu', 'Zn', 'Ga', 'Ge', 'Zr', 'Mo', 'Ru',
                'Ag', 'In', 'Ba', 'Hf', 'Ta', 'W', 'Pt', 'Au', 'Pb', 'Sn'
                'Pd', 'Y', 'Ni', 'V', 'Sr', 'Ir', 'Gd', 'Nb', 'Mn', 'Fe', 'Cr', 'Nb']
non_metal_needed = ["C", "N", "O", "P", "S", "Occ"]

elem_needed = metal_needed + non_metal_needed

element_features = pd.read_excel("../TrainSet/element_features_list.xlsx")
element_features_needed = element_features[element_features.Symbol.isin(elem_needed)]

element_features_needed.replace(' ', np.nan, inplace=True)
elem_embedding_list = pd.DataFrame(element_features_needed.dropna(axis=1))
elem_embedding_list.to_excel("../TrainSet/elem_embedding_list.xlsx")
print("over")
