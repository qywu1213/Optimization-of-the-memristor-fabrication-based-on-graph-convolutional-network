import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np

metal_needed = ['Li', 'C', 'Mg', 'Al', 'Si', 'Ti', 'Co', 'Cu', 'Zn', 'Ga', 'Ge', 'Zr', 'Mo', 'Ru',
                'Ag', 'In', 'Ba', 'Hf', 'Ta', 'W', 'Pt', 'Au', 'Pb', 'Sn',
                'Pd', 'Y', 'Ni', 'V', 'Sr', 'Ir', 'Gd', 'Nb', 'Mn', 'Fe', 'Cr', 'In', 'Cd',
                'Ma', 'Fa', 'Bi', 'Cs', 'Na', 'Sb']
non_metal_needed = ["C", "N", "O", "P", "S", "Cl", 'In', "Te", "Br", "I", "Occ"]
elem_needed = metal_needed + non_metal_needed


element_features = pd.read_excel("../ex5/elem_list.xlsx")
element_features_needed = element_features[element_features.Symbol.isin(elem_needed)]
ordered_elems = element_features_needed.iloc[:, 0]

elems_array = np.array(ordered_elems)
elems_list = elems_array.tolist()


feature_list = pd.DataFrame()
for i in range(len(elems_list)):
    features = np.zeros(len(elems_list))
    features[i] = 1
    vector = pd.DataFrame(features)
    vector = pd.DataFrame(vector.values.T)
    feature_list = pd.concat([feature_list, vector])


feature_list.to_excel("../ex5/OneHotList.xlsx")
element_features_needed.to_excel("../ex5/OneHot_Head.xlsx")
print("ok")
# elem_list_oh = pd.concat([element_features_needed, feature_list], axis=1)
# elem_list_oh.to_excel("../ex4/elem_list_oh.xlsx")







