import sys

import pandas as pd

MODE = -1

# A位元素列表：
a_elem_list = ["Ma", "Fa", "Cs", "Na"]
# B位元素列表：
b_elem_list = ["Bi", "Pb", "Cu", "Ag", "Sb", "Sn" ]
# C位元素列表：
c_elem_list = ["Cl", "Br", "S", "I"]


# 探究MaBC的B位和C位元素 B-Bi,Pb C-I,Cl,Br 还有不同配比。
te_list = ["Au", "Ag"]
be_list = ["SnO2", "In2O3"]
Ma_blist = ["Pb", "Bi", "Va"]  # Va表示空位
Ma_clist = ["I", "Br", "Cl", "Va"]

if MODE == -1:
    sto_list = [0.1, 0.2, 0.25, 0.33, 0.5, 0.66, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0]
else:
    sto_list = [0.1, 0.2, 0.25, 0.33, 0.5, 0.66, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0]

# 厚度信息：
if MODE == -1:
    te_tk_list = [50]
    be_tk_list = [100]
    mat_tk_list = [100]
else:
    te_tk_list = [50, 100, 200, 300]
    be_tk_list = [100, 200, 300, 400, 500, 600, 700, 800]
    mat_tk_list = [100, 200, 300, 400, 500, 600, 700, 800]

# 用来存储测试集
test_statistics = pd.DataFrame()


file_num = 1
total_len = 0

for te in te_list:
    for be in be_list:
        for b1_index in range(0, len(Ma_blist)-1):
            for b2_index in range(b1_index + 1, len(Ma_blist)):
                for sb2 in sto_list:
                    for c1_index in range(0, len(Ma_clist)-1):
                        for c2_index in range(c1_index + 1, len(Ma_clist)):
                            for sc2 in sto_list:
                                for te_tk in te_tk_list:
                                    for be_tk in be_tk_list:
                                        for mat_tk in mat_tk_list:
                                            b1 = Ma_blist[b1_index]
                                            b2 = Ma_blist[b2_index]
                                            c1 = Ma_clist[c1_index]
                                            c2 = Ma_clist[c2_index]
                                            sb1 = 1.0
                                            sc1 = 1.0
                                            # 防止出错，进行判定：
                                            if c1 == "Va" or b1 == "Va":
                                                print("error, c1 or b1 is vacancy")
                                                sys.exit()
                                            if b1 == b2 or c1 == c2:
                                                print("error, repeated items")
                                                print(b1, b2, c1, c2)
                                                sys.exit()

                                            if b2 == "Va":
                                                if sb2 == 1.0:
                                                    b_elem = b1
                                                    sb = "1"
                                                    b = b_elem + sb
                                                else:
                                                    break
                                            else:
                                                sb1_norm = sb1/(sb1 + sb2)
                                                sb2_norm = sb2/(sb1 + sb2)
                                                sb1_norm = format(sb1_norm, '.2f')
                                                sb2_norm = format(sb2_norm, '.2f')
                                                b = b1 + sb1_norm + b2 + sb2_norm

                                            if c2 == "Va":
                                                if sc2 == 1.0:
                                                    c_elem = c1
                                                    sc = "1"
                                                    c = c_elem + sc
                                                else:
                                                    break
                                            else:
                                                sc1_norm = sc1 / (sc1 + sc2)
                                                sc2_norm = sc2 / (sc1 + sc2)
                                                sc1_norm = format(sc1_norm, '.2f')
                                                sc2_norm = format(sc2_norm, '.2f')
                                                c = c1 + sc1_norm + c2 + sc2_norm

                                            mat = "Ma" + b + c
                                            row = pd.DataFrame(
                                                [total_len, mat, "pev", te, be, te_tk, be_tk, mat_tk]
                                            )
                                            row = pd.DataFrame(row.values.T)
                                            test_statistics = pd.concat([test_statistics, row], axis=0)
                                            total_len = total_len + 1

                                            if total_len % 1000 == 0:
                                                print("----------")
                                                print("step:{}".format(total_len))
                                                print("length of test = ", test_statistics.shape[0])
                                            if total_len % 125000 == 0:
                                                print("Saving...")
                                                test_statistics.to_excel(
                                                    "../ex6/test2/test_statistics{}.xlsx".format(file_num))
                                                file_num = file_num + 1
                                                test_statistics = pd.DataFrame()
                                            if MODE == -1545:
                                                if total_len > 100:
                                                    test_statistics.to_excel(
                                                        "../ex6/test2/test_statistics{}.xlsx".format(file_num))
                                                    sys.exit()


test_statistics.to_excel("../ex6/test2/test_statistics{}.xlsx".format(file_num))
