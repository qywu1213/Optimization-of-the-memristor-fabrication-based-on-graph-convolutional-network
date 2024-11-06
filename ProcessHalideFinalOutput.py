import pandas as pd


from tqdm import tqdm


num_of_outputs = 1
num_of_processed_files = 34

label_index = 9

total_list = pd.DataFrame()
for i in tqdm(range(num_of_outputs)):
    root_path = "../ex6/out/"
    each_list = pd.DataFrame()
    for file_index in tqdm(range(0, num_of_processed_files)):
        file_path = root_path + "prediction{}.csv".format(file_index)
        file_df = pd.read_csv(file_path)
        file_df = file_df.iloc[:, 1:]
        file_df.sort_values(by=['0.1'], inplace=True, ascending=False)
        file_df = file_df.iloc[0:3000, :]
        each_list = pd.concat((each_list, file_df), axis=0)
    each_list.sort_values(by=['0.1'], inplace=True, ascending=False)
    each_list = each_list.iloc[0:20000, :]
    total_list = pd.concat((total_list, each_list), axis=0)


total_list.sort_values(by=['0.1'], inplace=True, ascending=False)
total_list = total_list.iloc[0:5000, :]
total_list.to_excel("../ex6/out/final.xlsx")
