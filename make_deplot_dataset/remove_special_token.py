import pandas as pd

# 元のCSVファイルのパス
input_file = '/taiga/Datasets/cv_ml_figures/deplot_text_dataset_train.csv'

# 新しいCSVファイルのパス
output_file = '/taiga/Datasets/cv_ml_figures/deplot_text_data_train.csv'

# CSVファイルの読み込み
df = pd.read_csv(input_file, header=None)

# 3列目の文字列データに含まれる'<pad>'を削除
df[2] = df[2].str.replace('<pad>', '')
df[2] = df[2].str.replace('</s>', '')
df[2] = df[2].str.replace('<s>', '')
df[2] = df[2].str.replace('<0x0A>', '')
df[2] = df[2].str.replace('<0x85>', '')
df[2] = df[2].str.replace('<0xE2>', '')
df[2] = df[2].str.replace('<0x82>', '')
df[2] = df[2].str.replace('<0xE2>', '')
df[2] = df[2].str.replace('<0x82>', '')
df[2] = df[2].str.replace('<0x84>', '')
df[2] = df[2].str.replace('<0x93> ', '')
df[2] = df[2].str.replace('<0x86>', '')
df[2] = df[2].str.replace('<0x90>', '')


# '<pad>'を削除したCSVファイルの保存
df.to_csv(output_file, header=False, index=False)
