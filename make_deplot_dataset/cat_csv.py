import pandas as pd
import csv

file_dir = '/taiga/Datasets/cv_ml_figures/'

# 結合するCSVファイルのリスト
file_list = ['train_img_cap_deplot.csv', 'train1_img_cap_deplot.csv', 'train2_img_cap_deplot.csv', 'train3_img_cap_deplot.csv']

# 出力ファイル名
output_file = 'train_deplot_text_dataset.csv'

# 結合したいカラム名を指定（必要に応じて変更してください）
common_columns = ['column1', 'column2', 'column3']

# 結合結果を書き込むためのファイルを開く
with open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    
    # 各ファイルを順に読み込んで結合
    for file_name in file_list:
        with open(file_dir+file_name, 'r') as infile:
            reader = csv.reader(infile)
            
            # 結合元のファイルの行を結合結果に追加
            for row in reader:
                writer.writerow(row)
