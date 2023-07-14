import pandas as pd

def remove_empty_rows(csv_file, new_file):
    # CSVファイルを読み込む
    df = pd.read_csv(csv_file)
    
    # 空の要素が含まれる行を削除する
    df.dropna(inplace=True)
    
    # 新しいCSVファイルとして保存する
    df.to_csv(new_file, index=False)


if __name__=='__main__':
    csv_file = '/taiga/Datasets/cv_ml_figures/deplot_text_dataset_test.csv'  # 対象のCSVファイル名
    new_file = 'deplot_text_dataset_test_new.csv'
    remove_empty_rows(csv_file, new_file)
