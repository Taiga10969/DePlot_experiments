import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def kernel_density_estimation(csv_file, column_index):
    # CSVファイルを読み込む
    df = pd.read_csv(csv_file)

    # 指定された列のデータを取得
    data = df.iloc[:, column_index]

    # カーネル密度推定を行う
    sns.set(style="whitegrid")
    sns.kdeplot(data, fill=True)  # 'shade'ではなく'fill'を使用

    # プロットのカスタマイズ（必要に応じて変更）
    #plt.title(f"bleu score  {column_index}")
    plt.xlabel(f"bleu score ({column_index}-gram)")
    plt.ylabel("Density")

    # プロットを表示
    plt.savefig(f'KDE({column_index}-gram).png')
    plt.savefig(f'KDE({column_index}-gram).svg')
    plt.savefig(f'KDE({column_index}-gram).pdf')
    plt.close()

if __name__ == "__main__":
    # ファイル名と列インデックスを指定してプログラムを実行
    csv_file = "/taiga/experiment/T5_su/record/test_bleu_record.csv"  # あなたのCSVファイル名をここに入力してください
    column_index = 4  # カーネル密度推定を行う列のインデックスをここに入力してください
    kernel_density_estimation(csv_file, column_index)
