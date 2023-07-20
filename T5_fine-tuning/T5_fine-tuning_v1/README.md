# T5_fine-tuning_v1

T5にDePlotが出力したテーブルデータを入力し，”text-to-text”でキャプションを生成する．<br>
<br>
### ●```dataset.py```<br>
moonshot-table-datasetで作成したcsvファイルを読み込み，画像のパス，DePlotが出力したテーブルデータ，キャプション（教師ラベル）を取得可能．<br>
<br>
### ●```config.py```<br>
設定ファイル．<br>
<br>
### ●```main.py```<br>
学習プログラム<br>
<br>
### ●```make_loss_graph.py```<br>
main.py を動作した際に作成される　[epoch], [train_loss], [valid_loss] を含む .csv ファイルを読み込み，学習時の損失値の推移を可視化する．<br>
<br>
### ●```bleu_score.py```<br>
全てのテストデータに対して，Bleuスコアを計算し，テストデータの平均BLEUスコアを計算する．また，１つのデータに対する最大のBLEUスコアとその際のデータのインデックスを取得できる．また，各テストデータに対するBLEUスコアの計算結果を作成する．<br>（test_bleu_record.csv）<br>
<br>
### ●```KDE.py```<br>
n-gramにおけるBLEUスコアの各n値の際におけるBLEUスコアの分布を可視化する． import seaborn を利用して，カーネル密度推定を行い，分布を可視化する．<br>
<br>
### ●```show_lr_rate.py```<br>
学習率スケジューラによる学習率の推移を可視化する．<br>
<br>


