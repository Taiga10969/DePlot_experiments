# T5_fine-tuning_v1

T5にDePlotが出力したテーブルデータを入力し，”text-to-text”でキャプションを生成する．<br>
<br>
```dataset.py```<br>
moonshot-table-datasetで作成したcsvファイルを読み込み，画像のパス，DePlotが出力したテーブルデータ，キャプション（教師ラベル）を取得可能．<br>
<br>
```bleu_score.py```<br>
全てのテストデータに対して，Bleuスコアを計算し，テストデータの平均BLEUスコアを計算する．また，１つのデータに対する最大のBLEUスコアとその際のデータのインデックスを取得できる．また，各テストデータに対するBLEUスコアの計算結果を作成する．（test_bleu_record.csv）<br>
<br>
```
<br>
```KDE.py```
<br>


