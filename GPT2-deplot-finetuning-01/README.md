# GPT2-deplot-finetuning-01
論文図を deplot を用いてデレンダリングし，得られたテキスト形式のテーブルデータからキャプションを予測するように GPT2 をファインチューニングする．

## setup
dataset : cv_ml_figure.tar.gz (足立先輩提供)<br>
          を deplot に入力して，得られた出力を含めた，[image_pth]，[caption]．[output_deplot] の .csv ファイルを作成し，これを利用する．<br>
          ※ .csv ファイルの作成コード : https://github.com/Taiga10969/experiments/tree/main/make_deplot_dataset
 
