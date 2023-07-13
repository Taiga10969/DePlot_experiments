# make_dplot_dataset
・足立先輩提供の"cv_ml_figure"データセットに対して，チャート図のみを抽出し，またそのチャート図に対してDePlotを通してデレンダリングしたテーブルデータ（テキスト形式）のデータを抽出したコード．

　model = nn.DataParallel(Pix2StructForConditionalGeneration.from_pretrained('google/deplot').cuda())
　で，データパラレルのコードにはなっているが，実際には一つのGPUしか動かない状態．（エラーは吐かない）

・作成されたcsvファイルは，[image_pth], [caption], [output_deplot] の３つの要素を含んでいる．

・データの処理範囲を複数に分けて実行し，時間短縮を図った．.scv ファイルの結合コードは ```cat_csv.py``` です．

## 補足
データセット：cv_ml_figures.tar.gz
“cv_ml_figures.zip”の中からDino v2を用いてカテゴリ分けを行い，チャートデータを取得可能にするために図形タイプのラベル付けを行った．
”filter_cap_with_cat.json”が図形タイプのラベルを含んだファイルになる．
