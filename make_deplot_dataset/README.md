# make_dplot_dataset
・足立先輩提供の"cv_ml_figure"データセットに対して，チャート図のみを抽出し，またそのチャート図に対してDePlotを通してデレンダリングしたテーブルデータ（テキスト形式）のデータを抽出したコード．

　model = nn.DataParallel(Pix2StructForConditionalGeneration.from_pretrained('google/deplot').cuda())
　で，データパラレルのコードにはなっているが，実際には一つのGPUしか動かない状態．（エラーは吐かない）

・作成されたcsvファイルは，[image_pth], [caption], [output_deplot] の３つの要素を含んでいる．

・データの処理範囲を複数に分けて実行し，時間短縮を図った．.scv ファイルの結合コードは ```cat_csv.py``` です．

・また，処理の過程で空のテキストとして，保存されているものを発見したため，全ての要素が空でないかを確認して，空の要素を含む行を削除するコードは ```remove_empty_rows.py ``` です．
　※この処理により，train, test 共に５枚程度削除される．

## 補足
データセット：cv_ml_figures.tar.gz
“cv_ml_figures.zip”の中からDino v2を用いてカテゴリ分けを行い，チャートデータを取得可能にするために図形タイプのラベル付けを行った．
”filter_cap_with_cat.json”が図形タイプのラベルを含んだファイルになる．
