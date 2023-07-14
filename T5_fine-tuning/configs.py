class Config(object):
    def __init__(self):
        
        # パスの設定
        self.dataset_TRAIN_csv_file_path = '/taiga/Datasets/cv_ml_figures/deplot_text_dataset_train.csv'
        self.dataset_TEST_csv_file_path = '/taiga/Datasets/cv_ml_figures/deplot_text_dataset_test.csv'
        self.record_dir = '/taiga/experiment/T5/record'

        
        self.max_source_length = 512  #sourceの入力文字列の最大列数
        self.max_target_length = 128  #targetの入力文字列の最大列数

        self.batch_size = 128  #バッチサイズ
        self.lr = 5e-4  #学習率
        self.weight_decay = 0.1
        self.t0 = 15
        self.epoch = 100
