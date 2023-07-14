from torch.utils.data import Dataset

import pandas as pd
import configs

class  DePlotDataset(Dataset):
    def __init__(self, csv_path, max_source_length, max_target_length, change_id=False, tokenizer=None):
        self.dirpath = csv_path

        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.change_id = change_id

    
    def __getitem__(self, index):
        img_pth = self.data.iloc[index, 0]
        caption = self.data.iloc[index, 1]
        deplot_text = self.data.iloc[index, 2]

        if self.change_id and self.tokenizer is not None:
            caption = self.tokenizer(caption, padding='max_length', max_length=self.max_target_length, truncation=True, return_tensors='pt')
            deplot_text = self.tokenizer(deplot_text, padding='max_length', max_length=self.max_source_length, truncation=True, return_tensors='pt')

        return img_pth, caption, deplot_text
    

    def __len__(self):
        return  len(self.data)


if __name__=='__main__':
    from transformers import T5Tokenizer
    config = configs.Config()
    
    model_name = 't5-small'
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # exsample : TRAINデータ，ID化しない（tokenizerも渡さない）
    dataset = DePlotDataset(csv_path=config.dataset_TRAIN_csv_file_path, 
                            max_source_length=config.max_source_length, 
                            max_target_length=config.max_target_length, 
                            change_id=False, 
                            tokenizer=None)
    print('len(dataset) : ', len(dataset))
    img_pth, caption, deplot_text = dataset[3]
    print('img_pth : ', img_pth)
    print('caption : ', caption)
    print('deplot_text : ', deplot_text)

    # exsample : TESTデータ，ID化
    dataset = DePlotDataset(csv_path=config.dataset_TRAIN_csv_file_path, 
                            max_source_length=config.max_source_length, 
                            max_target_length=config.max_target_length, 
                            change_id=True, 
                            tokenizer=tokenizer)
    
    img_pth, caption, deplot_text = dataset[17]
    print('img_pth : ', img_pth)
    print('caption : ', caption['input_ids'])
    print('deplot_text : ', deplot_text['input_ids'])

    ids = caption['input_ids'].numpy()
    tokens = tokenizer.convert_ids_to_tokens(ids[0])
    string = tokenizer.convert_tokens_to_string(tokens)
    print('caption_convert_string : ', string)

    ids = deplot_text['input_ids'].numpy()
    tokens = tokenizer.convert_ids_to_tokens(ids[0])
    string = tokenizer.convert_tokens_to_string(tokens)
    print('deplot_text_convert_string : ', string)
