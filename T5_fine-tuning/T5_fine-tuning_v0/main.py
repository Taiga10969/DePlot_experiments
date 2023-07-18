from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm


import configs
import dataset
import utils

# 乱数シードを固定
utils.setup_torch_seed()


## GPU 使用確認
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.device_count()==0:
    print("Use 1 GPU")
else :
    print(f"Use {torch.cuda.device_count()} GPUs")



model_name = 't5-small'

config = configs.Config()

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)



train_dataset = dataset.DePlotDataset(csv_path=config.dataset_TRAIN_csv_file_path, 
                                      max_source_length=config.max_source_length, 
                                      max_target_length=config.max_target_length, 
                                      change_id=False, 
                                      tokenizer=tokenizer)

test_dataset = dataset.DePlotDataset(csv_path=config.dataset_TEST_csv_file_path, 
                                      max_source_length=config.max_source_length, 
                                      max_target_length=config.max_target_length, 
                                      change_id=False, 
                                      tokenizer=tokenizer)


train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

model = nn.DataParallel(model)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.999), 
                            eps=1e-6, weight_decay=config.weight_decay)
lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.t0)



utils.training(model=model, 
               tokenizer=tokenizer, 
               train_loader=train_loader, 
               test_loader=test_loader, 
               optimizer=optimizer, 
               scheduler=lr_scheduler, 
               epoch=config.epoch, 
               max_source_length=config.max_source_length, 
               max_target_length=config.max_target_length,
               record_dir=config.record_dir,  
               device=device)
