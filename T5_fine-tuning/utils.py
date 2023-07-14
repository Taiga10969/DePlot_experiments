import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt

def setup_torch_seed(seed=1):
    # pytorchに関連する乱数シードの固定を行う．
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# 乱数シードを固定
setup_torch_seed()


def training(model, tokenizer, train_loader, test_loader, optimizer, scheduler, epoch, max_source_length, max_target_length, record_dir, device):

    for epoch in range(1, epoch+1, 1):
        with tqdm(train_loader) as pbar:

            pbar.set_description(f'[train epoch : {epoch}]')
            # 学習モードに設定
            model.train()
            sum_loss = 0.0
            sum_num=0

            for _, caption, deplot_text in pbar:

                #print(image_ids)
                sum_num+=1

                caption = tokenizer(caption, padding='max_length', max_length=max_target_length, truncation=True, return_tensors='pt')
                deplot_text = tokenizer(deplot_text, padding='max_length', max_length=max_source_length, truncation=True, return_tensors='pt')

                caption_ids = caption['input_ids'].squeeze().to(device)
                caption_ids[caption_ids == tokenizer.pad_token_id] = -100


                deplot_text_ids = deplot_text['input_ids'].squeeze().to(device)

                attention_mask = deplot_text['attention_mask'].squeeze().to(device)

                outputs = model(input_ids=deplot_text_ids, attention_mask=attention_mask, labels=caption_ids)
                
                optimizer.zero_grad()
                
                loss = outputs.loss.mean()

                sum_loss += loss.item()

                #print(loss)
                
                loss.backward()
                optimizer.step()

                pbar.set_postfix(OrderedDict(loss=loss.item(), ave_loss=sum_loss/sum_num, lr = optimizer.param_groups[0]['lr']))
            
            train_loss=sum_loss/sum_num
            scheduler.step()
        
        with tqdm(test_loader) as pbar:

            pbar.set_description(f'[validation : {epoch}]')
            # 学習モードに設定
            model.eval()
            sum_val_loss = 0.0
            sum_num=0

            for _, caption, deplot_text in pbar:

                sum_num += 1

                caption = tokenizer(caption, padding='max_length', max_length=max_target_length, truncation=True, return_tensors='pt')
                deplot_text = tokenizer(deplot_text, padding='max_length', max_length=max_source_length, truncation=True, return_tensors='pt')

                caption_ids = caption['input_ids'].squeeze().to(device)
                caption_ids[caption_ids == tokenizer.pad_token_id] = -100


                deplot_text_ids = deplot_text['input_ids'].squeeze().to(device)

                attention_mask = deplot_text['attention_mask'].squeeze().to(device)

                outputs = model(input_ids=deplot_text_ids, attention_mask=attention_mask, labels=caption_ids)
                
                optimizer.zero_grad()
                
                loss = outputs.loss.mean()

                sum_val_loss += loss.item()

                pbar.set_postfix(OrderedDict(loss=loss.item(), ave_loss=sum_val_loss/sum_num))

            val_loss=sum_loss/sum_num
        
        torch.save(model.state_dict(), record_dir+"/epoch_"+str(epoch)+".pth")

        with open(record_dir+'/loss_record.csv', 'a') as f:
            print(f'{epoch}, {train_loss}, {val_loss}', file=f)


def plot_learning_rate(optimizer, lr_scheduler, num_epochs):
    # 学習率を記録するためのリスト
    lr_values = []

    # モデルのトレーニングループの中で
    for epoch in range(num_epochs):
        # 学習率を取得
        lr = optimizer.param_groups[0]['lr']
        # 学習率を記録
        lr_values.append(lr)

        # 以下、トレーニングステップなどのコード

        # 学習率の更新
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # 学習率の曲線をプロット
    plt.plot(lr_values)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.savefig('learning_rate_scheduler.svg')
    plt.savefig('learning_rate_scheduler.png')
    plt.close()
