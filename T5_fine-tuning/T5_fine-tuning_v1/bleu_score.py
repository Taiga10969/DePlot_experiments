from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

import dataset
import configs
import utils


# 乱数シードを固定
utils.setup_torch_seed()


## GPU 使用確認
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.device_count() == 0:
    print("Use 1 GPU")
else:
    print(f"Use {torch.cuda.device_count()} GPUs")

pth = '/taiga/experiment/T5_su/record/best_val_loss.pth'

model_name = 't5-small'
config = configs.Config()

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 学習済みパラメータの読み込み
model.load_state_dict(torch.load(pth, map_location=torch.device(device)))

model = nn.DataParallel(model)
model.to(device)

test_dataset = dataset.DePlotDataset(csv_path=config.dataset_TEST_csv_file_path,
                                     max_source_length=config.max_source_length,
                                     max_target_length=config.max_target_length,
                                     change_id=False,
                                     tokenizer=tokenizer)

bleu_scores = []  # BLEUスコアを保存するリスト

for i in tqdm(range(len(test_dataset))):
    img_pth, caption, deplot_text = test_dataset[i]
    #print(deplot_text)

    deplot_text = [deplot_text]

    encoding = tokenizer(deplot_text, padding='longest', max_length=config.max_target_length, truncation=True,
                         return_tensors='pt')
    input_ids, _ = encoding.input_ids, encoding.attention_mask

    input_ids = input_ids.to(device)

    output = model.module.generate(input_ids=input_ids, max_new_tokens=128)
    output_text = tokenizer.decode(output[0])
    #print("output_text:", output_text)
    #print("label:", caption)

    # BLEUスコアを計算する
    bleu_score_list = utils.calculate_bleu_score(caption, output_text)
    bleu_scores.append(bleu_score_list)

# BLEUスコアの平均を計算する
avg_bleu_scores = torch.tensor(bleu_scores).mean(dim=0).tolist()

print("平均BLEUスコア(1-gram):", avg_bleu_scores[0])
print("平均BLEUスコア(2-gram):", avg_bleu_scores[1])
print("平均BLEUスコア(3-gram):", avg_bleu_scores[2])
print("平均BLEUスコア(4-gram):", avg_bleu_scores[3])
