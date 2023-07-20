import torch
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

import config
import dataset

import csv

config = config.Config()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count()==0:
    print("Use 1 GPU")
else :
    print(f"Use {torch.cuda.device_count()} GPUs")


dataset = dataset.ImageDataset(dataset_path=config.dataset_dir_pth)
print(len(dataset))


model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot').cuda()
processor = Pix2StructProcessor.from_pretrained('google/deplot')


with open(config.output_file_name, 'w', newline='', encoding='utf8') as output_file:
    writer = csv.writer(output_file)

    for i in range(0, len(dataset):  # rangeを設定することで，擬似的にデータパラレルを行う（あとで作成されたcsvファイルを結合）
        print(i)

        image, img_path, cap_text, abstract = dataset[i]  # 1枚ずつデータを取得
        image = image.unsqueeze(0).to(device)  # バッチ次元を追加してデバイスに送る

        inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        predictions = model.generate(**inputs, max_new_tokens=256)
        deplot_text = processor.batch_decode(predictions)

        writer.writerow([img_path, deplot_text[0], cap_text, abstract])  # リストになっているdeplot_textを文字列に直して書き込み

print("処理が完了しました")
