from torch.utils.data import Dataset
from torchvision import transforms
from transformers import ViTImageProcessor

import csv
import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import json
from PIL import Image
from tqdm import tqdm

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor


dataset_pth = '/taiga/Datasets/cv_ml_figures'
input_filename = '/taiga/Datasets/cv_ml_figures/filter_cap_with_cat.json'
output_filename = '/taiga/Datasets/cv_ml_figures/train_img_cap_deplot.csv'
bath_size = 16

class ImageDataset(Dataset):
    def __init__(self, annotations, dataset_pth):
        self.annotations = annotations
        self.dataset_pth = dataset_pth

    def __getitem__(self, index):
        annotation = self.annotations[index]
        image_id = annotation['image_id']
        caption = annotation['caption']
        image = Image.open(self.dataset_pth+'/'+image_id)
        img = transforms.ToTensor()(image)
        return img, image_id, caption

    def __len__(self):
        return len(self.annotations)


# JSONファイルの読み込みとフィルタリング
with open(input_filename, 'r') as input_file:
    data = json.load(input_file)
    annotations = data['annotations']

# trainから始まるもの・cls_id=0 のものに限定 (読み込んだものの最初の100000要素に対して行う)
filtered_annotations = [annotation for annotation in annotations[:100000] if annotation['image_id'].startswith('train') and annotation['cls_id'] == 0]

# 画像データセットを作成
dataset = ImageDataset(filtered_annotations, dataset_pth)

dataloader = DataLoader(dataset, batch_size=bath_size, shuffle=False, pin_memory=True, num_workers=20)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count()==0:
    print("Use 1 GPU")
else :
    print(f"Use {torch.cuda.device_count()} GPUs")

model = nn.DataParallel(Pix2StructForConditionalGeneration.from_pretrained('google/deplot').cuda())
processor = Pix2StructProcessor.from_pretrained('google/deplot')
#nn.DataParallel(model)
#model.to(device)


# CSVファイルに書き込み
with open(output_filename, 'w', newline='', encoding='utf8') as output_file:
    writer = csv.writer(output_file)

    for img, image_id, caption in tqdm(dataloader, desc="Processing", unit="batch"):
        
        img = img.cuda()

        inputs = processor(images=img, text="Generate underlying data table of the figure below:", return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}


        predictions = model.module.generate(**inputs, max_new_tokens=512)

        texts = processor.batch_decode(predictions)

        for image_id, caption, text in zip(image_id, caption, texts):
            writer.writerow([image_id, caption, text])

print("処理が完了しました")

