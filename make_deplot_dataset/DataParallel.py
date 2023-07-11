import os
import csv
import json
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

class ImageDataset(Dataset):
    def __init__(self, path, is_train=True):
        self.path = path
        datapath = os.path.join(path, 'filter_cap_with_cat.json')
        with open(datapath, 'r') as f:
            load_json = json.load(f)['annotations']
        
        if is_train:
            self.datalist = [item for item in load_json if item['cls_id'] == 0 and 'train' in item['image_id']]
        else:
            self.datalist = [item for item in load_json if item['cls_id'] == 0 and 'test' in item['image_id']]
        self.T = transforms.ToTensor()
        
    def __getitem__(self, index):
        data = self.datalist[index]
        image_id = data['image_id']
        caption = data['caption']
        image = self.T(
            Image.open(os.path.join(self.path, image_id)))
        return image, image_id, caption

    def __len__(self):
        return len(self.datalist)

class DePlot(nn.Module):
    def __init__(self):
        super(DePlot, self).__init__()
        self.net = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
        
    def forward(self, inputs, max_new_tokens=512):
        return self.net.generate(**inputs, max_new_tokens=max_new_tokens)
        
#def main(model, processor, dataloader):
#    model.eval()
    

if __name__ == '__main__':
    path = '/home/workspace/Datasets/cv_ml_figures'
    output_filename = './train_img_cap_deplot.csv'
    batch_size = 64
    
    dataset = ImageDataset(path, is_train=False)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            pin_memory=True, 
                            num_workers=os.cpu_count())

    model = nn.DataParallel(DePlot().cuda())
    processor = Pix2StructProcessor.from_pretrained('google/deplot')

    for img, image_id, caption in tqdm(dataloader, desc="Processing", unit="batch"):

        img = img.cuda()

        inputs = processor(images=img, text="Generate underlying data table of the figure below:", return_tensors="pt")
        inputs = {key: value.cuda() for key, value in inputs.items()}


        predictions = model(inputs, max_new_tokens=512)

        texts = processor.batch_decode(predictions)

    print("処理が完了しました")
