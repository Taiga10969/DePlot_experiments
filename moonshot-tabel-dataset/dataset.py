import os
import csv
import json

from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        datapath = os.path.join(dataset_path, 'annotation_files.json')

        with open(datapath, 'r') as f:
            load_json = json.load(f)['annotations']
        
        self.datalist = [item for item in load_json if item['category'] == 0 ] # chart図(cls_id=0)に対して行う

        self.Transforms = transforms.ToTensor()
        
    def __getitem__(self, index):
        data = self.datalist[index]

        #conf = data['conf']
        cap_path = data['cap_path']
        img_path = data['img_path']
        txt_path = data['txt_path']
        #category = data['category']


        cap_path = os.path.join(self.dataset_path, cap_path)
        txt_path = os.path.join(self.dataset_path, txt_path)

        with open(cap_path, 'r') as f:
            cap_text = f.read()
        
        #print(txt_path)
        try:
            with open(txt_path, 'r') as f:
                try:
                    abstract = json.load(f)['Abstract']
                except KeyError:
                    # KeyErrorが発生した場合、例外処理で 'none' を代入する
                    abstract = 'none'
                except json.decoder.JSONDecodeError:
                    # JSONDecodeErrorが発生した場合、例外処理で 'none' を代入する
                    abstract = 'none'
        except FileNotFoundError:
            # ファイルが見つからない場合、例外処理で 'none' を代入する
            abstract = 'none'

        image = self.Transforms(Image.open(os.path.join(self.dataset_path, img_path)))
        
        return image, img_path, cap_text, abstract

    def __len__(self):
        return len(self.datalist)


if __name__=='__main__':

    import config
    config = config.Config()


    dataset = ImageDataset(dataset_path=config.dataset_dir_pth)

    image, img_path, cap_text, abstract =  dataset[8728]

    print('image : ', image)
    print('cap_text : ', cap_text)
    print('img_path : ', img_path)
    print('abstract : ', abstract)
