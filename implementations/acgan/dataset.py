import os 
import sys
import torch
import torchvision.transforms as tf
import pandas
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class Datasets(Dataset):
    
    def __init__(self, img_dir, opt):
        self.img_dir=img_dir
#        self.csv_file=csv_file
        self.opt = opt
        self.csv_file = opt.csv_file
        self.cclass = opt.cclass
        self.item = {}
        self.imglist = os.listdir(img_dir)
        self.count = len(self.imglist)

    
    def __getitem__(self,idx):
#        print(idx)
        img_name = self.imglist[idx]
        imgpath = os.path.join(self.img_dir , img_name)
        img = Image.open(imgpath)
        transform1 = tf.Compose([
                tf.Resize(self.opt.img_size),
                tf.CenterCrop(self.opt.img_size),
                tf.ToTensor(),
                tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        img = transform1(img)
        
        df = pandas.read_csv(self.csv_file)
        imgnum = int(img_name.split('.')[0])
        labels = df[self.cclass][imgnum]
        
#        print(img)
#        print(self.cclass , ' == ' , labels)
        
        return img, labels
        
    def __len__(self):
        return self.count









'''data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))

class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split(' ')[0]) for line in lines]
            
            self.img_label = [line.strip().split(' ')[-1] for line in lines]
            print(self.img_label)
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label
        
image_datasets = {x: customData(img_path='../../data/image',
                                    txt_path=("../../data/txtfile/train.txt"),
                                    data_transforms=data_transforms,
                                    dataset=x)for x in ['train', 'val']}'''