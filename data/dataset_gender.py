import os
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from PIL import Image
import json

class Gender(data.Dataset):
    def __init__(self, root, json_file, transforms=None, mode='train'):
        self.mode = mode
        with open(json_file, 'r') as f:
            files = json.load(f)
        # 获取所有图片的地址
        self.imgs = [os.path.join(root, img).replace('\\', '/') for img in files[self.mode]['data']]

        if transforms is None:
            # 数据转换操作，测试验证和训练的数据转换有所区别
            # # 对数据进行归一化
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], # 图像标准化
                                    std=[0.229, 0.224, 0.225])
            # 当是测试集和验证集时进行的操作
            if self.mode=='test' or self.mode=='val':
                self.transforms = T.Compose([
                    T.Resize(224),#重新设定大小
                    T.CenterCrop(224),#从图片中心截取
                    T.ToTensor(),#转成Tensor格式，大小范围为[0,1]
                    normalize #归一化处理,大小范围为[-1,1]
                ])

            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224), #从图片的任何部位随机截取224*224大小的图
                    T.RandomHorizontalFlip(), #随机水平翻转给定的PIL.Image,翻转概率为0.5
                    T.ToTensor(),
                    normalize
                ])


    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        label = int(self.imgs[index].split('/')[-1].split('_')[1])#如果是测试，得到图片路径中的数字标识作为label
        data = Image.open(img_path) # 打开该路径获得数据
        data = self.transforms(data) # 然后对图片数据进行transform
        return data, label # 最后得到统一的图片信息和label信息

    def __len__(self): # 图片数据的大小
        return len(self.imgs)

