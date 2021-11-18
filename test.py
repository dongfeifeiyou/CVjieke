import os
import torch as t
from data.dataset_gender import Gender
from config import opt
from torch.utils.data import DataLoader
import models
from torchnet import meter
from torch.autograd import Variable
from utils import Visualizer
import torch.nn as nn

import pandas as pd
import time

if __name__ == '__main__':
    json_file = 'data.json'  # 训练集验证集测试集划分json文件
    load_mode_path = 'checkpoints/11131221_ResNet18/model.pth'
    test_dataset = Gender(opt.train_data_root, json_file,  mode='test')
    test_dataloader = DataLoader(test_dataset,
                             batch_size=opt.batch_size,
                             shuffle=True,
                             num_workers=opt.num_workers)
    print(opt.model)
    model = getattr(models, opt.model)()
    model.load(load_mode_path)
  #  if opt.use_gpu: model.cuda()
