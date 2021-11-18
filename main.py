import os
import torch as t
from data.dataset_gender import Gender
from config import opt
from torch.utils.data import DataLoader
import models
from torchnet import meter
from torch.autograd import Variable
import torch.nn as nn

import pandas as pd
import time

def train():

    path_checkpoints = 'checkpoints'  # 模型保存目录
    json_file = 'data.json'  # 训练集验证集测试集划分json文件
    if not os.path.exists(path_checkpoints):
        os.mkdir(path_checkpoints)
    now_time = time.strftime("%m%d%H%M", time.localtime())
    path_save = path_checkpoints + '/' + now_time + '_' + opt.model
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    path_model_save = path_save + '/' + 'model.pth'
    path_loss_batch = path_save + '/' + 'loss_batch.csv'
    path_acc_epoch = path_save + '/' + 'acc_epoch.csv'

    # step1 模型
    print(opt.model)
    model = getattr(models, opt.model)()
    if opt.load_mode_path: # 加载预训练的模型的路径
        model.load(opt.load_mode_path)
    if opt.use_gpu: model.cuda()
    # model = nn.DataParallel(model, device_ids=[0,1,2,3])  # 指定多块gpu运行程序

    # step2 数据
    train_dataset = Gender(opt.train_data_root, json_file,  mode='train')#dataset类，决定数据从哪读取及如何读取，进行训练，定义训练数据和label集合train_dataset
    val_dataset = Gender(opt.train_data_root, json_file, mode='val')
    # 随机将数据分批，一批有opt.batch_size个，并行处理，打开opt.num_workers个进程
    train_dataloader = DataLoader(train_dataset,
                             batch_size=opt.batch_size, # 批大小
                             shuffle=True, # 每个epoch是否乱序
                             num_workers=opt.num_workers) # 是否多进程读取数据
    val_dataloader = DataLoader(val_dataset,
                                batch_size=opt.batch_size,
                                shuffle=True,
                                num_workers=opt.num_workers)

    # step3 目标函数和优化器
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    # optimizer = t.optim.SGD(model.parameters(),
    #                          lr=lr,
    #                          weight_decay=opt.weight_decay)
    optimizer = t.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    scheduler = t.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.max_epoch, T_mult=2, eta_min=0,
                                                                     last_epoch=-1)

    # step4 统计指标：平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter() # 能够计算所有数的平均值和标准差，用来统计一个epoch中损失的平均值
    confusion_matrix = meter.ConfusionMeter(2) # 用来统计分类问题中的分类情况，是一个比准确率更详细的统计指标

    best_acc = 0
    early_stop_count = 0
    # 保存loss
    loss_batch = pd.DataFrame(columns=['time', 'epoch', 'num_iter', 'loss'])
    acc_epoch = pd.DataFrame(columns=['time', 'epoch', 'acc_train', 'acc_val', 'lr', 'loss_train'])


    # 训练
    for epoch in range(opt.max_epoch): # 迭代次数
        new_lr = scheduler.get_last_lr()[0]
        loss_meter.reset()
        confusion_matrix.reset()
        # 显示得到的数据
        for ii, (data, label) in enumerate(train_dataloader):
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # 更新统计指标及可视化
            loss_meter.add(loss.item())
            confusion_matrix.add(score.data, target.data)

            if ii%opt.print_freq == opt.print_freq-1:
                now_time = time.strftime("%m-%d_%H:%M", time.localtime())
                print('{}, epoch:{}, batch:{}/{}, loss:{}'.format(now_time, epoch, ii+1, len(train_dataloader), loss_meter.value()[0]))
                num_iter = len(train_dataloader) * epoch + ii
                temp = pd.Series([now_time, epoch, num_iter, loss_meter.value()[0]], index=['time', 'epoch', 'num_iter', 'loss'])
                loss_batch = loss_batch.append(temp, ignore_index=True)
                loss_batch.to_csv(path_loss_batch, index=False)

        # 计算训练集和验证集上的指标及可视化
        scheduler.step()
        cm_value = confusion_matrix.value()
        train_accuracy = 100 * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
        # validate and visualize，计算验证集上的指标及可视化
        val_cm, val_accuracy = val(model,val_dataloader)
        now_time = time.strftime("%m-%d_%H:%M", time.localtime())
        print("{time}, epoch:{epoch},train_accuracy:{train_accuracy},val_accuracy:{val_accuracy},lr:{lr},loss{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            time=now_time,
            epoch=epoch,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            lr=new_lr,
            loss=loss_meter.value()[0],
            train_cm=str(confusion_matrix.value()),
            val_cm=str(val_cm.value())
        ))

        temp = pd.Series([now_time, epoch, train_accuracy, val_accuracy, new_lr, loss_meter.value()[0]],
                         index=['time', 'epoch', 'acc_train', 'acc_val', 'lr', 'loss_train'])
        acc_epoch = acc_epoch.append(temp, ignore_index=True)
        acc_epoch.to_csv(path_acc_epoch, index=False)

        early_stop_count += 1
        # 保存准确率最高的模型
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            now_checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc
            }
            t.save(now_checkpoint, path_model_save)
            print("=> saved best model")
            early_stop_count = 0

        # early_stop 设置
        if opt.early_stop != None:
            if early_stop_count >= opt.early_stop:
                print("=> early stopping")
                print('Now epoch is {}. best_acc hans\'t been updated for {} epoch.'.format(epoch, opt.early_stop))
                break

def val(model, dataloader):
    """
        计算模型在验证集上的准确率等信息
    """
    # 把模型设为验证模式
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data
        with t.no_grad():
            val_input = Variable(input)
            # val_label = Variable(label.long())
        if opt.use_gpu:
            val_input = val_input.cuda()
            # val_label = val_label.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data, label.long())
    # 把模型恢复为训练模式，要养成习惯，不实用验证模式后要将其调整回来
    model.train()

    # 计算准确率
    cm_value = confusion_matrix.value()
    accuracy = 100 * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy

def test():
    json_file = 'data.json'  # 训练集验证集测试集划分json文件
    load_mode_path = 'checkpoints/11122326_AlexNet/model.pth'  # 模型路径
    # load_mode_path = 'checkpoints/11131221_ResNet18/model.pth'
    test_dataset = Gender(opt.test_data_root, json_file, mode='test')
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=True,
                                 num_workers=opt.num_workers)
    print(opt.model)
    model = getattr(models, opt.model)()
    model = nn.DataParallel(model)
    model_CKPT = t.load(load_mode_path)
    model.load_state_dict(model_CKPT['model'])
    print('加载模型完毕，正在进行测试')
    if opt.use_gpu: model.cuda()
    test_cm, test_accuracy = val(model, test_dataloader)
    print('测试准确率为：{}'.format(test_accuracy))
def help():
    pass

if __name__ == '__main__':
    test()  # 训练就改成train()，测试就改成test()



