# @Time    : 2023/11/9 11:11
# @Author  : wang song
# @File    : pred.py
# @Description :
import glob
import numpy as np
import pandas as pd
import torch
import os
import cv2
from tqdm import tqdm
import torch.nn.functional as F
from dataloaders.datasets.medical import MedicalSegmentDataset
from modeling.unet import UNet
from utils.loss import SegmentationLosses
from utils.metrics import Evaluator
from medpy import metric

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    model = UNet(in_channels=3, n_classes=2, base_c=64)
    # 将网络拷贝到deivce中
    model.to(device=device)
    # 加载模型参数
    # checkpoint = torch.load("/home/temp58/wangsong/pytorch-segmentation/run/medical/unet-resnet/experiment3_10/checkpoint3.pth")
    checkpoint = torch.load("/home/temp58/wangsong/pytorch-segmentation/run/medical/unet-resnet/unet3_10/checkpoint.pth")
    model.load_state_dict(checkpoint['state_dict'])
    # model.load_state_dict(torch.load('best_model.pth', map_location=device))
    # 测试模式
    data_path = '/home/temp58/dataset/seg-CQ500'
    val_dataset = MedicalSegmentDataset(data_path,
                                        train=False,
                                        )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=8,
                                             pin_memory=True,
                                             )
    criterion = SegmentationLosses(cuda='cuda:0').build_loss(mode="ce")
    model.eval()
    evaluator = Evaluator(2)
    evaluator.reset()
    tbar = tqdm(val_loader, desc='\r')
    test_loss = 0.0
    dice_score = 0
    test_total = 0.0
    asd = 0
    count = 0
    dsc = 0
    output_file = "1.xlsx"  # 输出文件路径
    data = {'image': [], 'Dice': []}
    df = pd.DataFrame(data)
    # 新建一个文件夹来保存预测结果
    output_folder = "/home/temp58/wangsong/pytorch-segmentation/cq"
    os.makedirs(output_folder, exist_ok=True)

    # output_file = "results3.txt"  # 输出文件路径
    # with open(output_file, "w") as f:
    for i, sample in enumerate(tbar):
        image, target, filename = sample['image'], sample['label'], sample['filename']
        target = target.to(dtype=torch.long)
        mask_true = F.one_hot(target, model.n_classes).permute(0, 3, 1, 2).float()

        image, target = image.cuda(), target.cuda()
        mask_true = mask_true.cuda()
        with torch.no_grad():
            output = model(image)
        loss = criterion(output, target).cuda()
        test_loss += loss.item()
        tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

        test_total += target.size(0)

        mask_pred = F.one_hot(output.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()

        np.set_printoptions(threshold=np.inf)
        pred = output.cpu().numpy()
        target = target.cpu().numpy()

        # asd = metric.binary.asd(pred, target)
        pred = np.argmax(pred, axis=1)

        # Add batch sample into evaluator
        dsc = metric.binary.dc(pred, target)
        for j, file in enumerate(filename):
            output_filename = os.path.join(output_folder, file)
            cv2.imwrite(output_filename, pred[j] * 255)  # 假设预测结果是单通道图像，乘以255转为灰度图
        # 打印照片名称和Dice值
        print("image: {}, Dice: {:.4f}".format(filename, dsc))
        # 将照片名称和Dice值写入文件
        # line = "image: {}, Dice: {:.4f}\n".format(sample['filename'], dsc)
        # f.write(line)
        # 将照片名称和Dice值添加到DataFrame中
        new_row = {'Filename': filename[0], 'Dice': dsc}
        df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

    # 将DataFrame写入Excel文件
    df.to_excel(output_file, index=False)
