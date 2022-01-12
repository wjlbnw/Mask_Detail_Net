import numpy as np
import torch

import cv2.cv2 as cv2
import os

import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from data.data import *
import argparse
import tool.unet_tool as unet_tool
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue

from network.Mask_Detail_Net import *
from dataSetConfig.DibcoDataSet import Dibco2016DataSetIniter
from dataSetConfig.DibcoDataSet import Dibco2017DataSetIniter
from dataSetConfig.DibcoDataSet import Dibco2018DataSetIniter


class Adjust_Image():
    def __init__(self,brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        img = adjust_brightness(img, self.brightness)
        img = adjust_contrast(img, self.contrast)
        img = adjust_saturation(img, self.saturation)
        img = adjust_hue(img, self.hue)
        return img



def list_mean(_list):
    sum = 0
    for a in _list:
        sum += a
    return sum / len(_list)
def TP_FP_FN(out, label, th=0.5):

    n, c, h, w = label.size()
    assert n == 1 and c == 1
    n, c, h, w = out.size()
    # print(out.size())
    assert n ==1 and c == 2

    ret = torch.zeros(label.size(), dtype=torch.float32)
    # ret[out[:, 0:1, :, :] > th] = 1
    ret[out[:, 0:1, :, :] > out[:, 1:2, :, :]] = 1
    # 维度转换
    y_pred = ret.reshape(-1)
    y_true = label.reshape(-1)

    TP = y_true*y_pred
    FP = (1-y_true)*y_pred
    FN = (1-y_pred)*y_true

    iou = torch.sum(TP) / torch.sum(y_true + y_pred - TP)

    return torch.sum(TP).item(), torch.sum(FP).item(), torch.sum(FN).item(), iou.item()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", type=str)

    parser.add_argument("--data_root", default='/home/wujiali/DeepLearning/dataset/DIBCO', type=str)

    parser.add_argument("--infer_root", default='/home/wujiali/DeepLearning/test_out/DIBCO_out', type=str)
    parser.add_argument("--infer_session", default='test_session', type=str)
    parser.add_argument("--cut_size", default=1024, type=int)

    args = parser.parse_args()
    print(vars(args))

    weights_path = args.weight_path
    cut_size = args.cut_size
    infer_root = args.infer_root
    session_name = args.infer_session
    infer_out_path = os.path.join(infer_root, session_name)

    if not os.path.exists(infer_out_path):
        os.makedirs(infer_out_path)

    mask_model = Mask_Net(in_channel=3, out_channel=3)
    seg_model = Asymmetry_Net(in_channel=4)

    print('infer started...')

    weight_path = args.weight_path
    weight_dice = torch.load(weight_path)
    mask_model.load_state_dict(weight_dice['mask_weight'])
    mask_model.eval()
    mask_model.cuda()

    seg_model.load_state_dict(weight_dice['seg_weight'])
    seg_model.eval()
    seg_model.cuda()


    evl_dataset = DIBCOValDataset20211220(train_set_list=[
        Dibco2018DataSetIniter(os.path.join(args.data_root, 'dibco2018'))
    ], transform=transforms.Compose([
        transforms.Normalize(mean=(0.406, 0.456, 0.485),
                             std=(0.225, 0.224, 0.229))
    ]), transform_concat=None)

    evl_data_loader = DataLoader(evl_dataset, shuffle=False, num_workers=1, pin_memory=True)

    with torch.no_grad():

        f_score_list = []
        for iter, (img, gt, img_name) in enumerate(evl_data_loader):
            # bar.next()
            img_name = img_name[0]
            img = img.cuda()

            n, c, h, w = img.size()

            seg_ret = torch.zeros(size=(n,2,h,w), dtype=torch.float32)
            for top in range(0, h, cut_size):
                for left in range(0, w, cut_size):
                    pic_in_window = img[:, :, top:min(top + cut_size, h), left:min(left + cut_size, w)]

                    N, C, window_h, window_w = pic_in_window.size()
                    padding_pic, pad_top, pad_left = unet_tool.pad_img(pic_in_window)
                    mask_out = mask_model(padding_pic)
                    mask_out = mask_out[:, 0:1, :, :]
                    seg_out = seg_model(torch.concat((padding_pic.cuda(), mask_out), dim=1))

                    seg_out = seg_out[:, :, pad_top:pad_top + window_h, pad_left:pad_left + window_w].cpu()
                    seg_ret[:, :, top: top + window_h, left: left + window_w] = seg_out[:, :, :, :]

            gray = torch.zeros((1, 1, h, w), dtype=torch.float32)
            gray[seg_ret[:, 0:1, :, :] > seg_ret[:, 1:2, :, :]] = 1
            gray *= 255.0
            gray = 255 - gray
            gray = gray.reshape(h, w)
            gray = gray.cpu().numpy().astype(np.uint8)

            cv2.imwrite(os.path.join(infer_out_path, '%s_out.bmp' % (img_name.split('.')[0])), gray)

    # bar.finish()
    print('Infer end...')



