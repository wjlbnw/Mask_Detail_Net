
import numpy as np
import torch
import torchvision.transforms
from torch.utils.data import Dataset
import cv2.cv2 as cv2
import os.path


from dataSetConfig.DibcoDataSet import DibcoDataSetIniter



class DIBCODataset(Dataset):

    def __init__(self,  train_set_list, transform=None, transform_concat=None):

        self.img_name_list = []
        for train_set in train_set_list:
            assert isinstance(train_set, DibcoDataSetIniter)
            self.img_name_list.extend(train_set.data_list)
        self.transform = transform
        self.transform_concat = transform_concat
        self.toTensor = torchvision.transforms.ToTensor()


    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        data_item = self.img_name_list[idx]
        assert os.path.exists(data_item.in_path)
        assert os.path.exists(data_item.gt_path)
        img = self.toTensor(cv2.imread(data_item.in_path))
        gt = self.toTensor(cv2.imread(data_item.gt_path, cv2.IMREAD_GRAYSCALE))

        # h, w = gt.size()
        gt = 1.0 - gt

        if self.transform:
            img = self.transform(img)
        concat = torch.concat((img, gt), dim=0)

        if self.transform_concat:
            concat = self.transform_concat(concat)

        return concat[0:3, :, :], concat[3:4, :, :]


class DoubleNetDIBCODataSet(Dataset):

    def __init__(self, train_set_list, transform=None, transform_concat=None):

        self.img_name_list = []
        for train_set in train_set_list:
            assert isinstance(train_set, DibcoDataSetIniter)
            self.img_name_list.extend(train_set.data_list)
        self.transform = transform
        self.transform_concat = transform_concat
        self.toTensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        data_item = self.img_name_list[idx]
        # print(data_item)
        assert os.path.exists(data_item.in_path)
        assert os.path.exists(data_item.gt_path)
        img_np = cv2.imread(data_item.in_path)
        img = self.toTensor(img_np)
        gt = self.toTensor(cv2.imread(data_item.gt_path, cv2.IMREAD_GRAYSCALE))
        _, mask = cv2.threshold(cv2.cvtColor(img_np,cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = self.toTensor(mask)

        # h, w = gt.size()
        gt = 1.0 - gt
        mask[gt > 0.5] = 0
        if self.transform:
            img = self.transform(img)
        concat = torch.concat((img, gt, mask), dim=0)

        if self.transform_concat:
            concat = self.transform_concat(concat)

        return concat[0:3, :, :], concat[3:4, :, :], concat[4:5, :, :]


class DIBCOValDataset(DIBCODataset):

    def __init__(self,  train_set_list, transform=None, transform_concat=None):
        super(DIBCOValDataset, self).__init__(train_set_list=train_set_list, transform=transform, transform_concat=transform_concat)


    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        data_item = self.img_name_list[idx]
        assert os.path.exists(data_item.in_path)
        assert os.path.exists(data_item.gt_path)
        img = self.toTensor(cv2.imread(data_item.in_path))
        gt = self.toTensor(cv2.imread(data_item.gt_path, cv2.IMREAD_GRAYSCALE))

        # h, w = gt.size()
        gt = 1.0 - gt

        if self.transform:
            img = self.transform(img)
        concat = torch.concat((img, gt), dim=0)

        if self.transform_concat:
            concat = self.transform_concat(concat)

        return concat[0:3, :, :], concat[3:4, :, :], os.path.split(data_item.in_path)[-1]


class DIBCOValDataset20211220(DIBCODataset):

    def __init__(self, train_set_list, transform=None, transform_concat=None, data_aug=None):
        super(DIBCOValDataset20211220, self).__init__(train_set_list=train_set_list, transform=transform,
                                              transform_concat=transform_concat)
        self.data_aug = data_aug

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        data_item = self.img_name_list[idx]
        assert os.path.exists(data_item.in_path)
        assert os.path.exists(data_item.gt_path)
        img = cv2.imread(data_item.in_path)
        if self.data_aug:
            img = self.data_aug(img)
        img = self.toTensor(img)
        gt = self.toTensor(cv2.imread(data_item.gt_path, cv2.IMREAD_GRAYSCALE))


        gt = 1.0 - gt

        if self.transform:
            img = self.transform(img)
        concat = torch.concat((img, gt), dim=0)

        if self.transform_concat:
            concat = self.transform_concat(concat)

        return concat[0:3, :, :], concat[3:4, :, :], os.path.split(data_item.in_path)[-1]

