import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class VOCAugDataSet(Dataset):
    def __init__(self, dataset_path='/home/houyuenan/ApolloScapes/list', data_list='train', transform=None):

        with open(os.path.join(dataset_path, data_list + '.txt')) as f:
            self.img_list = []
            self.label_list = []
            for line in f:
                self.img_list.append(line.strip().split(" ")[0])
                self.label_list.append(line.strip().split(" ")[1])

        self.img_path = dataset_path
        self.gt_path = dataset_path
        self.transform = transform
        self.is_testing = data_list == 'test_img' # 'val' # 'val_save'

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_path, self.img_list[idx])).astype(np.float32)
        label = cv2.imread(os.path.join(self.gt_path, self.label_list[idx]), cv2.IMREAD_UNCHANGED)
        image = image[1700:, :, :]
        label = label[1700:, :, :]
        label = label.squeeze()
        if self.transform:
            image, label = self.transform((image, label))
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
            label = torch.from_numpy(label).contiguous().long()
        if self.is_testing:
            return image, label, self.img_list[idx]
        else:
            return image, label
