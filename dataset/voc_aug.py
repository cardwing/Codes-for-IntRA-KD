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
        self.transform_0 = transform[0]
        self.transform_1 = transform[1]
        self.transform_2 = transform[2]
        self.transform_3 = transform[3]
        #self.transform_4 = transform[4]
        self.is_testing = data_list == 'test_img' # 'val'

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_path, self.img_list[idx])).astype(np.float32)
        label = cv2.imread(os.path.join(self.gt_path, self.label_list[idx]), cv2.IMREAD_UNCHANGED)
        image = image[1700:, :, :]
        label = label[1700:, :, :]
        label = label.squeeze()
        image_copy, label_copy = image, label
        if self.transform_0:
            image, label = self.transform_0((image, label))
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
            label = torch.from_numpy(label).contiguous().long()

            image_1, label_1 = self.transform_1((image_copy, label_copy))
            image_1 = torch.from_numpy(image_1).permute(2, 0, 1).contiguous().float()
            label_1 = torch.from_numpy(label_1).contiguous().long()

            image_2, label_2 = self.transform_2((image_copy, label_copy))
            image_2 = torch.from_numpy(image_2).permute(2, 0, 1).contiguous().float()
            label_2 = torch.from_numpy(label_2).contiguous().long()

            image_3, label_3 = self.transform_3((image_copy, label_copy))
            image_3 = torch.from_numpy(image_3).permute(2, 0, 1).contiguous().float()
            label_3 = torch.from_numpy(label_3).contiguous().long()

            '''image_4, label_4 = self.transform_4((image_copy, label_copy))
            image_4 = torch.from_numpy(image_4).permute(2, 0, 1).contiguous().float()
            label_4 = torch.from_numpy(label_4).contiguous().long()'''

        if self.is_testing:
            return image, image_1, image_2, image_3, self.img_list[idx] #, image_4
        else:
            return image, label
