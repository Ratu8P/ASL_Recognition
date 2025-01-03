import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class HandKeypointsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.labels = []
        self.classes = sorted([
            cls_name for cls_name in os.listdir(data_dir)
            if not cls_name.startswith('.') and os.path.isdir(os.path.join(data_dir, cls_name))
        ])
        self.class_to_label = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            for img_name in os.listdir(cls_dir):
                if img_name.endswith(('.npy', '.jpg', '.jpeg', '.png')):
                    self.data.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_label[cls_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        if img_path.endswith('.npy'):
            keypoints = np.load(img_path)
            input_data = torch.tensor(keypoints, dtype=torch.float32)
        else:
            image = cv2.imread(img_path)
            input_data = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        return input_data, label
