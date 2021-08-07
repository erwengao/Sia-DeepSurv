import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np


def My_loader(path):
    totalpath, name_ext = os.path.split(path)
    id = os.path.basename(totalpath)
    data = h5py.File(path, 'r')
    img = np.array(data['image'][:])
    img = img[None, :, :, :]
    img = np.concatenate((img, img, img), axis=0)
    X = torch.from_numpy(img)
    return X, id


def make_dataset(dir):  ##dir 为h5总文件夹,下面包含所有病人的文件夹,每个文件夹下包含带标签的h5文件
    images = []
    classes = set()
    for patients_dir in os.listdir(dir):
        for root, _, h5files in os.walk(os.path.join(dir, patients_dir)):
            e1 = None
            e2 = None
            y1 = None
            y2 = None
            path1 = None
            path2 = None
            for file in h5files:
                if os.path.splitext(file)[-2] == '1':
                    path1 = os.path.join(root, file)
                    e1 = h5py.File(path1, 'r')['label'][:][0]
                    y1 = h5py.File(path1, 'r')['time'][:][0]
                if os.path.splitext(file)[-2] == '2':
                    path2 = os.path.join(root, file)
                    e2 = h5py.File(path2, 'r')['label'][:][0]
                    y2 = h5py.File(path2, 'r')['time'][:][0]
            if e1 != e2 or y1 != y2:
                raise ValueError('Labels do not match before and after treatment')
            else:
                classes.add(e1)
                item = (path1, path2, e1, y1)
                images.append(item)
    classes = list(classes)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return images, classes, class_to_idx


class SurvFolder(Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=My_loader):
        imgs, classes, class_to_idx = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ".h5"))
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path1, path2, e, y = self.imgs[index]
        img1, id1 = self.loader(path1)
        img2, id2 = self.loader(path2)

        if id1 != id2:
            raise ValueError('The ID before and after does not match')
        else:
            id = id1
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        if self.target_transform is not None:
            e = self.target_transform(e)
        return img1, img2, e, y, id

    def __len__(self):
        return len(self.imgs)
