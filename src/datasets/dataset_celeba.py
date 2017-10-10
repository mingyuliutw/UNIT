"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
from __future__ import print_function
import cv2
import torch
import os
import numpy as np
import torch.utils.data as data

class dataset_celeba(data.Dataset):
  def __init__(self, specs):
    self.root = specs['root']
    self.folder = specs['folder']
    self.list_name = specs['list_name']
    self.image_size = specs['crop_image_size']
    self.random_crop = True
    self.random_mirror = True
    list_fullpath = os.path.join(self.root, self.list_name)
    with open(list_fullpath) as f:
      content = f.readlines()
    self.images = [os.path.join(self.root, self.folder, x.strip().split(' ')[0]) for x in content]
    np.random.shuffle(self.images)
    self.dataset_size = len(self.images)

  def __getitem__(self, index):
    crop_img = self._load_one_image(self.images[index])
    raw_data = crop_img.transpose((2, 0, 1))  # convert to HWC
    data = ((torch.FloatTensor(raw_data)/255.0)-0.5)*2
    return data

  def _load_one_image(self, img_name, test=False):
    img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    img = np.float32(img)
    if test == True:
      x_offset = np.int((w - self.image_size) / 2)
      y_offset = np.int((h - self.image_size) / 2)
    else:
      if np.random.rand(1) > 0.5:
        img = cv2.flip(img, 1)
      x_offset = np.int32(np.random.randint(0, w - self.image_size + 1, 1))[0]
      y_offset = np.int32(np.random.randint(0, h - self.image_size + 1, 1))[0]
    crop_img = img[y_offset:(y_offset + self.image_size), x_offset:(x_offset + self.image_size), :]
    return crop_img

  def __len__(self):
    return self.dataset_size
