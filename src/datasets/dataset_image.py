"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
from __future__ import print_function
import os
import numpy as np
import cv2
import torch
import torch.utils.data as data

class dataset_image(data.Dataset):
  def __init__(self, specs):
    self.root = specs['root']
    self.folder = specs['folder']
    self.list_name = specs['list_name']
    self.scale = specs['scale']
    self.crop_image_height = specs['crop_image_height']
    self.crop_image_width = specs['crop_image_width']
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

  def __len__(self):
    return self.dataset_size

  def _load_one_image(self, img_name, test=False):
    img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    if self.scale > 0:
      img = cv2.resize(img,None,fx=self.scale,fy=self.scale)
    img = np.float32(img)
    h, w, c = img.shape
    if test==True:
      x_offset = np.int( (w - self.crop_image_width)/2 )
      y_offset = np.int( (h - self.crop_image_height)/2 )
    else:
      if np.random.rand(1) > 0.5:
        img = cv2.flip(img, 1)
      x_offset = np.int32(np.random.randint(0, w - self.crop_image_width + 1, 1))[0]
      y_offset = np.int32(np.random.randint(0, h - self.crop_image_height + 1, 1))[0]
    crop_img = img[y_offset:(y_offset + self.crop_image_height), x_offset:(x_offset + self.crop_image_width), :]
    return crop_img

class dataset_blur_image(dataset_image):
  def _load_one_image(self, img_name, test=False):
    img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (3,3), 0)
    if self.scale > 0:
      img = cv2.resize(img, None, fx=self.scale, fy=self.scale)
    img = np.float32(img)
    h, w, c = img.shape
    if test == True:
      x_offset = np.int((w - self.crop_image_width) / 2)
      y_offset = np.int((h - self.crop_image_height) / 2)
    else:
      if np.random.rand(1) > 0.5:
        img = cv2.flip(img, 1)
      x_offset = np.int32(np.random.randint(0, w - self.crop_image_width + 1, 1))[0]
      y_offset = np.int32(np.random.randint(0, h - self.crop_image_height + 1, 1))[0]
    crop_img = img[y_offset:(y_offset + self.crop_image_height), x_offset:(x_offset + self.crop_image_width), :]
    return crop_img

class dataset_imagenet_image(dataset_image):
  def __init__(self, specs):
    self.root = specs['root']
    self.folder = specs['folder']
    self.list_name = specs['list_name']
    self.crop_image_height = specs['crop_image_height']
    self.crop_image_width = specs['crop_image_width']
    self.scale = specs['scale']
    list_fullpath = os.path.join(self.root, self.list_name)
    with open(list_fullpath) as f:
      content = f.readlines()
    self.images = [os.path.join(self.root, self.folder, x.strip().split(' ')[0]) for x in content]
    np.random.shuffle(self.images)
    self.dataset_size = len(self.images)

  def _load_one_image(self, img_name, test=False):
    img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    if h > w:
      scale = self.crop_image_width * 1.0 / w
    else:
      scale = self.crop_image_height * 1.0 / h
    scale *= self.scale
    img = cv2.resize(img, None, fx=scale, fy=scale)
    img = np.float32(img)
    h, w, c = img.shape
    if test == True:
      x_offset = np.int((w - self.crop_image_width) / 2)[0]
      y_offset = np.int((h - self.crop_image_height) / 2)[0]
    else:
      if np.random.rand(1) > 0.5:
        img = cv2.flip(img, 1)
      x_offset = np.int32(np.random.randint(0, w - self.crop_image_width + 1, 1))[0]
      y_offset = np.int32(np.random.randint(0, h - self.crop_image_height + 1, 1))[0]
    crop_img = img[y_offset:(y_offset + self.crop_image_height), x_offset:(x_offset + self.crop_image_width), :]
    return crop_img

class dataset_dvd_image(dataset_image):
  def __init__(self, specs):
    self.root = specs['root']
    self.folder = specs['folder']
    self.list_name = specs['list_name']
    self.crop_image_height = specs['crop_image_height']
    self.crop_image_width = specs['crop_image_width']
    list_fullpath = os.path.join(self.root, self.list_name)
    with open(list_fullpath) as f:
      content = f.readlines()
    self.images = [os.path.join(self.root, self.folder, x.strip().split(' ')[0]) for x in content]
    np.random.shuffle(self.images)
    self.dataset_size = len(self.images)

  def _load_one_image(self, img_name, test=False):
    img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    # if h > w:
    #   scale = self.crop_image_width * 1.0 / w
    # else:
    #   scale = self.crop_image_height * 1.0 / h
    # img = cv2.resize(img, None, fx=scale, fy=scale)
    img = np.float32(img)
    h, w, c = img.shape
    if test == True:
      x_offset = np.int((w - self.crop_image_width) / 2)[0]
      y_offset = np.int((h - self.crop_image_height) / 2)[0]
    else:
      if np.random.rand(1) > 0.5:
        img = cv2.flip(img, 1)
      x_offset = np.int32(np.random.randint(0, w - self.crop_image_width + 1, 1))[0]
      y_offset = np.int32(np.random.randint(0, h - self.crop_image_height + 1, 1))[0]
    crop_img = img[y_offset:(y_offset + self.crop_image_height), x_offset:(x_offset + self.crop_image_width), :]
    return crop_img

