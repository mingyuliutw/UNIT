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

class dataset_image(data.Dataset):
  def __init__(self, root, folder, list_name, image_size, scale=0, context=0, random_crop=True, random_mirror=True):
    self.context = context
    self.scale = scale
    self.image_size = image_size
    self.random_crop = random_crop
    self.random_mirror = random_mirror
    list_fullpath = os.path.join(root, list_name)
    with open(list_fullpath) as f:
      content = f.readlines()
    self.images = [os.path.join(root, folder, x.strip().split(' ')[0]) for x in content]
    np.random.shuffle(self.images)
    self.dataset_size = len(self.images)
    if self.context==1:
      t_img = self._load_one_image(self.images[0])
      h, w, c = t_img.shape
      # Create y image
      y = np.linspace(-1,1,h)
      y_img = np.zeros((1,h,h), dtype=np.float32)
      for i in range(0,h):
          y_img[0,:,i] = y
      self.y_img = torch.FloatTensor(y_img)

  def __getitem__(self, index):
    crop_img = self._load_one_image(self.images[index])
    raw_data = crop_img.transpose((2, 0, 1))  # convert to HWC
    data = ((torch.FloatTensor(raw_data)/255.0)-0.5)*2
    if self.context==0:
      return data
    final_data = torch.cat((data, self.y_img),0)
    return final_data

  def __len__(self):
    return self.dataset_size

  def _load_one_image(self, img_name, test=False):
    img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    if self.scale > 0:
      img = cv2.resize(img,None,fx=self.scale,fy=self.scale)
    img = np.float32(img)
    h, w, c = img.shape
    if test==True:
      x_offset = np.int( (w - self.image_size)/2 )
      y_offset = np.int( (h - self.image_size)/2 )
    else:
      if np.random.rand(1) > 0.5:
        img = cv2.flip(img, 1)
      x_offset = np.int32(np.random.randint(0, w - self.image_size + 1, 1))
      y_offset = np.int32(np.random.randint(0, h - self.image_size + 1, 1))
    crop_img = img[y_offset:(y_offset + self.image_size), x_offset:(x_offset + self.image_size), :]
    return crop_img


class dataset_gopro_image(dataset_image):
  def _load_one_image(self, img_name, test=False):
    img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    new_h = np.int(h*2/3)
    img = img[0:new_h,::]
    h, w, c = img.shape
    new_width = np.int(self.image_size*w*1.0/h)
    img = cv2.resize(img, (new_width, self.image_size))
    h, w, c = img.shape
    img = np.float32(img)
    if test==True:
      x_offset = np.int( (w - self.image_size)/2 )
      y_offset = np.int( (h - self.image_size)/2 )
    else:
      if np.random.rand(1) > 0.5:
        img = cv2.flip(img, 1)
      x_offset = np.int32(np.random.randint(0, w - self.image_size + 1, 1))
      y_offset = np.int32(np.random.randint(0, h - self.image_size + 1, 1))
    crop_img = img[y_offset:(y_offset + self.image_size), x_offset:(x_offset + self.image_size), :]
    return crop_img

class dataset_thermal_image(dataset_image):
  def _load_one_image(self, img_name, test=False):
    img = cv2.imread(img_name, 0)
    if self.scale > 0:
      img = cv2.resize(img,None,fx=self.scale,fy=self.scale)
    h, w = img.shape
    img = np.float32(img)
    if test==True:
      x_offset = np.int( (w - self.image_size)/2 )
      y_offset = np.int( (h - self.image_size)/2 )
    else:
      if np.random.rand(1) > 0.5:
        img = cv2.flip(img, 1)
      x_offset = np.int32(np.random.randint(0, w - self.image_size + 1, 1))
      y_offset = np.int32(np.random.randint(0, h - self.image_size + 1, 1))
    crop_img = img[y_offset:(y_offset + self.image_size), x_offset:(x_offset + self.image_size)]
    crop_img = np.expand_dims(crop_img, axis=2)
    return crop_img

class dataset_square_image(dataset_image):
  def _load_one_image(self, img_name, test=False):
    img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    new_img_size = np.int(1.1*self.image_size)
    img = cv2.resize(img,(new_img_size,new_img_size))
    h, w, c = img.shape
    if test==True:
      x_offset = np.int( (w - self.image_size)/2 )
      y_offset = np.int( (h - self.image_size)/2 )
    else:
      if np.random.rand(1) > 0.5:
        img = cv2.flip(img, 1)
      x_offset = np.int32(np.random.randint(0, w - self.image_size + 1, 1))
      y_offset = np.int32(np.random.randint(0, h - self.image_size + 1, 1))
    img = np.float32(img)
    crop_img = img[y_offset:(y_offset + self.image_size), x_offset:(x_offset + self.image_size), :]
    return crop_img