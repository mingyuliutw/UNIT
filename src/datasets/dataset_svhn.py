"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

from __future__ import print_function
import scipy.io
import os
import numpy as np
import torch.utils.data as data
import torch
import urllib

class dataset_svhn_extra(data.Dataset):
# train 73,257, extra 531,131, test, 26,032
  def __init__(self, specs):
    self.url = 'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat'
    self.filename = 'extra_32x32.mat'
    self.root = specs['root']
    full_filepath = os.path.join(self.root, self.filename)
    self._download(full_filepath, self.url)
    data_set = self._load_samples(full_filepath)
    self.data = data_set[0]
    self.labels = data_set[1]
    self.num = self.data.shape[0]

  def __getitem__(self, index):
    img, label = self.data[index, ::], self.labels[index]
    label = torch.LongTensor([np.int64(label)])
    return img, label

  def __len__(self):
    return self.num

  def _download(self, filename, url):
    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
      os.mkdir(dirname)
    if os.path.isfile(filename):
      print("%s exists" % filename)
      return
    print("Download %s to %s" % (url, filename))
    urllib.urlretrieve(url, filename)
    print("Finish downloading %s" % filename)

  def _load_samples(self, file_path):
    print("[Loading samples.]")
    mat = scipy.io.loadmat(file_path)
    y = mat['y']
    item_index = np.where(y == 10)
    y[item_index] = 0
    x = mat['X']
    train_data = [2*np.float32(np.transpose(x, [3, 2, 0, 1]) / 255.0)-1, np.squeeze(y)]
    return train_data


class dataset_svhn_test(dataset_svhn_extra):
# train 73,257, extra 531,131, test, 26,032
  def __init__(self, specs):
    self.url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
    self.filename = 'test_32x32.mat'
    self.root = specs['root']
    full_filepath = os.path.join(self.root, self.filename)
    self._download(full_filepath, self.url)
    data_set = self._load_samples(full_filepath)
    self.data = data_set[0]
    self.labels = data_set[1]
    self.num = self.data.shape[0]


