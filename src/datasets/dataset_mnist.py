"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

from __future__ import print_function
import cPickle
import gzip
import cv2
import os
import numpy as np
import torch.utils.data as data
import torch
import urllib


class dataset_mnist32x32_train(data.Dataset):
  def __init__(self, specs):
    self.url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    self.filename = 'mnist32x32.pkl.gz'
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

  def _load_samples(self, full_filepath):
    f = gzip.open(full_filepath, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    images = np.concatenate((train_set[0], valid_set[0]), axis=0)
    labels = np.concatenate((train_set[1], valid_set[1]), axis=0)
    images = images.reshape((images.shape[0], 1, 32, 32))
    # images = np.concatenate((images, 1 - images), axis=0)
    # labels = np.concatenate((labels, labels), axis=0)
    images = (images - 0.5) * 2
    return np.float32(images), labels

  def _download(self, filename, url):
    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
      os.mkdir(dirname)
    if os.path.isfile(filename):
      print("%s exists." % filename)
      return
    print("Download %s to %s" % (url, filename))
    urllib.urlretrieve(url, filename)
    print("Finish downloading %s" % filename)
    print("Resize images to 32x32")
    self._resize32x32(filename)

  def _resize32x32(self, full_filepath):
    def _resize(data_in):
      num_samples = data_in.shape[0]
      tmp_data_out = np.zeros((num_samples, 1, 32, 32))
      for i in range(0, num_samples):
        tmp_img = data_in[i, :].reshape(28, 28)
        tmp_data_out[i, 0, :, :] = cv2.resize(tmp_img, dsize=(32, 32), interpolation=cv2.INTER_NEAREST)
      return tmp_data_out

    f = gzip.open(full_filepath, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    with gzip.open(full_filepath, 'wb') as handle:
      cPickle.dump(([_resize(train_set[0]), train_set[1]],
                    [_resize(valid_set[0]), valid_set[1]],
                    [_resize(test_set[0]), test_set[1]]),
                   handle)


class dataset_mnist32x32_train_w_inv(data.Dataset):
  def __init__(self, specs):
    self.url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    self.filename = 'mnist32x32.pkl.gz'
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

  def _load_samples(self, full_filepath):
    f = gzip.open(full_filepath, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    images = np.concatenate((train_set[0], valid_set[0]), axis=0)
    labels = np.concatenate((train_set[1], valid_set[1]), axis=0)
    images = images.reshape((images.shape[0], 1, 32, 32))
    images = np.concatenate((images, 1 - images), axis=0)
    labels = np.concatenate((labels, labels), axis=0)
    images = (images - 0.5) * 2
    return np.float32(images), labels

  def _download(self, filename, url):
    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
      os.mkdir(dirname)
    if os.path.isfile(filename):
      print("%s exists." % filename)
      return
    print("Download %s to %s" % (url, filename))
    urllib.urlretrieve(url, filename)
    print("Finish downloading %s" % filename)
    print("Resize images to 32x32")
    self._resize32x32(filename)

  def _resize32x32(self, full_filepath):
    def _resize(data_in):
      num_samples = data_in.shape[0]
      tmp_data_out = np.zeros((num_samples, 1, 32, 32))
      for i in range(0, num_samples):
        tmp_img = data_in[i, :].reshape(28, 28)
        tmp_data_out[i, 0, :, :] = cv2.resize(tmp_img, dsize=(32, 32), interpolation=cv2.INTER_NEAREST)
      return tmp_data_out

    f = gzip.open(full_filepath, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    with gzip.open(full_filepath, 'wb') as handle:
      cPickle.dump(([_resize(train_set[0]), train_set[1]],
                    [_resize(valid_set[0]), valid_set[1]],
                    [_resize(test_set[0]), test_set[1]]),
                   handle)


class dataset_mnist32x32_test(dataset_mnist32x32_train):
  def __init__(self, specs):
    self.url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    self.filename = 'mnist32x32.pkl.gz'
    self.root = specs['root']
    full_filepath = os.path.join(self.root, self.filename)
    self._download(full_filepath, self.url)
    data_set = self._load_samples(full_filepath)
    self.data = data_set[0]
    self.labels = data_set[1]
    self.num = self.data.shape[0]

  def _load_samples(self, full_filepath):
    f = gzip.open(full_filepath, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    images = test_set[0]
    labels = test_set[1]
    images = (images - 0.5) * 2
    return np.float32(images), labels