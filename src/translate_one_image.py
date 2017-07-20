#!/usr/bin/env python

"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import sys
import os
from trainers import *
import cv2
import torchvision
from tools import *
from train import standardize_image
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="net configuration")
parser.add_option('--image_name',
                  type=str,
                  help="dataset folder")
parser.add_option('--output_image_name',
                  type=str,
                  help="dataset folder")
parser.add_option('--weights',
                  type=str,
                  help="file location to the trained generator network weights")
parser.add_option('--a2b',
                  type=int,
                  help="1 for a2b and others for b2a",
                  default=1)
parser.add_option('--gpu',
                  type=int,
                  help="gpu id",
                  default=0)


def create_context_image(image_size):
  y = np.linspace(-1, 1, image_size)
  y_img = np.zeros((1, image_size, image_size), dtype=np.float32)
  for i in range(0, image_size):
    y_img[0, :, i] = y
  return y_img

def prepare_data(img_name, crop_image_size):
  img = cv2.imread(img_name, -1)
  if len(img.shape) == 3: # Color image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    my = (h - crop_image_size) / 2
    mx = (w - crop_image_size) / 2
    if h < crop_image_size or w < crop_image_size:
      print("Crop image size is larger than input image size.")
      return
    crop_img = img[my:(my+crop_image_size),mx:(mx+crop_image_size), :]
  else:
    h, w = img.shape
    my = (h - crop_image_size) / 2
    mx = (w - crop_image_size) / 2
    if h < crop_image_size or w < crop_image_size:
      print("Crop image size is larger than input image size.")
      return
    crop_img = img[my:(my+crop_image_size),mx:(mx+crop_image_size)]
  return crop_img


def main(argv):
  (opts, args) = parser.parse_args(argv)
  assert isinstance(opts, object)
  config = NetConfig(opts.config)

  batch_size = config.hyperparameters['batch_size']
  ch = config.hyperparameters['ch']
  gen_net = config.hyperparameters['gen']
  dis_net = config.hyperparameters['dis']

  image_size = config.datasets['a']['image_size']
  input_dims = list()
  input_dims.append(config.datasets['a']['channels'])
  input_dims.append(config.datasets['b']['channels'])

  context = 0

  # Prepare network
  trainer = UNITTrainer(gen_net, dis_net, batch_size, ch, input_dims, image_size)
  trainer.gen.load_state_dict(torch.load(opts.weights))
  trainer.cuda(opts.gpu)

  img = prepare_data(opts.image_name, image_size)
  raw_data = img.transpose((2, 0, 1))  # convert to HWC
  data = torch.FloatTensor((np.float32(raw_data) / 255.0 - 0.5) * 2)
  if context == 1:
    y_img = torch.FloatTensor(create_context_image(image_size))
    final_data = torch.cat((data, y_img), 0)
  else:
    final_data = data
  final_data = final_data.contiguous()
  final_data = Variable(final_data.view(1,final_data.size(0),final_data.size(1),final_data.size(2))).cuda(opts.gpu)

  if opts.a2b == 1:
    output_data = trainer.gen.translate_a_to_b(final_data, opts.gpu)
  else:
    output_data = trainer.gen.translate_b_to_a(final_data, opts.gpu)

  final_data = standardize_image(final_data)
  output_data = standardize_image(output_data[0])
  assembled_images = torch.cat((final_data, output_data), 3)

  torchvision.utils.save_image(assembled_images.data / 2.0 + 0.5, opts.output_image_name)

  return 0


if __name__ == '__main__':
  main(sys.argv)

