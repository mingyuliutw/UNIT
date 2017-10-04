#!/usr/bin/env python

"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
import sys
from tools import *
from trainers import *
from datasets import *
import torchvision
import itertools
from common import *
import tensorboard
from tensorboard import summary
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--gpu', type=int, help="gpu id", default=0)
parser.add_option('--resume', type=int, help="resume training?", default=0)
parser.add_option('--config', type=str, help="net configuration")
parser.add_option('--log', type=str, help="log path")

MAX_EPOCHS = 100000

def main(argv):
  (opts, args) = parser.parse_args(argv)

  # Load experiment setting
  assert isinstance(opts, object)
  config = NetConfig(opts.config)

  batch_size = config.hyperparameters['batch_size']
  max_iterations = config.hyperparameters['max_iterations']

  train_loader_a = get_data_loader(config.datasets['train_a'], batch_size)
  train_loader_b = get_data_loader(config.datasets['train_b'], batch_size)

  trainer = []
  exec ("trainer=%s(config.hyperparameters)" % config.hyperparameters['trainer'])
  # Check if resume training
  iterations = 0
  if opts.resume == 1:
    iterations = trainer.resume(config.snapshot_prefix)
  trainer.cuda(opts.gpu)

  ######################################################################################################################
  # Setup logger and repare image outputs
  train_writer = tensorboard.FileWriter("%s/%s" % (opts.log,os.path.splitext(os.path.basename(opts.config))[0]))
  image_directory, snapshot_directory = prepare_snapshot_and_image_folder(config.snapshot_prefix, iterations, config.image_save_iterations)

  for ep in range(0, MAX_EPOCHS):
    for it, (images_a, images_b) in enumerate(itertools.izip(train_loader_a,train_loader_b)):
      if images_a.size(0) != batch_size or images_b.size(0) != batch_size:
        continue
      images_a = Variable(images_a.cuda(opts.gpu))
      images_b = Variable(images_b.cuda(opts.gpu))

      # Main training code
      trainer.dis_update(images_a, images_b, config.hyperparameters)
      image_outputs = trainer.gen_update(images_a, images_b, config.hyperparameters)
      assembled_images = trainer.assemble_outputs(images_a, images_b, image_outputs)

      # Dump training stats in log file
      if (iterations+1) % config.display == 0:
        write_loss(iterations, max_iterations, trainer, train_writer)

      if (iterations+1) % config.image_save_iterations == 0:
        img_filename = '%s/gen_%08d.jpg' % (image_directory, iterations + 1)
        torchvision.utils.save_image(assembled_images.data / 2 + 0.5, img_filename, nrow=1)
        write_html(snapshot_directory + "/index.html", iterations + 1, config.image_save_iterations, image_directory)
      elif (iterations + 1) % config.image_display_iterations == 0:
        img_filename = '%s/gen.jpg' % (image_directory)
        torchvision.utils.save_image(assembled_images.data / 2 + 0.5, img_filename, nrow=1)

      # Save network weights
      if (iterations+1) % config.snapshot_save_iterations == 0:
        trainer.save(config.snapshot_prefix, iterations)

      iterations += 1
      if iterations >= max_iterations:
        return

if __name__ == '__main__':
  main(sys.argv)

