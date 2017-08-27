#!/usr/bin/env python
"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
import sys
from tools import *
from trainers import *
from common import *
import torchvision
import itertools
import tensorboard
from tensorboard import summary
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--gpu', type=int, help="gpu id", default=0)
parser.add_option('--resume', type=int, help="resume training?", default=0)
parser.add_option('--config',
                  type=str,
                  help="net configuration",
                  default="../exps/covaegan/svhn2mnist_v11.yaml")
parser.add_option('--log',
                  type=str,
                  help="log path",
                  default="../logs/unit_release")

MAX_EPOCHS = 100000

def main(argv):
  (opts, args) = parser.parse_args(argv)

  # Load experiment setting
  assert isinstance(opts, object)
  config = NetConfig(opts.config)

  # Read training parameters from the yaml file
  hyperparameters = {}
  for key in config.hyperparameters:
    exec ('hyperparameters[\'%s\'] = config.hyperparameters[\'%s\']' % (key,key))

  trainer = []
  exec ("trainer=%s(hyperparameters)" % hyperparameters['trainer'])
  trainer.cuda(opts.gpu)

  iterations = 0

  train_writer = tensorboard.FileWriter("%s/%s" % (opts.log,os.path.splitext(os.path.basename(opts.config))[0]))
  snapshot_directory = prepare_snapshot_folder(config.snapshot_prefix)
  image_directory = prepare_image_folder(snapshot_directory)

  batch_size = hyperparameters['batch_size']
  max_iterations = hyperparameters['max_iterations']

  # Load datasets
  train_loader_a = get_data_loader(config.datasets['train_a'], batch_size)
  train_loader_b = get_data_loader(config.datasets['train_b'], batch_size)
  test_loader_b = get_data_loader(config.datasets['test_b'], batch_size = hyperparameters['test_batch_size'])

  best_score = 0
  for ep in range(0, MAX_EPOCHS):
    for it, ((images_a, labels_a), (images_b,labels_b)) in enumerate(itertools.izip(train_loader_a, train_loader_b)):
      if images_a.size(0) != batch_size or images_b.size(0) != batch_size:
        continue
      trainer.dis.train()
      images_a = Variable(images_a.cuda(opts.gpu))
      labels_a = Variable(labels_a.cuda(opts.gpu)).view(images_a.size(0))
      images_b = Variable(images_b.cuda(opts.gpu))
      # Main training code
      trainer.dis_update(images_a, labels_a, images_b, hyperparameters)
      x_aa, x_ba, x_ab, x_bb = trainer.gen_update(images_a, images_b, hyperparameters)

      # Dump training stats in log file
      if (iterations+1) % config.display == 0:
        write_loss(iterations, max_iterations, trainer, train_writer)

      # # Save network weights
      if (iterations+1) % config.snapshot_save_iterations == 0:
        trainer.dis.eval()
        score = 0
        num_samples = 0
        for tit, (test_images_b, test_labels_b) in enumerate(test_loader_b):
          test_images_b = Variable(test_images_b.cuda(opts.gpu))
          test_labels_b = Variable(test_labels_b.cuda(opts.gpu)).view(test_images_b.size(0))
          cls_outputs = trainer.dis.classify_b(test_images_b)
          _, cls_predicts = torch.max(cls_outputs.data, 1)
          cls_acc = (cls_predicts == test_labels_b.data).sum()
          score += cls_acc
          num_samples += test_images_b.size(0)
        score /= 1.0 * num_samples
        print('Classification accuracy for Test_B dataset: %4.4f' % score)
        if score > best_score:
          best_score = score
          trainer.save(config.snapshot_prefix, iterations=-1)
        train_writer.add_summary(summary.scalar('test_b_acc', score), iterations + 1)
        img_name = image_directory + "/images_a.jpg"
        torchvision.utils.save_image(images_a.data / 2 + 0.5, img_name)
        img_name = image_directory + "/images_b.jpg"
        torchvision.utils.save_image(images_b.data / 2 + 0.5, img_name)
        img_name = image_directory + "/x_aa.jpg"
        torchvision.utils.save_image(x_aa.data / 2 + 0.5, img_name)
        img_name = image_directory + "/x_ab.jpg"
        torchvision.utils.save_image(x_ab.data / 2 + 0.5, img_name)
        img_name = image_directory + "/x_bb.jpg"
        torchvision.utils.save_image(x_bb.data / 2 + 0.5, img_name)
        img_name = image_directory + "/x_ba.jpg"
        torchvision.utils.save_image(x_ba.data / 2 + 0.5, img_name)


      iterations += 1
      if iterations == max_iterations:
        return

if __name__ == '__main__':
  main(sys.argv)

