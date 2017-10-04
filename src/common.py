"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
import os
from datasets import *
import torchvision
from tensorboard import summary

def get_data_loader(conf, batch_size):
  dataset = []
  print("dataset=%s(conf)" % conf['class_name'])
  exec ("dataset=%s(conf)" % conf['class_name'])
  return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=10)

def prepare_snapshot_folder(snapshot_prefix):
  snapshot_directory = os.path.dirname(snapshot_prefix)
  if not os.path.exists(snapshot_directory):
    os.makedirs(snapshot_directory)
  return snapshot_directory

def prepare_image_folder(snapshot_directory):
  image_directory = os.path.join(snapshot_directory, 'images')
  if not os.path.exists(image_directory):
    os.makedirs(image_directory)
  return image_directory

def prepare_snapshot_and_image_folder(snapshot_prefix, iterations, image_save_iterations, all_size=1536):
  snapshot_directory = prepare_snapshot_folder(snapshot_prefix)
  image_directory = prepare_image_folder(snapshot_directory)
  write_html(snapshot_directory + "/index.html", iterations + 1, image_save_iterations, image_directory, all_size)
  return image_directory, snapshot_directory

def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
  html_file = open(filename, "w")
  html_file.write('''
  <!DOCTYPE html>
  <html>
  <head>
    <title>Experiment name = UnitNet</title>
    <meta content="1" http-equiv="reflesh">
  </head>
  <body>
  ''')
  html_file.write("<h3>current</h3>")
  img_filename = '%s/gen.jpg' % (image_directory)
  html_file.write("""
        <p>
        <a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
  for j in range(iterations,image_save_iterations-1,-1):
    if j % image_save_iterations == 0:
      img_filename = '%s/gen_%08d.jpg' % (image_directory, j)
      html_file.write("<h3>iteration [%d]</h3>" % j)
      html_file.write("""
            <p>
            <a href="%s">
              <img src="%s" style="width:%dpx">
            </a><br>
            <p>
            """ % (img_filename, img_filename, all_size))
  html_file.write("</body></html>")
  html_file.close()

def write_loss(iterations, max_iterations, trainer, train_writer):
  print("Iteration: %08d/%08d" % (iterations + 1, max_iterations))
  members = [attr for attr in dir(trainer) \
             if not callable(getattr(trainer, attr)) and not attr.startswith("__") and 'loss' in attr]
  for m in members:
    train_writer.add_summary(summary.scalar(m, getattr(trainer, m)), iterations + 1)
  members = [attr for attr in dir(trainer) \
             if not callable(getattr(trainer, attr)) and not attr.startswith("__") and 'acc' in attr]
  for m in members:
    train_writer.add_summary(summary.scalar(m, getattr(trainer, m)), iterations + 1)