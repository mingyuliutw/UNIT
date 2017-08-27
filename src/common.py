import os
from datasets import *
import torchvision
from tensorboard import summary

def get_data_loader(conf, batch_size):
  dataset = []
  print("dataset=%s(conf)" % conf['class_name'])
  exec ("dataset=%s(conf)" % conf['class_name'])
  return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

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