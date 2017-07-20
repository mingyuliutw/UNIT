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
import tensorboard
from tensorboard import summary
from optparse import OptionParser
#!/usr/bin/env python
import sys
from tools import *
from trainers import *
from datasets import *
import torchvision
import itertools
import tensorboard
from tensorboard import summary
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--lr',
                  type=float,
                  help="learning rate",
                  default=0.0001)
parser.add_option('--config',
                  type=str,
                  help="net configuration",
                  default="../exps_cyclops/cyclops_all_v1.yaml")
parser.add_option('--log',
                  type=str,
                  help="log path",
                  default="/data/projects/unit2_pytorch/logs")
parser.add_option('--gpu',
                  type=int,
                  help="gpu id",
                  default=0)
parser.add_option('--resume',
                  type=int,
                  help="resume training?",
                  default=0)

MAX_EPOCHS = 100000

def standardize_image(images):
  channels = images.size(1)
  if channels > 3:
    new_images = images[:,0:3,::]
  elif channels == 1:
    new_images = torch.cat((images,images,images),1)
  elif channels == 2:
    new_images = torch.cat((images[:,0:1,::], images[:,0:1,::], images[:,0:1,::]), 1)
  elif channels == 3:
    new_images = images
  return new_images

def make_save_image(images_a, x_aa, x_ab, images_b, x_ba, x_bb):
  images_a = standardize_image(images_a)
  x_aa = standardize_image(x_aa)
  x_ab = standardize_image(x_ab)
  images_b = standardize_image(images_b)
  x_ba = standardize_image(x_ba)
  x_bb = standardize_image(x_bb)
  assembled_images = torch.cat((images_a, x_aa, x_ab, images_b, x_ba, x_bb), 3)
  return assembled_images

def get_model_list(dirname, key):
  if os.path.exists(dirname) is False:
    return None
  gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                os.path.isfile(os.path.join(dirname, f)) and key in f and "pkl" in f]
  if gen_models is None:
    return None
  gen_models.sort()
  last_model_name = gen_models[-1]
  return last_model_name

def resume(trainer, snapshot_prefix):
  dirname = os.path.dirname(snapshot_prefix)
  last_model_name = get_model_list(dirname,"gen")
  if last_model_name is None:
    return 0
  trainer.gen.load_state_dict(torch.load(last_model_name))
  iterations = int(last_model_name[-12:-4])
  last_model_name = get_model_list(dirname, "dis")
  trainer.dis.load_state_dict(torch.load(last_model_name))
  print('Resume from iteration %d' % iterations)
  return iterations

def get_data_loader(conf, batch_size, context=0):
  data_a = []
  exec ("data_a=%s(conf['root'],conf['folder'],conf['list'],conf['image_size'],conf['scale'],context)"%conf['class_name'])
  return torch.utils.data.DataLoader(dataset=data_a, batch_size=batch_size, shuffle=True)

def write_html(filename, iterations, image_save_iterations, image_directory, image_size):
  all_size = 6*image_size
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


def main(argv):
  (opts, args) = parser.parse_args(argv)
  # Load experiment setting
  assert isinstance(opts, object)
  config = NetConfig(opts.config)
  train_writer = tensorboard.FileWriter("%s/%s" % (opts.log,os.path.splitext(os.path.basename(opts.config))[0]))

  max_iterations = config.hyperparameters['max_iterations']
  batch_size = config.hyperparameters['batch_size']
  vae_enc_w = config.hyperparameters['vae_enc_w']
  vae_ll_w = config.hyperparameters['vae_ll_w']
  gan_w = config.hyperparameters['gan_w']
  ch = config.hyperparameters['ch']
  gen_net = config.hyperparameters['gen']
  dis_net = config.hyperparameters['dis']

  image_size = config.datasets['a']['image_size']
  input_dims = list()
  input_dims.append(config.datasets['a']['channels'])
  input_dims.append(config.datasets['b']['channels'])

  # Load datasets
  train_loader_a = get_data_loader(config.datasets['a'], batch_size)
  train_loader_b = get_data_loader(config.datasets['b'], batch_size)
  train_loader_a2 = get_data_loader(config.datasets['a'], batch_size)
  train_loader_b2 = get_data_loader(config.datasets['b'], batch_size)
  trainer = UNITTrainer(gen_net, dis_net, batch_size, ch, input_dims, image_size, opts.lr)

  iterations = 0
  if opts.resume == 1:
    iterations = resume(trainer, config.snapshot_prefix)

  trainer.cuda(opts.gpu)

  directory = os.path.dirname(config.snapshot_prefix)
  image_directory = directory + "/images"
  if not os.path.exists(directory):
    os.makedirs(directory)
  if not os.path.exists(image_directory):
    os.makedirs(image_directory)

  write_html(directory + "/index.html", iterations + 1, config.image_save_iterations, image_directory, image_size)

  for ep in range(0, MAX_EPOCHS):
    for it, (images_a, images_b, images_a2, images_b2) in enumerate(itertools.izip(train_loader_a, train_loader_b, train_loader_a2, train_loader_b2)):
      if images_a.size(0) != batch_size or images_b.size(0) != batch_size:
        continue
      images_a = Variable(images_a.cuda(opts.gpu))
      images_b = Variable(images_b.cuda(opts.gpu))
      images_a2 = Variable(images_a2.cuda(opts.gpu))
      images_b2 = Variable(images_b2.cuda(opts.gpu))
      # Main training code
      trainer.dis_update(images_a, images_b, images_a2, images_b2)
      x_aa, x_ba, x_ab, x_bb = trainer.gen_update(images_a, images_b, gan_w, vae_ll_w, vae_enc_w)

      # Dump training stats in log file
      if (iterations+1) % config.display == 0:
        print("Iteration: %08d/%08d" %(iterations+1,max_iterations))
        members = [attr for attr in dir(trainer) \
                   if not callable(getattr(trainer, attr)) and not attr.startswith("__") and 'loss' in attr]
        for m in members:
          train_writer.add_summary(summary.scalar(m, getattr(trainer, m)), iterations + 1)

        members = [attr for attr in dir(trainer) \
                   if not callable(getattr(trainer, attr)) and not attr.startswith("__") and 'acc' in attr]
        for m in members:
          train_writer.add_summary(summary.scalar(m, getattr(trainer, m)), iterations + 1)

      # Save intermediate visualization results
      if (iterations+1) % config.image_save_iterations == 0:
        assembled_images = make_save_image(images_a[0:1,::], x_aa[0:1,::], x_ab[0:1,::], images_b[0:1,::], x_ba[0:1,::], x_bb[0:1,::])
        img_filename = '%s/gen_%08d.jpg' % (image_directory, iterations + 1)
        torchvision.utils.save_image(assembled_images.data / 2 + 0.5, img_filename, nrow=1)
        write_html(directory + "/index.html", iterations + 1, config.image_save_iterations, image_directory, image_size)
      else:
        assembled_images = make_save_image(images_a[0:1,::], x_aa[0:1,::], x_ab[0:1,::], images_b[0:1,::],x_ba[0:1,::], x_bb[0:1,::])
        img_filename = '%s/gen.jpg' % (image_directory)
        torchvision.utils.save_image(assembled_images.data / 2 + 0.5, img_filename, nrow=1)

      # Save network weights
      if (iterations+1) % config.snapshot_save_iterations == 0:
        gen_filename = '%s_gen_%08d.pkl' % (config.snapshot_prefix, iterations + 1)
        dis_filename = '%s_dis_%08d.pkl' % (config.snapshot_prefix, iterations + 1)
        torch.save(trainer.gen.state_dict(), gen_filename)
        torch.save(trainer.dis.state_dict(), dis_filename)




      iterations += 1
      if iterations == max_iterations:
        return

if __name__ == '__main__':
  main(sys.argv)

