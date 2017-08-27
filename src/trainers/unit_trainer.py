"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
from unit_nets import *
from init import *
import itertools
import torch
import torch.nn as nn
import numpy as np
from helpers import *


def _compute_true_acc(predictions):
  predictions = torch.ge(predictions.data, 0.5)
  if len(predictions.size()) == 3:
    predictions = predictions.view(predictions.size(0) * predictions.size(1) * predictions.size(2))
  acc = (predictions == 1).sum() / (1.0 * predictions.size(0))
  return acc


def _compute_true_acc2(predictions):
  _, true_predicts = torch.max(predictions.data, 1)
  acc = (true_predicts == 1).sum() / (1.0 * true_predicts.size(0))
  return acc


def _compute_fake_acc(predictions):
  predictions = torch.le(predictions.data, 0.5)
  if len(predictions.size()) == 3:
    predictions = predictions.view(predictions.size(0) * predictions.size(1) * predictions.size(2))
  acc = (predictions == 1).sum() / (1.0 * predictions.size(0))
  return acc


def _compute_fake_acc2(predictions):
  _, true_predicts = torch.max(predictions.data, 1)
  acc = (true_predicts == 0).sum() / (1.0 * true_predicts.size(0))
  return acc


class CoVAEGAN4DATrainer(nn.Module):
  # CoVAEGAN for domain adaptation
  def __init__(self, hyperparameters):
    super(CoVAEGAN4DATrainer, self).__init__()
    gen_ch = hyperparameters['gen_ch']
    dis_ch = hyperparameters['dis_ch']
    output_dim_a = hyperparameters['input_dim_a']
    output_dim_b = hyperparameters['input_dim_b']
    input_dim_a = output_dim_a
    input_dim_b = output_dim_b
    exec( 'self.dis = %s(dis_ch, input_dim_a, input_dim_b)' % hyperparameters['dis'])
    exec( 'self.gen = %s(gen_ch, input_dim_a, output_dim_a, input_dim_b, output_dim_b)' % hyperparameters['gen'])
    self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0005)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0005)
    self.dis.apply(xavier_weights_init)
    self.gen.apply(gaussian_weights_init)
    self.ll_loss_criterion = torch.nn.MSELoss()

  def _compute_kl(self, mu, sd):
    mu_2 = torch.pow(mu, 2)
    sd_2 = torch.pow(sd, 2)
    encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
    return encoding_loss

  def cuda(self, gpu=0):
    self.gpu = gpu
    self.dis.cuda(self.gpu)
    self.gen.cuda(self.gpu)

  def _compute_ll_loss(self,a,b):
    return self.ll_loss_criterion(a, b) * b.size(1) * b.size(2) * b.size(3)

  def gen_update(self, x_a, x_b, hyperparameters):
    self.gen.zero_grad()
    x_aa, x_ba, x_ab, x_bb, lt_codes = self.gen(x_a, x_b, self.gpu)
    fake_trans_res, fake_feat_ba, fake_feat_ab = self.dis(x_ba, x_ab)
    fake_recon_res, fake_feat_aa, fake_feat_bb = self.dis(x_aa, x_bb)
    ones = Variable(torch.LongTensor(np.ones(fake_trans_res.size(0), dtype=np.int)).cuda(self.gpu))
    ad_fake_trans_loss = nn.functional.cross_entropy(fake_trans_res, ones)
    ad_fake_recon_loss = nn.functional.cross_entropy(fake_recon_res, ones)
    ad_loss = ad_fake_trans_loss + ad_fake_recon_loss
    ll_loss_a = self._compute_ll_loss(x_aa, x_a)
    ll_loss_b = self._compute_ll_loss(x_bb, x_b)
    encoding_loss = 0
    for i, lt in enumerate(lt_codes):
      encoding_loss += 2 * self._compute_kl(*lt)
    total_loss = hyperparameters['gan_w'] * ad_loss + \
                 hyperparameters['vae_enc_w'] * encoding_loss + \
                 hyperparameters['vae_ll_w'] * (ll_loss_a + ll_loss_b)
    total_loss.backward()
    self.gen_opt.step()
    self.gen_ad_loss = ad_loss.data.cpu().numpy()[0]
    self.gen_ll_loss_a = ll_loss_a.data.cpu().numpy()[0]
    self.gen_ll_loss_b = ll_loss_b.data.cpu().numpy()[0]
    self.gen_enc_loss = encoding_loss.data.cpu().numpy()[0]
    self.gen_total_loss = total_loss.data.cpu().numpy()[0]
    return x_aa, x_ba, x_ab, x_bb

  def dis_update(self, images_a, labels_a, images_b, hyperparameters):
    self.dis.zero_grad()
    true_res, true_feat_a, true_feat_b = self.dis(images_a, images_b)
    ones = Variable(torch.LongTensor(np.ones(true_res.size(0), dtype=np.int)).cuda(self.gpu))
    ad_true_loss = nn.functional.cross_entropy(true_res, ones)

    x_aa, x_ba, x_ab, x_bb, lt_codes = self.gen(images_a, images_b, self.gpu)
    fake_trans_res, fake_feat_ba, fake_feat_ab = self.dis(x_ba, x_ab)
    fake_recon_res, fake_feat_aa, fake_feat_bb = self.dis(x_aa, x_bb)
    zeros = Variable(torch.LongTensor(np.zeros(fake_recon_res.size(0), dtype=np.int)).cuda(self.gpu))
    ad_fake_trans_loss = nn.functional.cross_entropy(fake_trans_res, zeros)
    ad_fake_recon_loss = nn.functional.cross_entropy(fake_recon_res, zeros)
    ad_fake_loss = 0.5 * ( ad_fake_trans_loss + ad_fake_recon_loss )

    dummy_variable = Variable(torch.zeros(fake_feat_aa.size()).cuda(self.gpu))
    feature_loss_a = self._compute_ll_loss(fake_feat_ab - fake_feat_aa, dummy_variable)
    feature_loss_b = self._compute_ll_loss(fake_feat_ba - fake_feat_bb, dummy_variable)

    # Classification loss
    cls_outputs = self.dis.classify_a(images_a)
    cls_loss = nn.functional.cross_entropy(cls_outputs, labels_a)
    _, cls_predicts = torch.max(cls_outputs.data, 1)
    cls_acc = (cls_predicts == labels_a.data).sum() / (1.0 * cls_predicts.size(0))

    true_acc = _compute_true_acc2(true_res)
    fake_trans_acc = _compute_fake_acc2(fake_trans_res)
    fake_recon_acc = _compute_fake_acc2(fake_recon_res)
    fake_acc = 0.5 * (fake_trans_acc + fake_recon_acc)

    total_loss = hyperparameters['gan_w'] * ( ad_true_loss + ad_fake_loss) + \
                 hyperparameters['cls_w'] * cls_loss + \
                 hyperparameters['feature_w'] * (feature_loss_a + feature_loss_b)
    total_loss.backward()
    self.dis_opt.step()
    self.dis_true_acc = true_acc
    self.dis_fake_acc = fake_acc
    self.dis_cls_acc  = cls_acc

    self.dis_cls_loss = cls_loss.data.cpu().numpy()[0]
    self.dis_ad_true_loss = ad_true_loss.data.cpu().numpy()[0]
    self.dis_ad_fake_loss_a = ad_fake_loss.data.cpu().numpy()[0]
    self.dis_feature_loss_a = feature_loss_a.data.cpu().numpy()[0]
    self.dis_feature_loss_b = feature_loss_b.data.cpu().numpy()[0]
    self.dis_total_loss = total_loss.data.cpu().numpy()[0]
    return

  def resume(self, snapshot_prefix):
    dirname = os.path.dirname(snapshot_prefix)
    last_model_name = get_model_list(dirname,"gen")
    if last_model_name is None:
      return 0
    self.gen.load_state_dict(torch.load(last_model_name))
    iterations = int(last_model_name[-12:-4])
    last_model_name = get_model_list(dirname, "dis")
    self.dis.load_state_dict(torch.load(last_model_name))
    print('Resume from iteration %d' % iterations)
    return iterations

  def save(self, snapshot_prefix, iterations):
    gen_filename = '%s_gen_%08d.pkl' % (snapshot_prefix, iterations + 1)
    dis_filename = '%s_dis_%08d.pkl' % (snapshot_prefix, iterations + 1)
    torch.save(self.gen.state_dict(), gen_filename)
    torch.save(self.dis.state_dict(), dis_filename)


class CoVAEGAN4DAXYCTrainer(nn.Module):
  # CoVAEGAN for domain adaptation with XY context images
  def __init__(self, hyperparameters):
    super(CoVAEGAN4DAXYCTrainer, self).__init__()
    gen_ch = hyperparameters['gen_ch']
    dis_ch = hyperparameters['dis_ch']
    output_dim_a = hyperparameters['input_dim_a']
    output_dim_b = hyperparameters['input_dim_b']
    input_dim_a = output_dim_a + 2
    input_dim_b = output_dim_b + 2
    exec( 'self.dis = %s(dis_ch, output_dim_a, output_dim_b)' % hyperparameters['dis'])
    exec( 'self.gen = %s(gen_ch, input_dim_a, output_dim_a, input_dim_b, output_dim_b)' % hyperparameters['gen'])
    self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0005)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0005)
    self.dis.apply(xavier_weights_init)
    self.gen.apply(gaussian_weights_init)       # Generator makes use of batchnorm so we use gaussian
    self.ll_loss_criterion = torch.nn.MSELoss() # We use MSELoss here
    xy = self._create_xy_image()
    self.xy = xy.unsqueeze(0).expand(hyperparameters['batch_size'], xy.size(0), xy.size(1), xy.size(2))

  def cuda(self, gpu=0):
    self.gpu = gpu
    self.dis.cuda(self.gpu)
    self.gen.cuda(self.gpu)
    self.xy = self.xy.cuda(self.gpu)

  def _create_xy_image(self, width=32):
    coordinates = list(itertools.product(range(width), range(width)))
    arr = (np.reshape(np.asarray(coordinates), newshape=[width, width, 2]) - width/2 ) / (width/2)
    new_map = np.transpose(np.float32(arr), [2, 0, 1])
    xy = Variable(torch.from_numpy(new_map), requires_grad=False)
    return xy

  def _compute_kl(self, mu, sd):
    mu_2 = torch.pow(mu, 2)
    sd_2 = torch.pow(sd, 2)
    encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
    return encoding_loss

  def _compute_ll_loss(self,a,b):
    return self.ll_loss_criterion(a, b) * b.size(1) * b.size(2) * b.size(3)

  def gen_update(self, x_a, x_b, hyperparameters):
    self.gen.zero_grad()
    x_a_xy = torch.cat((x_a, self.xy), 1) # Create input image to the generator a
    x_b_xy = torch.cat((x_b, self.xy), 1) # Create input image to the generator b
    x_aa, x_ba, x_ab, x_bb, lt_codes = self.gen(x_a_xy, x_b_xy, self.gpu)

    fake_recon_res, fake_feat_aa, fake_feat_bb = self.dis(x_aa, x_bb)
    fake_trans_res, fake_feat_ba, fake_feat_ab = self.dis(x_ba, x_ab)

    ones = Variable(torch.LongTensor(np.ones(fake_trans_res.size(0), dtype=np.int)).cuda(self.gpu))
    ad_fake_recon_loss = nn.functional.cross_entropy(fake_recon_res, ones)
    ad_fake_trans_loss = nn.functional.cross_entropy(fake_trans_res, ones)

    ad_loss = ad_fake_trans_loss + ad_fake_recon_loss
    ll_loss_a = self._compute_ll_loss(x_aa, x_a)
    ll_loss_b = self._compute_ll_loss(x_bb, x_b)
    encoding_loss = 0
    for i, lt in enumerate(lt_codes):
      encoding_loss += 2 * self._compute_kl(*lt)
    total_loss = hyperparameters['gan_w'] * ad_loss + \
                 hyperparameters['vae_enc_w'] * encoding_loss + \
                 hyperparameters['vae_ll_w'] * (ll_loss_a + ll_loss_b)
    total_loss.backward()
    self.gen_opt.step()
    self.gen_ad_loss = ad_loss.data.cpu().numpy()[0]
    self.gen_ll_loss_a = ll_loss_a.data.cpu().numpy()[0]
    self.gen_ll_loss_b = ll_loss_b.data.cpu().numpy()[0]
    self.gen_enc_loss = encoding_loss.data.cpu().numpy()[0]
    self.gen_total_loss = total_loss.data.cpu().numpy()[0]
    return x_aa, x_ba, x_ab, x_bb

  def dis_update(self, images_a, labels_a, images_b, hyperparameters):
    self.dis.zero_grad()
    true_res, true_feat_a, true_feat_b = self.dis(images_a, images_b)
    ones = Variable(torch.LongTensor(np.ones(true_res.size(0), dtype=np.int)).cuda(self.gpu))
    ad_true_loss = nn.functional.cross_entropy(true_res, ones)

    x_a_xy = torch.cat((images_a, self.xy), 1)
    x_b_xy = torch.cat((images_b, self.xy), 1)
    x_aa, x_ba, x_ab, x_bb, lt_codes = self.gen(x_a_xy, x_b_xy, self.gpu)
    fake_recon_res, fake_feat_aa, fake_feat_bb = self.dis(x_aa, x_bb)
    fake_trans_res, fake_feat_ba, fake_feat_ab = self.dis(x_ba, x_ab)
    zeros = Variable(torch.LongTensor(np.zeros(fake_recon_res.size(0), dtype=np.int)).cuda(self.gpu))
    ad_fake_recon_loss = nn.functional.cross_entropy(fake_recon_res, zeros)
    ad_fake_trans_loss = nn.functional.cross_entropy(fake_trans_res, zeros)
    ad_fake_loss = 0.5 * ( ad_fake_trans_loss + ad_fake_recon_loss )

    dummy_variable = Variable(torch.zeros(fake_feat_aa.size()).cuda(self.gpu))
    feature_loss_a = self._compute_ll_loss(fake_feat_ab - fake_feat_aa, dummy_variable)
    feature_loss_b = self._compute_ll_loss(fake_feat_ba - fake_feat_bb, dummy_variable)

    # Classification loss
    cls_outputs = self.dis.classify_a(images_a)
    cls_loss = nn.functional.cross_entropy(cls_outputs, labels_a)
    _, cls_predicts = torch.max(cls_outputs.data, 1)
    cls_acc = (cls_predicts == labels_a.data).sum() / (1.0 * cls_predicts.size(0))

    true_acc = _compute_true_acc2(true_res)
    fake_trans_acc = _compute_fake_acc2(fake_trans_res)
    fake_recon_acc = _compute_fake_acc2(fake_recon_res)
    fake_acc = 0.5 * (fake_trans_acc + fake_recon_acc)

    total_loss = hyperparameters['gan_w'] * ( ad_true_loss + ad_fake_loss) + \
                 hyperparameters['cls_w'] * cls_loss + \
                 hyperparameters['feature_w'] * (feature_loss_a + feature_loss_b)
    total_loss.backward()
    self.dis_opt.step()
    self.dis_true_acc = true_acc
    self.dis_fake_acc = fake_acc
    self.dis_cls_acc  = cls_acc

    self.dis_cls_loss = cls_loss.data.cpu().numpy()[0]
    self.dis_ad_true_loss = ad_true_loss.data.cpu().numpy()[0]
    self.dis_ad_fake_loss_a = ad_fake_loss.data.cpu().numpy()[0]
    self.dis_feature_loss_a = feature_loss_a.data.cpu().numpy()[0]
    self.dis_feature_loss_b = feature_loss_b.data.cpu().numpy()[0]
    self.dis_total_loss = total_loss.data.cpu().numpy()[0]
    return

  def resume(self, snapshot_prefix):
    dirname = os.path.dirname(snapshot_prefix)
    last_model_name = get_model_list(dirname,"gen")
    if last_model_name is None:
      return 0
    self.gen.load_state_dict(torch.load(last_model_name))
    iterations = int(last_model_name[-12:-4])
    last_model_name = get_model_list(dirname, "dis")
    self.dis.load_state_dict(torch.load(last_model_name))
    print('Resume from iteration %d' % iterations)
    return iterations

  def save(self, snapshot_prefix, iterations):
    gen_filename = '%s_gen_%08d.pkl' % (snapshot_prefix, iterations + 1)
    dis_filename = '%s_dis_%08d.pkl' % (snapshot_prefix, iterations + 1)
    torch.save(self.gen.state_dict(), gen_filename)
    torch.save(self.dis.state_dict(), dis_filename)



class UNITTrainer(nn.Module):
  # def _compute_ll_loss(self,a,b):
  #   dummy_tensor = Variable(torch.zeros(b.size(0), b.size(1), b.size(2), b.size(3))).cuda(self.gpu)
  #   return self.ll_loss_criterion(a - b, dummy_tensor) * b.size(1) * b.size(2) * b.size(3)

  def _compute_ll_loss(self,a,b):
    return self.ll_loss_criterion(a, b) * b.size(1) * b.size(2) * b.size(3)

  def _compute_kl(self, mu, sd):
    mu_2 = torch.pow(mu, 2)
    sd_2 = torch.pow(sd, 2)
    encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
    return encoding_loss

  def _compute_true_acc(self,predictions):
    predictions = torch.ge(predictions.data, 0.5)
    if len(predictions.size()) == 3:
      predictions = predictions.view(predictions.size(0) * predictions.size(1) * predictions.size(2))
    acc = (predictions == 1).sum() / (1.0 * predictions.size(0))
    return acc

  def _compute_fake_acc(self,predictions):
    predictions = torch.le(predictions.data, 0.5)
    if len(predictions.size()) == 3:
      predictions = predictions.view(predictions.size(0) * predictions.size(1) * predictions.size(2))
    acc = (predictions == 1).sum() / (1.0 * predictions.size(0))
    return acc

  def cuda(self, gpu):
    self.gpu = gpu
    self.dis.cuda(self.gpu)
    self.gen.cuda(self.gpu)
    self.dis_labels = self.dis_labels.cuda(self.gpu)
    self.gen_labels = self.gen_labels.cuda(self.gpu)

  def __init__(self, gen, dis, batch_size=1, ch=32, input_dims=[4, 4], image_size=256, lr=0.0001):
    super(UNITTrainer, self).__init__()
    true_input_dims = input_dims
    true_output_dims = input_dims
    exec( 'self.dis = %s(ch, true_input_dims)' % dis)
    exec( 'self.gen = %s(ch, true_input_dims, true_output_dims, image_size)' % gen )
    self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    n_per_side = image_size/self.dis.input_size
    if n_per_side < 1:
      ones_labels = Variable(torch.ones((batch_size * 2)))
      zeros_labels = Variable(torch.zeros((batch_size * 2)))
    else:
      ones_labels = Variable(torch.ones((batch_size*2,n_per_side,n_per_side)))
      zeros_labels = Variable(torch.zeros((batch_size*2,n_per_side,n_per_side)))
    self.dis_labels = torch.cat((ones_labels, zeros_labels, ones_labels, zeros_labels), 0)
    self.gen_labels = torch.cat((ones_labels, ones_labels), 0)
    self.dis.apply(gaussian_weights_init)
    self.gen.apply(gaussian_weights_init)
    self.ll_loss_criterion = torch.nn.L1Loss()
    self.tv_loss_criterion = torch.nn.L1Loss()
    self.gpu = 0

  def gen_update(self, images_a, images_b, gan_w, vae_ll_w, vae_enc_w):
    self.gen.zero_grad()
    x_aa, x_ba, x_ab, x_bb, lt_codes = self.gen(images_a, images_b, self.gpu)
    data_a = torch.cat((x_aa, x_ba), 0)
    data_b = torch.cat((x_bb, x_ab), 0)
    outputs = nn.functional.sigmoid(self.dis(data_a, data_b))
    ad_loss = nn.functional.binary_cross_entropy(outputs, self.gen_labels)
    ll_loss_a = self._compute_ll_loss(x_aa, images_a)
    ll_loss_b = self._compute_ll_loss(x_bb, images_b)
    encoding_loss = 0
    for i, lt in enumerate(lt_codes):
      encoding_loss += self._compute_kl(*lt)
    total_loss = gan_w * ad_loss + vae_ll_w * (ll_loss_a + ll_loss_b) + vae_enc_w * encoding_loss
    total_loss.backward()
    self.gen_opt.step()
    self.gen_ad_loss = ad_loss.data.cpu().numpy()[0]
    self.gen_ll_loss_a = ll_loss_a.data.cpu().numpy()[0]
    self.gen_ll_loss_b = ll_loss_b.data.cpu().numpy()[0]
    self.gen_encoding_loss = encoding_loss.data.cpu().numpy()[0]
    self.gen_total_loss = total_loss.data.cpu().numpy()[0]
    return x_aa, x_ba, x_ab, x_bb

  def dis_update(self, images_a1, images_b1, images_a2, images_b2):

    self.dis.zero_grad()
    x_aa, x_ba, x_ab, x_bb, lt_codes = self.gen(images_a2, images_b2, self.gpu)
    data_a = torch.cat((images_a1, images_a2, x_aa, x_ba), 0)
    data_b = torch.cat((images_b1, images_b2, x_bb, x_ab), 0)
    outputs = nn.functional.sigmoid(self.dis(data_a, data_b))
    loss = nn.functional.binary_cross_entropy(outputs, self.dis_labels)
    loss.backward()
    self.dis_opt.step()

    outputs_a, outputs_b = torch.split(outputs, data_a.size(0), dim=0)
    outputs_a_true, outputs_a_fake = torch.split(outputs_a, data_a.size(0) // 2, dim=0)
    outputs_b_true, outputs_b_fake = torch.split(outputs_b, data_b.size(0) // 2, dim=0)
    true_a_acc = _compute_true_acc(outputs_a_true)
    true_b_acc = _compute_true_acc(outputs_b_true)
    fake_a_acc = _compute_fake_acc(outputs_a_fake)
    fake_b_acc = _compute_fake_acc(outputs_b_fake)
    self.dis_loss = loss.data.cpu().numpy()[0]
    self.dis_true_acc = 0.5 * (true_a_acc + true_b_acc)
    self.dis_fake_acc = 0.5 * (fake_a_acc + fake_b_acc)
    return

