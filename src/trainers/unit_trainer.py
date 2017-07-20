"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

from unit_nets import *
from init import *
import torch
import torch.nn as nn
import numpy as np


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
    true_a_acc = self._compute_true_acc(outputs_a_true)
    true_b_acc = self._compute_true_acc(outputs_b_true)
    fake_a_acc = self._compute_fake_acc(outputs_a_fake)
    fake_b_acc = self._compute_fake_acc(outputs_b_fake)
    self.dis_loss = loss.data.cpu().numpy()[0]
    self.dis_true_acc = 0.5 * (true_a_acc + true_b_acc)
    self.dis_fake_acc = 0.5 * (fake_a_acc + fake_b_acc)
    return

