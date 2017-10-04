"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
from cocogan_nets import *
from init import *
from helpers import get_model_list, _compute_fake_acc, _compute_true_acc
import torch
import torch.nn as nn
import os
import itertools


class COCOGANTrainer(nn.Module):
  def __init__(self, hyperparameters):
    super(COCOGANTrainer, self).__init__()
    lr = hyperparameters['lr']
    # Initiate the networks
    exec( 'self.dis = %s(hyperparameters[\'dis\'])' % hyperparameters['dis']['name'])
    exec( 'self.gen = %s(hyperparameters[\'gen\'])' % hyperparameters['gen']['name'] )
    # Setup the optimizers
    self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    # Network weight initialization
    self.dis.apply(gaussian_weights_init)
    self.gen.apply(gaussian_weights_init)
    # Setup the loss function for training
    self.ll_loss_criterion_a = torch.nn.L1Loss()
    self.ll_loss_criterion_b = torch.nn.L1Loss()


  def _compute_kl(self, mu):
    # def _compute_kl(self, mu, sd):
    # mu_2 = torch.pow(mu, 2)
    # sd_2 = torch.pow(sd, 2)
    # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
    # return encoding_loss
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def gen_update(self, images_a, images_b, hyperparameters):
    self.gen.zero_grad()
    x_aa, x_ba, x_ab, x_bb, shared = self.gen(images_a, images_b)
    x_bab, shared_bab = self.gen.forward_a2b(x_ba)
    x_aba, shared_aba = self.gen.forward_b2a(x_ab)
    outs_a, outs_b = self.dis(x_ba,x_ab)
    for it, (out_a, out_b) in enumerate(itertools.izip(outs_a, outs_b)):
      outputs_a = nn.functional.sigmoid(out_a)
      outputs_b = nn.functional.sigmoid(out_b)
      all_ones = Variable(torch.ones((outputs_a.size(0))).cuda(self.gpu))
      if it==0:
        ad_loss_a = nn.functional.binary_cross_entropy(outputs_a, all_ones)
        ad_loss_b = nn.functional.binary_cross_entropy(outputs_b, all_ones)
      else:
        ad_loss_a += nn.functional.binary_cross_entropy(outputs_a, all_ones)
        ad_loss_b += nn.functional.binary_cross_entropy(outputs_b, all_ones)

    enc_loss  = self._compute_kl(shared)
    enc_bab_loss = self._compute_kl(shared_bab)
    enc_aba_loss = self._compute_kl(shared_aba)
    ll_loss_a = self.ll_loss_criterion_a(x_aa, images_a)
    ll_loss_b = self.ll_loss_criterion_b(x_bb, images_b)
    ll_loss_aba = self.ll_loss_criterion_a(x_aba, images_a)
    ll_loss_bab = self.ll_loss_criterion_b(x_bab, images_b)
    total_loss = hyperparameters['gan_w'] * (ad_loss_a + ad_loss_b) + \
                 hyperparameters['ll_direct_link_w'] * (ll_loss_a + ll_loss_b) + \
                 hyperparameters['ll_cycle_link_w'] * (ll_loss_aba + ll_loss_bab) + \
                 hyperparameters['kl_direct_link_w'] * (enc_loss + enc_loss) + \
                 hyperparameters['kl_cycle_link_w'] * (enc_bab_loss + enc_aba_loss)
    total_loss.backward()
    self.gen_opt.step()
    self.gen_enc_loss = enc_loss.data.cpu().numpy()[0]
    self.gen_enc_bab_loss = enc_bab_loss.data.cpu().numpy()[0]
    self.gen_enc_aba_loss = enc_aba_loss.data.cpu().numpy()[0]
    self.gen_ad_loss_a = ad_loss_a.data.cpu().numpy()[0]
    self.gen_ad_loss_b = ad_loss_b.data.cpu().numpy()[0]
    self.gen_ll_loss_a = ll_loss_a.data.cpu().numpy()[0]
    self.gen_ll_loss_b = ll_loss_b.data.cpu().numpy()[0]
    self.gen_ll_loss_aba = ll_loss_aba.data.cpu().numpy()[0]
    self.gen_ll_loss_bab = ll_loss_bab.data.cpu().numpy()[0]
    self.gen_total_loss = total_loss.data.cpu().numpy()[0]
    return (x_aa, x_ba, x_ab, x_bb, x_aba, x_bab)

  def dis_update(self, images_a, images_b, hyperparameters):
    self.dis.zero_grad()
    x_aa, x_ba, x_ab, x_bb, shared = self.gen(images_a, images_b)
    data_a = torch.cat((images_a, x_ba), 0)
    data_b = torch.cat((images_b, x_ab), 0)
    res_a, res_b = self.dis(data_a,data_b)
    # res_true_a, res_true_b = self.dis(images_a,images_b)
    # res_fake_a, res_fake_b = self.dis(x_ba, x_ab)
    for it, (this_a, this_b) in enumerate(itertools.izip(res_a, res_b)):
      out_a = nn.functional.sigmoid(this_a)
      out_b = nn.functional.sigmoid(this_b)
      out_true_a, out_fake_a = torch.split(out_a, out_a.size(0) // 2, 0)
      out_true_b, out_fake_b = torch.split(out_b, out_b.size(0) // 2, 0)
      out_true_n = out_true_a.size(0)
      out_fake_n = out_fake_a.size(0)
      all1 = Variable(torch.ones((out_true_n)).cuda(self.gpu))
      all0 = Variable(torch.zeros((out_fake_n)).cuda(self.gpu))
      ad_true_loss_a = nn.functional.binary_cross_entropy(out_true_a, all1)
      ad_true_loss_b = nn.functional.binary_cross_entropy(out_true_b, all1)
      ad_fake_loss_a = nn.functional.binary_cross_entropy(out_fake_a, all0)
      ad_fake_loss_b = nn.functional.binary_cross_entropy(out_fake_b, all0)
      if it==0:
        ad_loss_a = ad_true_loss_a + ad_fake_loss_a
        ad_loss_b = ad_true_loss_b + ad_fake_loss_b
      else:
        ad_loss_a += ad_true_loss_a + ad_fake_loss_a
        ad_loss_b += ad_true_loss_b + ad_fake_loss_b
      true_a_acc = _compute_true_acc(out_true_a)
      true_b_acc = _compute_true_acc(out_true_b)
      fake_a_acc = _compute_fake_acc(out_fake_a)
      fake_b_acc = _compute_fake_acc(out_fake_b)
      exec( 'self.dis_true_acc_%d = 0.5 * (true_a_acc + true_b_acc)' %it)
      exec( 'self.dis_fake_acc_%d = 0.5 * (fake_a_acc + fake_b_acc)' %it)
    loss = hyperparameters['gan_w'] * ( ad_loss_a + ad_loss_b )
    loss.backward()
    self.dis_opt.step()
    self.dis_loss = loss.data.cpu().numpy()[0]
    return

  def assemble_outputs(self, images_a, images_b, network_outputs):
    images_a = self.normalize_image(images_a)
    images_b = self.normalize_image(images_b)
    x_aa = self.normalize_image(network_outputs[0])
    x_ba = self.normalize_image(network_outputs[1])
    x_ab = self.normalize_image(network_outputs[2])
    x_bb = self.normalize_image(network_outputs[3])
    x_aba = self.normalize_image(network_outputs[4])
    x_bab = self.normalize_image(network_outputs[5])
    return torch.cat((images_a[0:1, ::], x_aa[0:1, ::], x_ab[0:1, ::], x_aba[0:1, ::],
                      images_b[0:1, ::], x_bb[0:1, ::], x_ba[0:1, ::], x_bab[0:1, ::]), 3)

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

  def cuda(self, gpu):
    self.gpu = gpu
    self.dis.cuda(gpu)
    self.gen.cuda(gpu)

  def normalize_image(self, x):
    return x[:,0:3,:,:]
