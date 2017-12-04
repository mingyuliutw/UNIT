#!/usr/bin/env python
"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from common_net import *

# Coupled discriminator model for digit classification
class CoDis32x32(nn.Module):
  def _conv2d(self, n_in, n_out, kernel_size, stride, padding):
      return nn.Sequential(
        nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=1, padding=padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=stride)
      )

  def __init__(self, ch=64, input_dim_a=3, input_dim_b=1):
    super(CoDis32x32, self).__init__()
    self.conv0_a = self._conv2d(input_dim_a, ch, kernel_size=5, stride=2, padding=2)
    self.conv0_b = self._conv2d(input_dim_b, ch, kernel_size=5, stride=2, padding=2)
    self.conv1 = self._conv2d(ch * 1, ch * 2, kernel_size=5, stride=2, padding=2)
    self.conv2 = self._conv2d(ch * 2, ch * 4, kernel_size=5, stride=2, padding=2)
    self.conv3 = self._conv2d(ch * 4, ch * 8, kernel_size=5, stride=2, padding=2)
    self.conv4 = nn.Conv2d(ch * 8, 2, kernel_size=2, stride=1, padding=0)
    self.conv_cl = nn.Conv2d(ch * 8, 10, kernel_size=2, stride=1, padding=0)
    self.dropout0 = nn.Dropout(0.1)
    self.dropout1 = nn.Dropout(0.3)
    self.dropout2 = nn.Dropout(0.5)
    self.dropout3 = nn.Dropout(0.5)

  def forward(self, x_a, x_b):
    h0_a = self.conv0_a(x_a)
    h0_b = self.conv0_b(x_b)
    h3_a = self._forward_core(h0_a)
    h3_b = self._forward_core(h0_b)
    h0 = torch.cat((h0_a, h0_b), 0)
    h4_dropout = self.conv4(self._forward_core_dropout(h0))
    return h4_dropout.squeeze(), h3_a, h3_b

  def classify_a(self, x_a):
    h3_a = self._forward_core_dropout(self.conv0_a(x_a))
    h4_a = self.conv_cl(h3_a)
    return h4_a.squeeze()

  def classify_b(self, x_b):
    h3_b = self._forward_core_dropout(self.conv0_b(x_b))
    h4_b = self.conv_cl(h3_b)
    return h4_b.squeeze()

  def _forward_core_dropout(self, h0):
    h0 = self.dropout0(h0)
    h1 = self.dropout1(self.conv1(h0))
    h2 = self.dropout2(self.conv2(h1))
    return self.dropout3(self.conv3(h2))

  def _forward_core(self, h0):
    h1 = self.conv1(h0)
    h2 = self.conv2(h1)
    return self.conv3(h2)

# Coupled generator model for digit classification
class CoVAE32x32(nn.Module):
  def __init__(self, ch=32, input_dim_a=3, output_dim_a=3, input_dim_b=1, output_dim_b=1):
    super(CoVAE32x32, self).__init__()
    # Encoder layer #0
    self.g_en_conv0_a = LeakyReLUBNNSConv2d(input_dim_a, ch, kernel_size=5, stride=2, padding=2)
    self.g_en_conv0_b = LeakyReLUBNNSConv2d(input_dim_b, ch, kernel_size=5, stride=2, padding=2)
    self.g_en_conv1 = LeakyReLUBNNSConv2d(ch * 1, ch * 2, kernel_size=5, stride=2, padding=2)
    self.g_en_conv2 = LeakyReLUBNNSConv2d(ch * 2, ch * 4, kernel_size=8, stride=1, padding=0)
    self.g_en_conv3 = LeakyReLUBNNSConv2d(ch * 4, ch * 8, kernel_size=1, stride=1, padding=0)
    # Latent layer
    self.g_vae = GaussianVAE2D(ch * 8, ch * 8, kernel_size=1, stride=1)
    # Decoder layer #0
    self.g_de_conv0 = LeakyReLUBNNSConvTranspose2d(ch * 8, ch * 8, kernel_size=4, stride=2, padding=0)
    self.g_de_conv1 = LeakyReLUBNNSConvTranspose2d(ch * 8, ch * 4, kernel_size=4, stride=2, padding=1)
    self.g_de_conv2 = LeakyReLUBNNSConvTranspose2d(ch * 4, ch * 2, kernel_size=4, stride=2, padding=1)
    self.g_de_conv3_a = LeakyReLUBNNSConvTranspose2d(ch * 2, ch * 1, kernel_size=4, stride=2, padding=1)
    self.g_de_conv3_b = LeakyReLUBNNSConvTranspose2d(ch * 2, ch * 1, kernel_size=4, stride=2, padding=1)
    # Decoder layer #4
    self.de_conv4_a = nn.ConvTranspose2d(ch * 1, output_dim_a, kernel_size=1, stride=1, padding=0)
    self.de_conv4_b = nn.ConvTranspose2d(ch * 1, output_dim_b, kernel_size=1, stride=1, padding=0)
    self.de_tanh4_a = nn.Tanh()
    self.de_tanh4_b = nn.Tanh()

  def forward(self, x_a, x_b, gpu):
    en_h0_a = self.g_en_conv0_a(x_a)
    en_h0_b = self.g_en_conv0_b(x_b)
    en_h0 = torch.cat((en_h0_a, en_h0_b), 0)
    en_h1 = self.g_en_conv1(en_h0)
    en_h2 = self.g_en_conv2(en_h1)
    en_h3 = self.g_en_conv3(en_h2)
    z, mu, sd = self.g_vae.sample(en_h3)
    de_h0 = self.g_de_conv0(z)
    de_h1 = self.g_de_conv1(de_h0)
    de_h2 = self.g_de_conv2(de_h1)
    de_h3_a = self.g_de_conv3_a(de_h2)
    de_h3_b = self.g_de_conv3_b(de_h2)
    de_h4_a = self.de_tanh4_a(self.de_conv4_a(de_h3_a))
    de_h4_b = self.de_tanh4_b(self.de_conv4_b(de_h3_b))
    x_aa, x_ba = torch.split(de_h4_a, x_a.size(0), dim=0)
    x_ab, x_bb = torch.split(de_h4_b, x_a.size(0), dim=0)
    codes = (mu, sd)
    return x_aa, x_ba, x_ab, x_bb, [codes]
