"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


# Network basic building blocks
class Bias2d(nn.Module):
  def __init__(self, channels):
    super(Bias2d, self).__init__()
    self.bias = nn.Parameter(torch.Tensor(channels))
    self.reset_parameters()

  def reset_parameters(self):
    self.bias.data.normal_(0, 0.002)

  def forward(self, x):
    n, c, h, w = x.size()
    return x + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(n, c, h, w)


class GaussianVAE2D(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(GaussianVAE2D, self).__init__()
    self.en_mu = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
    self.reset_parameters()

  def reset_parameters(self):
    self.en_mu.weight.data.normal_(0, 0.002)
    self.en_mu.bias.data.normal_(0, 0.002)

  def forward(self, x):
    mu = self.en_mu(x)
    return mu

  def sample(self, x, gpu):
    mu = self.en_mu(x)
    sd = Variable(torch.ones(mu.size(0), mu.size(1), mu.size(2), mu.size(3))).cuda(gpu)
    noise = Variable(torch.randn(mu.size(0), mu.size(1), mu.size(2), mu.size(3))).cuda(gpu)
    return mu + sd.mul(noise), mu, sd


class GaussianVAE2DFull(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(GaussianVAE2DFull, self).__init__()
    self.en_mu = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
    self.en_sigma = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
    self.softplus = nn.Softplus()
    self.reset_parameters()

  def reset_parameters(self):
    self.en_mu.weight.data.normal_(0, 0.002)
    self.en_mu.bias.data.normal_(0, 0.002)
    self.en_sigma.weight.data.normal_(0, 0.002)
    self.en_sigma.bias.data.normal_(0, 0.002)

  def forward(self, x):
    mu = self.en_mu(x)
    sd = self.softplus(self.en_sigma(x))
    return mu, sd

  def sample(self, x, gpu):
    mu = self.en_mu(x)
    sd = self.softplus(self.en_sigma(x))
    noise = Variable(torch.randn(mu.size(0), mu.size(1), mu.size(2), mu.size(3))).cuda(gpu)
    return mu + sd.mul(noise), mu, sd


class CoDis(nn.Module):
  def _conv2d(self, n_in, n_out, kernel_size, stride, padding=0):
    return nn.Sequential(
      nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.LeakyReLU()
    )

  def _conv2d_w_bn(self, n_in, n_out, kernel_size, stride, padding=0):
    return nn.Sequential(
      nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.BatchNorm2d(n_out),
      nn.LeakyReLU()
    )

  def __init__(self, ch=32, input_dims = [3, 3]):
    super(CoDis, self).__init__()
    self.input_size = 128
    self.g_conv0_a = self._conv2d(input_dims[0], ch, kernel_size=5, stride=2, padding=2)  # 64
    self.g_conv0_b = self._conv2d(input_dims[1], ch, kernel_size=5, stride=2, padding=2)  # 64
    self.g_conv1_a = self._conv2d(ch * 1, ch * 2, kernel_size=5, stride=2, padding=2)  # 32
    self.g_conv1_b = self._conv2d(ch * 1, ch * 2, kernel_size=5, stride=2, padding=2)  # 32
    self.g_conv2 = self._conv2d(ch * 2, ch * 4, kernel_size=5, stride=2, padding=2)  # 16
    self.g_conv3 = self._conv2d(ch * 4, ch * 8, kernel_size=3, stride=2, padding=1)  # 8
    self.g_conv4 = self._conv2d(ch * 8, ch * 16, kernel_size=3, stride=2, padding=1)  # 4
    self.g_conv5 = self._conv2d(ch * 16, ch * 32, kernel_size=3, stride=2, padding=1)  # 2
    self.g_conv6 = self._conv2d(ch * 32, ch * 64, kernel_size=3, stride=2, padding=1)  # 1
    self.conv7 = nn.Conv2d(ch * 64, 1, kernel_size=1, stride=1, padding=0)

  def forward(self, x_a, x_b):
    h0_a = self.g_conv0_a(x_a)
    h0_b = self.g_conv0_b(x_b)
    h1_a = self.g_conv1_a(h0_a)
    h1_b = self.g_conv1_b(h0_b)
    h1 = torch.cat((h1_a, h1_b), 0)
    h2 = self.g_conv2(h1)
    h3 = self.g_conv3(h2)
    h4 = self.g_conv4(h3)
    h5 = self.g_conv5(h4)
    h6 = self.g_conv6(h5)
    h7 = self.conv7(h6)
    return h7.squeeze()


class CoVAE(CoDis):
  def _convtranspose2d(self, n_in, n_out, kernel_size, stride, padding=0):
    return nn.Sequential(
      nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.LeakyReLU()
      # nn.LeakyReLU(0.1)
    )

  def _pad(self,x_in,pad):
    return nn.functional.pad(x_in, (pad,pad,pad,pad), mode='replicate')

  def __init__(self, ch=32, input_dims = [3, 3], output_dims= [3, 3], image_size=256):
    super(CoVAE, self).__init__()
    self.image_size = image_size
    self.g_en_conv0_a = self._conv2d(input_dims[0], ch, kernel_size=5, stride=2)  # 64
    self.g_en_conv0_b = self._conv2d(input_dims[1], ch, kernel_size=5, stride=2)  # 64
    self.g_en_conv1_a = self._conv2d(ch * 1, ch * 2, kernel_size=5, stride=2)  # 32
    self.g_en_conv1_b = self._conv2d(ch * 1, ch * 2, kernel_size=5, stride=2)  # 32
    self.g_en_conv2 = self._conv2d(ch * 2, ch * 4, kernel_size=5, stride=2)  # 16
    self.g_en_conv3 = self._conv2d(ch * 4, ch * 8, kernel_size=3, stride=2)  # 8
    self.g_en_conv4 = self._conv2d(ch * 8, ch * 16, kernel_size=3, stride=2)  # 4
    self.g_en_conv5 = self._conv2d(ch * 16, ch * 32, kernel_size=3, stride=2)  # 2
    # Latent layers
    self.g_vae2 = GaussianVAE2D(ch * 2, ch * 4, kernel_size=1, stride=1)
    self.g_vae3 = GaussianVAE2D(ch * 4, ch * 8, kernel_size=1, stride=1)
    self.g_vae4 = GaussianVAE2D(ch * 8, ch * 16, kernel_size=1, stride=1)
    self.g_vae5 = GaussianVAE2D(ch * 16, ch * 32, kernel_size=1, stride=1)
    self.g_vae6 = GaussianVAE2D(ch * 32, ch * 64, kernel_size=1, stride=1)
    # Decoder layer #0
    self.g_de_conv5 = self._convtranspose2d(ch * 64, ch * 32, kernel_size=2, stride=2, padding=0)
    self.g_de_conv4 = self._convtranspose2d(ch * 32, ch * 16, kernel_size=4, stride=2, padding=1)
    self.g_de_conv3 = self._convtranspose2d(ch * 16, ch * 8, kernel_size=4, stride=2, padding=1)
    self.g_de_conv2 = self._convtranspose2d(ch * 8, ch * 4, kernel_size=4, stride=2, padding=1)
    self.g_de_conv1_a = self._convtranspose2d(ch * 4, ch * 2, kernel_size=4, stride=2, padding=1)
    self.g_de_conv1_b = self._convtranspose2d(ch * 4, ch * 2, kernel_size=4, stride=2, padding=1)
    self.g_de_conv0_a = self._convtranspose2d(ch * 2, ch * 1, kernel_size=4, stride=2, padding=1)
    self.g_de_conv0_b = self._convtranspose2d(ch * 2, ch * 1, kernel_size=4, stride=2, padding=1)
    self.de_conv_out_a = nn.ConvTranspose2d(ch * 1, output_dims[0], kernel_size=1, stride=1, padding=0)
    self.de_conv_out_b = nn.ConvTranspose2d(ch * 1, output_dims[1], kernel_size=1, stride=1, padding=0)
    self.de_tanh_out_a = nn.Tanh()
    self.de_tanh_out_b = nn.Tanh()

  def forward(self, x_a, x_b, gpu):
    en_h0_a = self.g_en_conv0_a(self._pad(x_a,2))
    en_h0_b = self.g_en_conv0_b(self._pad(x_b,2))
    en_h1_a = self.g_en_conv1_a(self._pad(en_h0_a,2))
    en_h1_b = self.g_en_conv1_b(self._pad(en_h0_b,2))
    en_h1 = torch.cat((en_h1_a, en_h1_b), 0)
    z2, mu2, sd2 = self.g_vae2.sample(en_h1, gpu)
    en_h2 = self.g_en_conv2(self._pad(en_h1,2))
    z3, mu3, sd3 = self.g_vae3.sample(en_h2, gpu)
    en_h3 = self.g_en_conv3(self._pad(en_h2,1))
    z4, mu4, sd4 = self.g_vae4.sample(en_h3, gpu)
    en_h4 = self.g_en_conv4(self._pad(en_h3,1))
    z5, mu5, sd5 = self.g_vae5.sample(en_h4, gpu)
    en_h5 = self.g_en_conv5(self._pad(en_h4, 1))
    z6, mu6, sd6 = self.g_vae6.sample(en_h5, gpu)
    de_h5 = self.g_de_conv5(z6) + z5  # 4
    de_h4 = self.g_de_conv4(de_h5) + z4
    de_h3 = self.g_de_conv3(de_h4) + z3
    de_h2 = self.g_de_conv2(de_h3) + z2
    de_h1_a = self.g_de_conv1_a(de_h2)
    de_h1_b = self.g_de_conv1_b(de_h2)
    de_h0_a = self.g_de_conv0_a(de_h1_a)
    de_h0_b = self.g_de_conv0_b(de_h1_b)
    out_a = self.de_tanh_out_a(self.de_conv_out_a(de_h0_a))
    out_b = self.de_tanh_out_b(self.de_conv_out_b(de_h0_b))
    x_aa, x_ba = torch.split(out_a, x_a.size(0), dim=0)
    x_ab, x_bb = torch.split(out_b, x_a.size(0), dim=0)
    codes = ((mu2, sd2), (mu3, sd3), (mu4, sd4), (mu5, sd5), (mu6, sd6))
    return x_aa, x_ba, x_ab, x_bb, codes


  def translate_a_to_b(self, x_a, gpu):
    en_h0_a = self.g_en_conv0_a(self._pad(x_a,2))
    en_h1_a = self.g_en_conv1_a(self._pad(en_h0_a,2))
    z2, mu2, sd2 = self.g_vae2.sample(en_h1_a, gpu)
    en_h2 = self.g_en_conv2(self._pad(en_h1_a,2))
    z3, mu3, sd3 = self.g_vae3.sample(en_h2, gpu)
    en_h3 = self.g_en_conv3(self._pad(en_h2,1))
    z4, mu4, sd4 = self.g_vae4.sample(en_h3, gpu)
    en_h4 = self.g_en_conv4(self._pad(en_h3,1))
    z5, mu5, sd5 = self.g_vae5.sample(en_h4, gpu)
    en_h5 = self.g_en_conv5(self._pad(en_h4, 1))
    z6, mu6, sd6 = self.g_vae6.sample(en_h5, gpu)
    de_h5 = self.g_de_conv5(mu6) + mu5  # 4
    de_h4 = self.g_de_conv4(de_h5) + mu4
    de_h3 = self.g_de_conv3(de_h4) + mu3
    de_h2 = self.g_de_conv2(de_h3) + mu2
    de_h1_b = self.g_de_conv1_b(de_h2)
    de_h0_b = self.g_de_conv0_b(de_h1_b)
    x_ab = self.de_tanh_out_b(self.de_conv_out_b(de_h0_b))
    codes = ((mu2, sd2), (mu3, sd3), (mu4, sd4), (mu5, sd5), (mu6, sd6))
    return x_ab, codes

  def translate_b_to_a(self, x_b, gpu):
    en_h0_b = self.g_en_conv0_b(self._pad(x_b,2))
    en_h1_b = self.g_en_conv1_b(self._pad(en_h0_b,2))
    z2, mu2, sd2 = self.g_vae2.sample(en_h1_b, gpu)
    en_h2 = self.g_en_conv2(self._pad(en_h1_b,2))
    z3, mu3, sd3 = self.g_vae3.sample(en_h2, gpu)
    en_h3 = self.g_en_conv3(self._pad(en_h2,1))
    z4, mu4, sd4 = self.g_vae4.sample(en_h3, gpu)
    en_h4 = self.g_en_conv4(self._pad(en_h3,1))
    z5, mu5, sd5 = self.g_vae5.sample(en_h4, gpu)
    en_h5 = self.g_en_conv5(self._pad(en_h4, 1))
    z6, mu6, sd6 = self.g_vae6.sample(en_h5, gpu)
    de_h5 = self.g_de_conv5(mu6) + mu5  # 4
    de_h4 = self.g_de_conv4(de_h5) + mu4
    de_h3 = self.g_de_conv3(de_h4) + mu3
    de_h2 = self.g_de_conv2(de_h3) + mu2
    de_h1_a = self.g_de_conv1_a(de_h2)
    de_h0_a = self.g_de_conv0_a(de_h1_a)
    x_ba = self.de_tanh_out_a(self.de_conv_out_a(de_h0_a))
    codes = ((mu2, sd2), (mu3, sd3), (mu4, sd4), (mu5, sd5), (mu6, sd6))
    return x_ba, codes


#############################################################################
# Below are for digit classification
#############################################################################

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
  def _conv2d(self, n_in, n_out, kernel_size, stride, padding):
    return nn.Sequential(
      nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.BatchNorm2d(n_out, affine=False),
      Bias2d(n_out),
      nn.LeakyReLU()
    )

  def _convtranspose2d(self, n_in, n_out, kernel_size, stride, padding):
    return nn.Sequential(
      nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding),
      nn.BatchNorm2d(n_out, affine=False),
      Bias2d(n_out),
      nn.LeakyReLU()
    )

  def __init__(self, ch=32, input_dim_a=3, output_dim_a=3, input_dim_b=1, output_dim_b=1):
    super(CoVAE32x32, self).__init__()
    # Encoder layer #0
    self.g_en_conv0_a = self._conv2d(input_dim_a, ch, kernel_size=5, stride=2, padding=2)
    self.g_en_conv0_b = self._conv2d(input_dim_b, ch, kernel_size=5, stride=2, padding=2)
    self.g_en_conv1 = self._conv2d(ch * 1, ch * 2, kernel_size=5, stride=2, padding=2)
    self.g_en_conv2 = self._conv2d(ch * 2, ch * 4, kernel_size=8, stride=1, padding=0)
    self.g_en_conv3 = self._conv2d(ch * 4, ch * 8, kernel_size=1, stride=1, padding=0)
    # Latent layer
    self.g_vae = GaussianVAE2DFull(ch * 8, ch * 8, kernel_size=1, stride=1)
    # Decoder layer #0
    self.g_de_conv0 = self._convtranspose2d(ch * 8, ch * 8, kernel_size=4, stride=2, padding=0)
    self.g_de_conv1 = self._convtranspose2d(ch * 8, ch * 4, kernel_size=4, stride=2, padding=1)
    self.g_de_conv2 = self._convtranspose2d(ch * 4, ch * 2, kernel_size=4, stride=2, padding=1)
    self.g_de_conv3_a = self._convtranspose2d(ch * 2, ch * 1, kernel_size=4, stride=2, padding=1)
    self.g_de_conv3_b = self._convtranspose2d(ch * 2, ch * 1, kernel_size=4, stride=2, padding=1)
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
    z, mu, sd = self.g_vae.sample(en_h3, gpu)
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