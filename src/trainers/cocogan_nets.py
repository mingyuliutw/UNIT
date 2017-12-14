"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from common_net import *

class COCOSharedDis(nn.Module):
  def __init__(self, params):
    super(COCOSharedDis, self).__init__()
    ch = params['ch']
    input_dim_a = params['input_dim_a']
    input_dim_b = params['input_dim_b']
    n_front_layer = params['n_front_layer']
    n_shared_layer = params['n_shared_layer']
    self.model_A, tch = self._make_front_net(ch, input_dim_a, n_front_layer, n_shared_layer==0)
    self.model_B, tch = self._make_front_net(ch, input_dim_b, n_front_layer, n_shared_layer==0)
    self.model_S = self._make_shared_net(tch, n_shared_layer)

  def _make_front_net(self, ch, input_dim, n_layer, add_classifier_layer=False):
    model = []
    model += [LeakyReLUConv2d(input_dim, ch, kernel_size=7, stride=2, padding=3)] #16
    tch = ch
    for i in range(1, n_layer):
      model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)] # 8
      tch *= 2
    if add_classifier_layer:
      model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]  # 1
    return nn.Sequential(*model), tch

  def _make_shared_net(self, ch, n_layer):
    model = []
    if n_layer == 0:
      return nn.Sequential(*model)
    tch = ch
    for i in range(0, n_layer):
      model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)] # 8
      tch *= 2
    model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]  # 1
    return nn.Sequential(*model)

  def cuda(self,gpu):
    self.model_A.cuda(gpu)
    self.model_B.cuda(gpu)
    self.model_S.cuda(gpu)

  def forward(self, x_A, x_B):
    out_A = self.model_S(self.model_A(x_A))
    out_A = out_A.view(-1)
    outs_A = []
    outs_A.append(out_A)
    out_B = self.model_S(self.model_B(x_B))
    out_B = out_B.view(-1)
    outs_B = []
    outs_B.append(out_B)
    return outs_A, outs_B

class COCOMsDis(nn.Module):
# Multi-scale discriminator architecture
# This one applies Gaussian smoothing before down-sampling
  def __init__(self, params):
    super(COCOMsDis, self).__init__()
    ch = params['ch']
    input_dim_a = params['input_dim_a']
    input_dim_b = params['input_dim_b']
    n_layer = params['n_layer']
    self.downsampler = GaussianSmoother(3)
    self.model_1_A = self._make_net(ch, input_dim_a, n_layer)
    self.model_2_A = self._make_net(ch, input_dim_a, n_layer)
    self.model_4_A = self._make_net(ch, input_dim_a, n_layer)
    self.model_1_B = self._make_net(ch, input_dim_b, n_layer)
    self.model_2_B = self._make_net(ch, input_dim_b, n_layer)
    self.model_4_B = self._make_net(ch, input_dim_b, n_layer)

  def _make_net(self, ch, input_dim, n_layer):
    model = []
    model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1)] #16
    tch = ch
    for i in range(1, n_layer):
      model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)] # 8
      tch *= 2
    model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]  # 1
    return nn.Sequential(*model)

  def cuda(self,gpu):
    self.model_1_A.cuda(gpu)
    self.model_2_A.cuda(gpu)
    self.model_4_A.cuda(gpu)
    self.model_1_B.cuda(gpu)
    self.model_2_B.cuda(gpu)
    self.model_4_B.cuda(gpu)
    self.downsampler.cuda(gpu)

  def forward(self, x_A, x_B):
    return self.forward_A(x_A), self.forward_B(x_B)

  def forward_A(self, x):
    x2 = self.downsample(x)
    x4 = self.downsample(x2)
    out_1 = self.model_1_A(x)
    out_2 = self.model_2_A(x2)
    out_4 = self.model_4_A(x4)
    out_1 = out_1.view(-1)
    out_2 = out_2.view(-1)
    out_4 = out_4.view(-1)
    return out_1, out_2, out_4

  def forward_B(self, x):
    x2 = self.downsample(x)
    x4 = self.downsample(x2)
    out_1 = self.model_1_B(x)
    out_2 = self.model_2_B(x2)
    out_4 = self.model_4_B(x4)
    out_1 = out_1.view(-1)
    out_2 = out_2.view(-1)
    out_4 = out_4.view(-1)
    return out_1, out_2, out_4

  def downsample(self, x):
    x2 = self.downsampler(x)
    h = x2.size(2)
    w = x2.size(2)
    x3 = x2[:, :, 0:h:2, 0:w:2]
    return x3

class COCODis(nn.Module):
  def __init__(self, params):
    super(COCODis, self).__init__()
    ch = params['ch']
    input_dim_a = params['input_dim_a']
    input_dim_b = params['input_dim_b']
    n_layer = params['n_layer']
    self.model_A = self._make_net(ch, input_dim_a, n_layer)
    self.model_B = self._make_net(ch, input_dim_b, n_layer)

  def _make_net(self, ch, input_dim, n_layer):
    model = []
    model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1)] #16
    tch = ch
    for i in range(1, n_layer):
      model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)] # 8
      tch *= 2
    model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]  # 1
    return nn.Sequential(*model)

  def cuda(self,gpu):
    self.model_A.cuda(gpu)
    self.model_B.cuda(gpu)

  def forward(self, x_A, x_B):
    out_A = self.model_A(x_A)
    out_A = out_A.view(-1)
    outs_A = []
    outs_A.append(out_A)
    out_B = self.model_B(x_B)
    out_B = out_B.view(-1)
    outs_B = []
    outs_B.append(out_B)
    return outs_A, outs_B

class COCOResGen(nn.Module):
# In COCOResGen, the first convolutional layers in the encoders are based on LeakyReLU with no normalization layers.
# But all the other non residual-block based layers are based on ReLU with Instance Norm activation.
  def __init__(self, params):
    super(COCOResGen, self).__init__()
    input_dim_a = params['input_dim_a']
    input_dim_b = params['input_dim_b']
    ch = params['ch']
    n_enc_front_blk  = params['n_enc_front_blk']
    n_enc_res_blk    = params['n_enc_res_blk']
    n_enc_shared_blk = params['n_enc_shared_blk']
    n_gen_shared_blk = params['n_gen_shared_blk']
    n_gen_res_blk    = params['n_gen_res_blk']
    n_gen_front_blk  = params['n_gen_front_blk']
    if 'res_dropout_ratio' in params.keys():
      res_dropout_ratio = params['res_dropout_ratio']
    else:
      res_dropout_ratio = 0
    ##############################################################################
    # BEGIN of ENCODERS
    # Convolutional front-end
    encA = []
    encB = []
    encA += [LeakyReLUConv2d(input_dim_a, ch, kernel_size=7, stride=1, padding=3)]
    encB += [LeakyReLUConv2d(input_dim_b, ch, kernel_size=7, stride=1, padding=3)]
    tch = ch
    for i in range(1,n_enc_front_blk):
      encA += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      encB += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      tch *= 2
    # Residual-block back-end
    for i in range(0, n_enc_res_blk):
      encA += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
      encB += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # END of ENCODERS
    ##############################################################################

    ##############################################################################
    # BEGIN of SHARED LAYERS
    # Shared residual-blocks
    enc_shared = []
    for i in range(0, n_enc_shared_blk):
      enc_shared += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    enc_shared += [GaussianNoiseLayer()]
    # Shared residual-blocks
    dec_shared = []
    for i in range(0, n_gen_shared_blk):
      dec_shared += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # END of SHARED LAYERS
    ##############################################################################

    ##############################################################################
    # BEGIN of DECODERS
    decA = []
    decB = []
    # Residual-block front-end
    for i in range(0, n_gen_res_blk):
      decA += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
      decB += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # Convolutional back-end
    for i in range(0, n_gen_front_blk-1):
      decA += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
      decB += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
      tch = tch//2
    decA += [nn.ConvTranspose2d(tch, input_dim_a, kernel_size=1, stride=1, padding=0)]
    decB += [nn.ConvTranspose2d(tch, input_dim_b, kernel_size=1, stride=1, padding=0)]
    decA += [nn.Tanh()]
    decB += [nn.Tanh()]
    # END of DECODERS
    ##############################################################################
    self.encode_A = nn.Sequential(*encA)
    self.encode_B = nn.Sequential(*encB)
    self.enc_shared = nn.Sequential(*enc_shared)
    self.dec_shared = nn.Sequential(*dec_shared)
    self.decode_A = nn.Sequential(*decA)
    self.decode_B = nn.Sequential(*decB)

  def forward(self, x_A, x_B):
    out = torch.cat((self.encode_A(x_A), self.encode_B(x_B)), 0)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out_A = self.decode_A(out)
    out_B = self.decode_B(out)
    x_Aa, x_Ba = torch.split(out_A, x_A.size(0), dim=0)
    x_Ab, x_Bb = torch.split(out_B, x_A.size(0), dim=0)
    return x_Aa, x_Ba, x_Ab, x_Bb, shared

  def forward_a2b(self, x_A):
    out = self.encode_A(x_A)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out = self.decode_B(out)
    return out, shared

  def forward_b2a(self, x_B):
    out = self.encode_B(x_B)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out = self.decode_A(out)
    return out, shared

# In COCOResGen2, all the non residual-block layers are based on LeakyReLU with no normalization layers.
class COCOResGen2(nn.Module):
  def __init__(self, params):
    super(COCOResGen2, self).__init__()
    input_dim_a = params['input_dim_a']
    input_dim_b = params['input_dim_b']
    ch = params['ch']
    n_enc_front_blk  = params['n_enc_front_blk']
    n_enc_res_blk    = params['n_enc_res_blk']
    n_enc_shared_blk = params['n_enc_shared_blk']
    n_gen_shared_blk = params['n_gen_shared_blk']
    n_gen_res_blk    = params['n_gen_res_blk']
    n_gen_front_blk  = params['n_gen_front_blk']
    if 'res_dropout_ratio' in params.keys():
      res_dropout_ratio = params['res_dropout_ratio']
    else:
      res_dropout_ratio = 0

    ##############################################################################
    # BEGIN of ENCODERS
    # Convolutional front-end
    encA = []
    encB = []
    encA += [LeakyReLUConv2d(input_dim_a, ch, kernel_size=7, stride=1, padding=3)]
    encB += [LeakyReLUConv2d(input_dim_b, ch, kernel_size=7, stride=1, padding=3)]
    tch = ch
    for i in range(1,n_enc_front_blk):
      encA += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      encB += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      tch *= 2
    # Residual-block back-end
    for i in range(0, n_enc_res_blk):
      encA += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
      encB += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # END of ENCODERS
    ##############################################################################

    ##############################################################################
    # BEGIN of SHARED LAYERS
    # Shared residual-blocks
    enc_shared = []
    for i in range(0, n_enc_shared_blk):
      enc_shared += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    enc_shared += [GaussianNoiseLayer()]
    dec_shared = []
    for i in range(0, n_gen_shared_blk):
      dec_shared += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # END of SHARED LAYERS
    ##############################################################################

    ##############################################################################
    # BEGIN of DECODERS
    decA = []
    decB = []
    # Residual-block front-end
    for i in range(0, n_gen_res_blk):
      decA += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
      decB += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # Convolutional back-end
    for i in range(0, n_gen_front_blk-1):
      decA += [LeakyReLUConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
      decB += [LeakyReLUConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
      tch = tch//2
    decA += [nn.ConvTranspose2d(tch, input_dim_a, kernel_size=1, stride=1, padding=0)]
    decB += [nn.ConvTranspose2d(tch, input_dim_b, kernel_size=1, stride=1, padding=0)]
    decA += [nn.Tanh()]
    decB += [nn.Tanh()]
    # END of DECODERS
    ##############################################################################
    self.encode_A = nn.Sequential(*encA)
    self.encode_B = nn.Sequential(*encB)
    self.enc_shared = nn.Sequential(*enc_shared)
    self.dec_shared = nn.Sequential(*dec_shared)
    self.decode_A = nn.Sequential(*decA)
    self.decode_B = nn.Sequential(*decB)

  def forward(self, x_A, x_B):
    out = torch.cat((self.encode_A(x_A), self.encode_B(x_B)), 0)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out_A = self.decode_A(out)
    out_B = self.decode_B(out)
    x_Aa, x_Ba = torch.split(out_A, x_A.size(0), dim=0)
    x_Ab, x_Bb = torch.split(out_B, x_A.size(0), dim=0)
    return x_Aa, x_Ba, x_Ab, x_Bb, shared

  def forward_a2b(self, x_A):
    out = self.encode_A(x_A)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out = self.decode_B(out)
    return out, shared

  def forward_b2a(self, x_B):
    out = self.encode_B(x_B)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out = self.decode_A(out)
    return out, shared
