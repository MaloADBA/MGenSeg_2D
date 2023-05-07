# UAGAN model

from collections import OrderedDict
import torch
from torch import nn
from torch.nn import init
from torch.nn.utils import remove_spectral_norm
from torch.functional import F
from torch.nn.utils import spectral_norm
import numpy as np
import functools
from fcn_maker.model import assemble_resunet
from fcn_maker.loss import dice_loss
from model.common.losses import (bce,
                            cce,
                            dist_ratio_mse_abs,
                            gan_objective,
                            mae,
                            mse)
from model.uagan_segmentation import segmentation_model
    

def build_model(lambda_seg=100,
                lambda_id=10,
                lambda_disc=1, 
                lambda_class=1,
                lambda_gp=10):
                
    lambda_scale = 1.
    lambda_sum = ( lambda_seg
                   +lambda_id
                   +lambda_disc
                   +lambda_class
                   +lambda_gp)
    lambda_scale = 1./lambda_sum    
    
    submodel = {
        'generator'            : UAGAN(1, 1, 3, 1, feature_maps=64, levels=3, norm_type='instance', use_dropout=True),
        'disc'            : Discriminator(image_size=120, conv_dim=64, c_dim=2, repeat_num=6),
        'classifieur'     : Classifior()}
        
    model = segmentation_model(**submodel,
                               loss_seg=dice_loss([1]),
                               rng=np.random.RandomState(1234),
                               lambda_seg=lambda_scale*lambda_seg,
                               lambda_disc=lambda_scale*lambda_disc,
                               lambda_id=lambda_scale*lambda_id,
                               lambda_class=lambda_scale*lambda_class,
                               lambda_gp=lambda_scale*lambda_gp)

    return OrderedDict((
        ('G', model),
        ('D', nn.ModuleList([model.separate_networks['disc'],
                             model.separate_networks['classifieur']]))
        ))

class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, c_dim=4, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = [nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1),
                  nn.LeakyReLU(0.001)]

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim *= 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        
        self.main = nn.Sequential(*layers)
        
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        #self.lin = nn.Linear(6144, 2, bias=True)

    def forward(self, inputs):
        h = self.main(inputs)
        out_src = self.conv1(h)
        #out_cls = self.lin(h.view(h.size(0), h.size(1)*h.size(2)))
        return out_src
        
class Classifior(nn.Module):
    def __init__(self):
        super(Classifior, self).__init__()
        self.lin = nn.Linear(6144, 2, bias=True)
    def forward(self, inputs):
        out_cls = self.lin(inputs.view(inputs.size(0), inputs.size(1)*inputs.size(2)))
        return out_cls

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                       nn.Sigmoid())

    def forward(self, inputs):
        return self.attention(inputs)


class Decoder(nn.Module):
    def __init__(self, out_channels=2, feature_maps=64, levels=4, norm_type='instance', bias=True):
        super(Decoder, self).__init__()

        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()

        for i in range(levels):
            att = AttentionBlock(2 ** (levels - i - 1) * feature_maps, 2 ** (levels - i - 1) * feature_maps)
            self.features.add_module('atten%d' % (i + 1), att)

            w = ConvNormRelu(2**(levels-i-1) * feature_maps, 2**(levels-i-1) * feature_maps,
                                 norm_type=norm_type, bias=bias)
            self.features.add_module('w%d' % (i + 1), w)

            upconv = UNetUpSamplingBlock(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                         deconv=True, bias=bias)
            self.features.add_module('upconv%d' % (i + 1), upconv)

            conv_block = UNetConvBlock(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                       norm_type=norm_type, bias=bias)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

        self.score = nn.Conv2d(feature_maps, out_channels, kernel_size=1)

    def forward(self, inputs, self_encoder_outputs, aux_encoder_outputs):
        decoder_outputs = []
        self_encoder_outputs.reverse()
        aux_encoder_outputs.reverse()

        outputs = inputs
        for i in range(self.levels):

            w = getattr(self.features, 'w%d' % (i+1))(aux_encoder_outputs[i])
            a = getattr(self.features, 'atten%d' % (i+1))(aux_encoder_outputs[i])
            aux_encoder_output = w.mul(a)
            
            fuse_encoder_output = aux_encoder_output + self_encoder_outputs[i]

            outputs = getattr(self.features, 'upconv%d' % (i+1))(fuse_encoder_output, outputs)
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)

            decoder_outputs.append(outputs)

        self_encoder_outputs.reverse()
        aux_encoder_outputs.reverse()

        return decoder_outputs, self.score(outputs)


class UAGAN(nn.Module):
    def __init__(self, seg_in, seg_out, syn_in, syn_out, feature_maps=64, levels=4, norm_type='instance',
                 bias=True, use_dropout=True):
        super(UAGAN, self).__init__()

        self.seg_encoder = UNetEncoder(seg_in, feature_maps, levels, norm_type, use_dropout, bias=bias,
                                       use_last_block=False)
        self.syn_encoder = UNetEncoder(syn_in, feature_maps, levels, norm_type, use_dropout, bias=bias,
                                       use_last_block=False)
        self.center_conv = UNetConvBlock(2**(levels-1) * feature_maps, 2**levels * feature_maps)
        self.seg_decoder = Decoder(seg_out, feature_maps, levels, norm_type, bias=bias)
        self.syn_decoder = Decoder(syn_out, feature_maps, levels, norm_type, bias=bias)

    def forward(self, seg_inputs, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, seg_inputs.size(2), seg_inputs.size(3))
        
        syn_inputs = torch.cat([seg_inputs, c], dim=1)
        

        seg_encoder_outputs, seg_output = self.seg_encoder(seg_inputs)

        syn_encoder_outputs, syn_output = self.syn_encoder(syn_inputs)

        seg_bottleneck = self.center_conv(seg_output)
        syn_bottleneck = self.center_conv(syn_output)

        _, seg_score = self.seg_decoder(seg_bottleneck, seg_encoder_outputs, syn_encoder_outputs)
        _, syn_score = self.syn_decoder(syn_bottleneck, syn_encoder_outputs, seg_encoder_outputs)
        return torch.sigmoid(seg_score), torch.tanh(syn_score)
    
class ConvNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding='SAME', bias=True, dilation=1, norm_type='instance'):

        super(ConvNormRelu, self).__init__()
        norm = nn.BatchNorm2d if norm_type == 'batch' else nn.InstanceNorm2d
        if padding == 'SAME':
            p = kernel_size // 2
        else:
            p = 0

        self.unit = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                            padding=p, stride=stride, bias=bias, dilation=dilation),
                                  norm(out_channels),
                                  nn.LeakyReLU(0.01))

    def forward(self, inputs):
        return self.unit(inputs)


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='SAME', norm_type='instance', bias=True):
        super(UNetConvBlock, self).__init__()

        self.conv1 = ConvNormRelu(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                  norm_type=norm_type, bias=bias)
        self.conv2 = ConvNormRelu(out_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                  norm_type=norm_type, bias=bias)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UNetUpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, bias=True):
        super(UNetUpSamplingBlock, self).__init__()
        self.deconv = deconv
        if self.deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, 1, bias=False)
            )
            
    def forward(self, *inputs):
        if len(inputs) == 2:
            return self.forward_concat(inputs[0], inputs[1])
        else:
            return self.forward_standard(inputs[0])

    def forward_concat(self, inputs1, inputs2):
        return torch.cat([inputs1, self.up(inputs2)], 1)

    def forward_standard(self, inputs):
        return self.up(inputs)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm_type='instance', use_dropout=True, bias=True,
                 use_last_block=True):
        super(UNetEncoder, self).__init__()

        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()
        self.use_dropout = use_dropout
        self.use_last_block = use_last_block
        if self.use_dropout:
            self.dropout = nn.Dropout()

        in_features = in_channels
        for i in range(levels):
            out_features = (2**i) * feature_maps

            conv_block = UNetConvBlock(in_features, out_features, norm_type=norm_type, bias=bias)
            self.features.add_module('convblock%d' % (i+1), conv_block)

            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i+1), pool)

            in_features = out_features
        if use_last_block:
            self.center_conv = UNetConvBlock(2**(levels-1) * feature_maps, 2**levels * feature_maps)

    def forward(self, inputs):
        encoder_outputs = []

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)
            if i == self.levels - 1 and self.use_dropout:
                encoder_outputs.append(self.dropout(outputs))
            else:
                encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i+1))(outputs)
        if self.use_last_block:
            outputs = self.center_conv(outputs)
        return encoder_outputs, outputs


class UNetDecoder(nn.Module):
    def __init__(self, out_channels, feature_maps=64, levels=4, norm_type='instance', bias=True):
        super(UNetDecoder, self).__init__()

        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()

        for i in range(levels):
            upconv = UNetUpSamplingBlock(2**(levels-i) * feature_maps, 2**(levels-i-1) * feature_maps, deconv=True,
                                         bias=bias)
            self.features.add_module('upconv%d' % (i+1), upconv)

            conv_block = UNetConvBlock(2**(levels-i) * feature_maps, 2**(levels-i-1) * feature_maps,
                                       norm_type=norm_type, bias=bias)
            self.features.add_module('convblock%d' % (i+1), conv_block)

        self.score = nn.Conv2d(feature_maps, out_channels, kernel_size=1, bias=bias)

    def forward(self, inputs, encoder_outputs):
        decoder_outputs = []
        encoder_outputs.reverse()

        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'upconv%d' % (i+1))(encoder_outputs[i], outputs)
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)
            decoder_outputs.append(outputs)
        encoder_outputs.reverse()
        return decoder_outputs, self.score(outputs)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, feature_map=64, levels=4, norm_type='instance',
                 use_dropout=True, bias=True):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(in_channels, feature_map, levels, norm_type, use_dropout, bias=bias)
        self.decoder = UNetDecoder(out_channels, feature_map, levels, norm_type, bias=bias)

    def forward(self, inputs):
        encoder_outputs, final_output = self.encoder(inputs)
        decoder_outputs, outputs = self.decoder(final_output, encoder_outputs)
        return outputs