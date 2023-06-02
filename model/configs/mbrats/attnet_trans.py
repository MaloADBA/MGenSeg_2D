# Translation model for AttENT
from collections import OrderedDict
import torch
from torch import nn
from torch.nn.utils import remove_spectral_norm
from torch.functional import F
from torch.nn.utils import spectral_norm
import numpy as np
from fcn_maker.model import assemble_resunet
from fcn_maker.loss import dice_loss
from model.attnet_translation import translation_model
    

def build_model(lambda_disc=1, 
                lambda_cyc=1):
    
    # Rescale lambdas if a sum is enforced.
    lambda_scale = 1.
    
    lambda_sum = (lambda_disc+lambda_cyc)
    
    lambda_scale = 1/lambda_sum
    
    image_size = (1, 240, 120)
    
    submodel = {
        'generator_S'           : Generator(input_nc=1, output_nc=1),
        'generator_T'           : Generator(input_nc=1, output_nc=1),
        'disc_S'            : Discriminator(input_nc=1),
        'disc_T'            : Discriminator(input_nc=1)}
    
    model = translation_model(**submodel,
                               loss_gan='hinge',
                               rng=np.random.RandomState(1234),
                               lambda_disc=lambda_scale*lambda_disc,
                               lambda_cyc=lambda_scale*lambda_cyc)
    
    return OrderedDict((
        ('G', model),
        ('D', nn.ModuleList([model.separate_networks['disc_S'],
                             model.separate_networks['disc_T']]))
        ))

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Attention_block(nn.Module):
    """
    refer from attention u-net (https://arxiv.org/abs/1804.03999)
    """
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        # self.ep=epoch
    def forward(self,g,x):
        # down-sampling g conv used as gate signal
        g1 = self.W_g(g)
        # up-sampling l conv
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        result = x*psi*2
        # return re-weigted output
        return result
    
class Generator(nn.Module):
    """
    attention_cyclegan_generator
    replace ConvTranspose2d with Up+Conv
    """
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        # Initial convolution block 
        self.initial_block = nn.Sequential(
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True))

        # Downsampling
        in_features = 64
        out_features = in_features*2
        # for _ in range(2):
        self.down_sampling1=nn.Sequential(
                    nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features*2
        self.down_sampling2=nn.Sequential(
                    nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features*2

        # Residual blocks
        tmp=[]
        for _ in range(n_residual_blocks):
            tmp += [ResidualBlock(in_features)]
        self.ResidualBlock = nn.Sequential(*tmp)
        # Upsampling
        out_features = in_features//2
        self.up_sampling1=nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(in_features*2, out_features,  1, 1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features//2

        self.up_sampling2=nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),#nearest
                    nn.Conv2d(in_features*2, out_features, 1, 1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True) )
        in_features = out_features
        out_features = in_features//2

        # Output layer
        self.Output_layer=nn.Sequential( nn.ReflectionPad2d(3),
                    nn.Conv2d(64*2, output_nc, 7),
                    nn.Tanh() )


        self.Att3 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Att2 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Att1 = Attention_block(F_g=64, F_l=64, F_int=32)
        # self.model = nn.Sequential(*model)

    def forward(self, x):
        # print('x',x.shape)
        initial=self.initial_block(x)
        # print('initial',initial.shape)
        down_sampling1=self.down_sampling1(initial)
        # print('down_sampling1',down_sampling1.shape)
        down_sampling2=self.down_sampling2(down_sampling1)
        # print('down_sampling2',down_sampling2.shape)
        res_out = self.ResidualBlock(down_sampling2)
        # print('res_out',res_out.shape)
        att3=self.Att3(g=res_out,x=down_sampling2)
        sum_level3 = torch.cat((att3,res_out),dim=1)
        up_sampling1 =self.up_sampling1(sum_level3)
        # print('up_sampling1',up_sampling1.shape)
        att2=self.Att2(g=up_sampling1,x=down_sampling1)
        sum_level2 = torch.cat((att2,up_sampling1),dim=1)
        up_sampling2 =self.up_sampling2(sum_level2)
        # print('up_sampling2',up_sampling2.shape)
        att1=self.Att1(g=up_sampling2,x=initial)
        sum_level1 = torch.cat((att1,up_sampling2),dim=1)
        output = self.Output_layer(sum_level1)
        # print('output',output.shape)
        return output
    
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    
