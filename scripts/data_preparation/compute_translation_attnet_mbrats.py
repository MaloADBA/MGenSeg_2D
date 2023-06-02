import os
import argparse
from collections import OrderedDict
import numpy as np
import scipy.misc
from scipy import ndimage
import SimpleITK as sitk
import h5py
import imageio
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
"""
AttENT is a two stages method : modality translation + segmentation
Once the translation model trained this file can be used to generate a synthetic target dataset from source data
Modify the paths below
"""

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


save_path = '/lustre04/scratch/maloadba/experiments/mbrats_attent_trans/'

# Where the training models are stored 
model_paths = ['t1_flair/', 't1_t1ce/', 't1_t2/', 't1ce_flair/', 't1ce_t2/', 't2_flair/']
numbers = ['1/', '2/', '1/', '2/', '2/', '1/']

# Modality pairs for the corresponding trained models 
mod_1 = ['t1', 't1', 't1', 't1ce', 't1ce', 't2']
mod_2 = ['flair', 't1ce', 't2', 'flair', 't2', 'flair']

for j in range(6):
    saved_state_dict = torch.load(save_path+model_paths[j]+numbers[j]+'state_dict_100.pth')
        
    generator_S = Generator(input_nc=1, output_nc=1)
    generator_T = Generator(input_nc=1, output_nc=1)
    
    generator_S.cuda()
    generator_T.cuda()
    
    new_params = generator_S.state_dict().copy()
    for i in saved_state_dict['G']['model_state'].keys():
        i_parts = i.split('.')
        if i_parts[0] == 'generator_S':
            new_params['.'.join(i_parts[1:])] = saved_state_dict['G']['model_state'][i]
    generator_S.load_state_dict(new_params)
    
    new_params = generator_T.state_dict().copy()
    for i in saved_state_dict['G']['model_state'].keys():
        i_parts = i.split('.')
        if i_parts[0] == 'generator_T':
            new_params['.'.join(i_parts[1:])] = saved_state_dict['G']['model_state'][i]
    generator_T.load_state_dict(new_params)
    
    h5py_file = h5py.File(save_path+'data/'+model_paths[j][:-1]+'.h5', mode='w')
    data = h5py.File("/home/maloadba/Data/multimodal_brats/data.h5", mode='r')
    for case_id in data.keys():
        dic = OrderedDict()
        print("Processing case "+case_id)
        group_p = h5py_file.create_group(case_id)
        if not len(data[case_id]['sick_'+mod_1[j]]):
            dic['translated_sick_1'] = np.zeros((1,1,240,120))[[],:]
            dic['translated_sick_2'] = np.zeros((1,1,240,120))[[],:]
            dic['sick_1'] = np.zeros((1,1,240,120))[[],:]
            dic['sick_2'] = np.zeros((1,1,240,120))[[],:]
            dic['segmentation'] = np.zeros((1,1,240,120))[[],:]
        else:
            translated_t1 = []
            translated_t2 = []
            t1 = []
            t2 = []
            seg = []
            for i in range(len(data[case_id]['sick_'+mod_1[j]])):
                t1.append(data[case_id]['sick_'+mod_1[j]][i,0])
                t2.append(data[case_id]['sick_'+mod_2[j]][i,0])
                seg.append(data[case_id]['segmentation'][i,0])
                t1_sample = torch.unsqueeze(torch.from_numpy(np.array(np.clip(data[case_id]['sick_'+mod_1[j]][i].astype(np.float32), -1., 1.))).cuda(),dim=0)
                t2_sample = torch.unsqueeze(torch.from_numpy(np.array(np.clip(data[case_id]['sick_'+mod_2[j]][i].astype(np.float32), -1., 1.))).cuda(),dim=0)
                trans_t2 = generator_S(t2_sample)
                trans_t1 = generator_T(t1_sample)
                translated_t2.append(np.squeeze(trans_t2.cpu().detach().numpy()))
                translated_t1.append(np.squeeze(trans_t1.cpu().detach().numpy()))
            dic['sick_1'] = np.expand_dims(np.array(t1),axis=1)
            dic['sick_2'] = np.expand_dims(np.array(t2),axis=1)
            dic['translated_sick_1'] = np.expand_dims(np.array(translated_t1),axis=1)
            dic['translated_sick_2'] = np.expand_dims(np.array(translated_t2),axis=1)        
            dic['segmentation'] = np.expand_dims(np.array(seg),axis=1)
    
        for key in dic.keys() :
            if len(dic[key])==0:
                kwargs = {}
            else:
                kwargs = {'chunks': (1,)+dic[key].shape[1:],
                          'compression': 'lzf'}
            group_p.create_dataset(key,
                                   shape=dic[key].shape,
                                   data=dic[key],
                                   dtype=dic[key].dtype,
                                   **kwargs)
    

