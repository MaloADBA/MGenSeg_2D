
from __future__ import (print_function,
                        division)
import os
import argparse
from collections import OrderedDict
#from concurrent.futures import ThreadPoolExecutor
import threading
try:
    import queue            # python 3
except ImportError:
    import Queue as queue   # python 2

import numpy as np
import scipy.misc
from scipy import ndimage
import SimpleITK as sitk
import h5py
import imageio
import random


def parse():
    parser = argparse.ArgumentParser(description="Prepare BRATS data. Loads "
        "BRATS 2020 data and stores volume slices in an HDF5 archive. "
        "Slices are organized as a group per patient, containing three "
        "groups: \'sick\', \'healthy\', and \'segmentations\' for each of the 2 modalities."
        "Sick cases contain any anomolous class, healthy cases contain no anomalies, and "
        "segmentations are the segmentations of the anomalies. Each group "
        "contains subgroups for each of the three orthogonal planes along "
        "which slices are extracted. For each case, MRI sequences are stored "
        "in a single volume")
    parser.add_argument('--data_dir',
                        help="The directory containing the preprocessed CrossModa data. ",
                        required=True, type=str)
    parser.add_argument('--save_to',
                        help="Path to save the HDF5 file to.",
                        required=True, type=str)
    parser.add_argument('--num_threads',
                        help="The number of parallel threads to execute.",
                        required=False, type=int, default=None)
    return parser.parse_args()

def resize(image, size, interpolator=sitk.sitkLinear):
    new_spacing = [x*y/z for x, y, z in zip(
                   image.GetSpacing(),
                   image.GetSize(),
                   size)]
                   
    sitk_out = sitk.Resample(image,
                             size,
                             sitk.Transform(),
                             interpolator,
                             image.GetOrigin(),
                             new_spacing,
                             image.GetDirection(),
                             0,
                             image.GetPixelID())
    return sitk_out

def resampleDCM(itk_imgs, new_spacing=[1,1,1], is_label=False, new_size=None, new_origin=None):
    """ resample SimpleITK Image variable
    itk_imgs:    SimpleITK Image variable for input image to be resampled
    new_spacing: output spacing
    is_label:    whether to resample a image or a label
    new_size:    output size
    new_origin:  output origin
    """
    resample = sitk.ResampleImageFilter()
    
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    resample.SetOutputDirection(itk_imgs.GetDirection())
    if new_origin is None:
        new_origin = itk_imgs.GetOrigin()
    resample.SetOutputOrigin(new_origin)
    resample.SetOutputSpacing(new_spacing)

    ori_size = np.array(itk_imgs.GetSize(), dtype=np.int32)
    ori_spacing = np.asarray(itk_imgs.GetSpacing())
    if new_size is None:
        new_size = ori_size*(ori_spacing/new_spacing)
        new_size = np.ceil(new_size).astype(np.int32)
        new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)

    out = resample.Execute(itk_imgs)
    return out, new_size, new_origin

def window_img(image, window_center = 50, window_width = 100):
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    
    image[image < img_min] = img_min
    image[image > img_max] = img_max
    return image

def data_loader(data_dir):
    for dn in sorted(os.listdir(data_dir+'/Images/')):
            
        path_CT = os.path.join(data_dir+'/Images/', dn)
        path_seg = os.path.join(data_dir+'/Masks/', dn.replace('.nii.gz','_segmentation_ANW_1.nii.gz'))
        path_brain = os.path.join(data_dir+'/Brain_masks/', dn)
        
        # Load files.
        vol = sitk.ReadImage(path_CT)
        seg = sitk.ReadImage(path_seg)
        brain = sitk.ReadImage(path_brain)
        
        spacing = vol.GetSpacing()

        
        vol, _, _ = resampleDCM(vol, new_spacing=[spacing[0],spacing[1],3], is_label=False)
        brain, _, _ = resampleDCM(brain, new_spacing=[spacing[0],spacing[1],3], is_label=True)
        seg, _, _ = resampleDCM(seg, new_spacing=[spacing[0],spacing[1],3], is_label=True)
        
        new_size = vol.GetSize()
        
        vol = resize(vol,
                    size=(256,256,new_size[2]),
                    interpolator=sitk.sitkLinear)
        brain = resize(brain,
                    size=(256,256,new_size[2]),
                    interpolator=sitk.sitkNearestNeighbor)
        seg = resize(seg,
                    size=(256,256,new_size[2]),
                    interpolator=sitk.sitkNearestNeighbor)
                    
        vol_np = sitk.GetArrayFromImage(vol)
        vol_np = vol_np.astype(np.float32)
        
        brain_seg = sitk.GetArrayFromImage(brain)
        brain_seg = brain_seg.astype(np.int64)       
        vol_np[brain_seg==0] = 0
        
        vol_np = np.expand_dims(window_img(vol_np), 0)
        
        segmentation = sitk.GetArrayFromImage(seg)==1
        segmentation = segmentation.astype(np.int64)
        segmentation = np.expand_dims(segmentation, 0)

        yield vol_np, segmentation, dn
          

def get_slices(volume, segmentation, brain_mask):
    
    # Axis transpose order.
    axis = 1
    order = [1,0,2,3]

    volume = volume.transpose(order)
    segmentation = segmentation.transpose(order)
    brain_mask = brain_mask.transpose(order)

    indices_anomaly = []
    indices_healthy = []

    for i in range(len(volume)):
        count_total = np.product(volume[i].shape)
        count_brain = np.count_nonzero(brain_mask[i])
        count_tumor = np.count_nonzero(segmentation[i])        
        if count_brain==0:
            continue
        tumor_fraction = count_tumor/float(count_brain)
        brain_fraction = count_brain/float(count_total)
        if brain_fraction>0.2:
            if tumor_fraction>0.01:
                indices_anomaly.append(i)
            if count_tumor==0:
                indices_healthy.append(i)    

    # Sort slices.
    slices_dict = OrderedDict()
    slices_dict['ct_h'] = np.expand_dims(volume[indices_healthy,0,:,:],axis=1)
    slices_dict['ct_s'] = np.expand_dims(volume[indices_anomaly,0,:,:],axis=1)
    slices_dict['segmentation'] = segmentation[indices_anomaly]
    print(slices_dict['ct_h'].shape)
    print(slices_dict['ct_s'].shape)
    print(slices_dict['segmentation'].shape)
    return slices_dict


def preprocess(volume, segmentation):
    volume_out = volume.copy()
    
    # Mean center and normalize by std.
    brain_mask = volume!=0
    volume_out[brain_mask] -= volume_out[brain_mask].mean()
    volume_out[brain_mask] /= volume_out[brain_mask].std()*5    # fit in tanh
    
    # Split volume along hemispheres.
    mid0 = volume.shape[-1]//2
    mid1 = mid0
    if volume.shape[-1]%2:
        mid0 += 1
    volume_out = np.concatenate([volume_out[:,:,:,:mid0],
                                 volume_out[:,:,:,mid1:]], axis=1)
    segmentation_out = np.concatenate([segmentation[:,:,:,:mid0],
                                       segmentation[:,:,:,mid1:]], axis=1)    
    brain_mask = np.concatenate([brain_mask[:,:,:,:mid0],
                                 brain_mask[:,:,:,mid1:]], axis=1)       
    
       
    return volume_out, segmentation_out, brain_mask


def process_case(case_num, h5py_file, volume, segmentation, fn):
    print("Processing case {}: {}".format(case_num, fn))
    group_p = h5py_file.create_group(str(case_num))
    # TODO: set attribute containing fn.
    print('preprocessing')
    volume, seg, brain_mask = preprocess(volume, segmentation)
    
    print('getting_slices')
    slices = get_slices(volume, seg, brain_mask)

    for key in slices.keys():
        if len(slices[key])==0:
            kwargs = {}
        else:
            kwargs = {'chunks': (1,)+slices[key].shape[1:],
                      'compression': 'lzf'}
        group_p.create_dataset(key,
                               shape=slices[key].shape,
                               data=slices[key],
                               dtype=slices[key].dtype,
                               **kwargs)
    
                                       
if __name__=='__main__':
    args = parse()
    #if os.path.exists(args.save_to):
        #raise ValueError("Path to save data already exists. Aborting.")
    h5py_file = h5py.File(args.save_to, mode='w')
    for i, (volume, seg, fn) in enumerate(data_loader(args.data_dir)):
        process_case(i, h5py_file, volume, seg, fn)
