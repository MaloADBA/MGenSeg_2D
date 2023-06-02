
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
        "groups: \'sick\', \'healthy\', and \'segmentations\' for each of the 4 contrasts."
        "Sick cases contain any anomolous class, healthy cases contain no anomalies, and "
        "segmentations are the segmentations of the anomalies. Each group "
        "contains subgroups for each of the three orthogonal planes along "
        "which slices are extracted. For each case, MRI sequences are stored "
        "in a single volume")
    parser.add_argument('--data_dir',
                        help="The directory containing the BRATS 2020 data. ",
                        required=True, type=str)
    parser.add_argument('--save_to',
                        help="Path to save the HDF5 file to.",
                        required=True, type=str)
    parser.add_argument('--skip_bias_correction',
                        help="Whether to skip N4 bias correction.",
                        required=False, action='store_true')
    parser.add_argument('--no_crop',
                        help="Whether to not crop slices to the minimal "
                             "bounding box containing the brain.",
                        required=False, action='store_true')
    parser.add_argument('--min_tumor_fraction',
                        help="Minimum amount of tumour per slice in [0, 1].",
                        required=False, type=float, default=0.01)
    parser.add_argument('--min_brain_fraction',
                        help="Minimum amount of brain per slice in [0, 1].",
                        required=False, type=float, default=0.05)
    parser.add_argument('--num_threads',
                        help="The number of parallel threads to execute.",
                        required=False, type=int, default=None)
    parser.add_argument('--save_debug_to',
                        help="Save images of each slice to this directory, "
                             "for inspection.",
                        required=False, type=str, default=None)
    return parser.parse_args()


def data_loader(data_dir, crop=True):
    tags = ['flair', 't1ce', 't1', 't2', 'seg']
    tags_modality = ['flair', 't1ce', 't1', 't2', 'seg']
    for dn in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, dn)
        
        # Match filenames to tags.
        # NOTE: t1ce chosen when both t1 and t1ce matched
        fn_dict = OrderedDict()
        for fn in sorted(os.listdir(path)):
            match = [t for t in tags if t in fn]
            fn_dict[match[0]] = fn
        print(fn_dict)    
        # Load files.
        vol_all = []
        segmentation = None
        size = None
        size = None
        for t in tags_modality:
            vol = sitk.ReadImage(os.path.join(path, fn_dict[t]))
            vol_np = sitk.GetArrayFromImage(vol)
            if size is None:
                size = vol_np.shape
            if vol_np.shape != size:
                raise Exception("Expected {} to have a size of {} but got {}."
                                "".format(fn_dict[t], size, vol_np.shape))
            
            if t=='seg':
                segmentation = vol_np.astype(np.int64)
                segmentation = np.expand_dims(segmentation, 0)
            else:
                vol_np = vol_np.astype(np.float32)
                vol_all.append(np.expand_dims(vol_np, 0))
                
        # Concatenate on channel axis.
        volume = np.concatenate(vol_all, axis=0)
        
        # Crop to volume.
        if crop:
            bbox = ndimage.find_objects(volume!=0)[0]
            volume = volume[bbox]
            segmentation = segmentation[bbox]
            
        yield volume, segmentation, dn
        

        
        
def get_slices(volume, segmentation, brain_mask, indices,
               min_tumor_fraction, min_brain_fraction):
    
    assert min_tumor_fraction>=0 and min_tumor_fraction<=1
    assert min_brain_fraction>=0 and min_brain_fraction<=1
    
    # Axis transpose order.
    axis = 1
    order = [1,0,2,3]
    volume = volume.transpose(order)
    segmentation = segmentation.transpose(order)
    brain_mask = brain_mask.transpose(order)
    
    # Select slices. Slices with anomalies are sick, others are healthy.
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
        if brain_fraction>min_brain_fraction:
            if tumor_fraction>min_tumor_fraction:
                indices_anomaly.append(i)
            if count_tumor==0:
                indices_healthy.append(i)

    # Sort slices.
    slices_dict = OrderedDict()
    slices_dict['healthy_flair'] = np.expand_dims(volume[indices_healthy,0,:,:],axis=1)
    slices_dict['sick_flair'] = np.expand_dims(volume[indices_anomaly,0,:,:],axis=1)
    slices_dict['healthy_t1ce'] = np.expand_dims(volume[indices_healthy,1,:,:],axis=1)
    slices_dict['sick_t1ce'] = np.expand_dims(volume[indices_anomaly,1,:,:],axis=1)
    slices_dict['healthy_t1'] = np.expand_dims(volume[indices_healthy,2,:,:],axis=1)
    slices_dict['sick_t1'] = np.expand_dims(volume[indices_anomaly,2,:,:],axis=1)
    slices_dict['healthy_t2'] = np.expand_dims(volume[indices_healthy,3,:,:],axis=1)
    slices_dict['sick_t2'] = np.expand_dims(volume[indices_anomaly,3,:,:],axis=1)
    slices_dict['segmentation'] = segmentation[indices_anomaly]
    slices_dict['h_indices'] = indices[indices_healthy]
    slices_dict['s_indices'] = indices[indices_anomaly]
    
    return slices_dict

""""Crop or pad to the right dimensions functions"""

def pad_or_crop_image(image, seg, target_size=(128, 128, 128), random_crop = False):
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim, random_crop = random_crop) for target, dim in zip(target_size, (z, y, x))]
    image = image[:, z_slice, y_slice, x_slice]
    seg = seg[:, z_slice, y_slice, x_slice]
    todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(target_size, [z, y, x])]
    padlist = [(0, 0)]  # channel dim
    for to_pad in todos:
        if to_pad[0]:
            padlist.append((to_pad[1], to_pad[2]))
        else:
            padlist.append((0, 0))
    image = np.pad(image, padlist)
    seg = np.pad(seg, padlist)
    return image, seg

def get_left_right_idx_should_pad(target_size, dim):
    if dim >= target_size:
        return [False]
    elif dim < target_size:
        pad_extent = target_size - dim
        left = int(pad_extent/2)
        right = pad_extent - left
        return True, left, right


def get_crop_slice(target_size, dim, random_crop):
    if dim > target_size:
        crop_extent = dim - target_size
        if random_crop:
            left = random.randint(0, crop_extent)
        else:
            left = int(crop_extent/2) #
        right = crop_extent - left
        return slice(left, dim - right)
    elif dim <= target_size:
        return slice(0, dim)




def preprocess(volume, segmentation, skip_bias_correction=False):
    volume_out = volume.copy()
    segmentation_out = segmentation.copy()
    
    # Mean center and normalize by std.
    brain_mask = volume!=0
    volume_out[brain_mask] -= volume_out[brain_mask].mean()
    volume_out[brain_mask] /= volume_out[brain_mask].std()*5    # fit in tanh
    
    # Get slice indices, with 0 at the center.
    #brain_mask_ax1 = brain_mask.sum(axis=(0,2,3))>0
    #idx_min = np.argmax(brain_mask_ax1)
    #idx_max = len(brain_mask_ax1)-1-np.argmax(np.flipud(brain_mask_ax1))
    #idx_mid = (idx_max-idx_min)//2
    #a = idx_mid-len(brain_mask_ax1)
    #b = len(brain_mask_ax1)+a-1
    #indices = np.arange(a, b)
    indices = np.arange(brain_mask.shape[1])

        
    # Split volume along hemispheres.
    mid0 = volume_out.shape[-1]//2
    print(mid0)
    mid1 = mid0
    if volume_out.shape[-1]%2:
        mid0 += 1
    volume_out = np.concatenate([volume_out[:,:,:,:mid0],
                                 volume_out[:,:,:,mid1:]], axis=1)
    segmentation_out = np.concatenate([segmentation_out[:,:,:,:mid0],
                                       segmentation_out[:,:,:,mid1:]], axis=1)
    brain_mask = np.concatenate([brain_mask[:,:,:,:mid0],
                                 brain_mask[:,:,:,mid1:]], axis=1)
    indices = np.concatenate([indices, indices])
    return volume_out, segmentation_out, brain_mask, indices


def process_case(case_num, h5py_file, volume, segmentation, fn,
                 min_tumor_fraction, min_brain_fraction,
                 skip_bias_correction=False, save_debug_to=None):
    print("Processing case {}: {}".format(case_num, fn))
    group_p = h5py_file.create_group(str(case_num))
    # TODO: set attribute containing fn.
    print('preprocessing')
    volume, seg, m, indices = preprocess(volume, segmentation,
                                      skip_bias_correction)
    
    print('getting_slices')
    slices = get_slices(volume, seg, m, indices,
                        min_tumor_fraction, min_brain_fraction)

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
    
    # Debug outputs for inspection.
    if save_debug_to is not None:
        for key in slices.keys():
            if "indices" in key:
                continue
            dest = os.path.join(save_debug_to, key)
            if not os.path.exists(dest):
                os.makedirs(dest)
            for i in range(len(slices[key])):
                im = slices[key][i]
                for ch, im_ch in enumerate(im):
                    imageio.imwrite(os.path.join(dest, "{}_{}_{}.png"
                                                   "".format(case_num, i, ch)),
                                      slices[key][i][ch])
                                       

if __name__=='__main__':
    args = parse()
    #if os.path.exists(args.save_to):
        #raise ValueError("Path to save data already exists. Aborting.")
    h5py_file = h5py.File(args.save_to, mode='w')
    for i, (volume, seg, fn) in enumerate(data_loader(args.data_dir,
                                               not args.no_crop)):
        process_case(i, h5py_file, volume, seg, fn,
                        args.min_tumor_fraction,
                        args.min_brain_fraction,
                        args.skip_bias_correction,
                        args.save_debug_to)

