from collections import OrderedDict

import h5py
import numpy as np
from scipy import ndimage
import SimpleITK as sitk

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms \
    import (BrightnessMultiplicativeTransform,
            ContrastAugmentationTransform,
            BrightnessTransform)
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.noise_transforms \
    import (GaussianNoiseTransform,
            GaussianBlurTransform)
from batchgenerators.transforms.resample_transforms \
    import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2

from data_tools.data_augmentation import image_random_transform
from data_tools.wrap import multi_source_array

def prepare_mr_ct(path_mr, path_ct, masked_fraction_mr=0, masked_fraction_ct=0, 
                   drop_masked=False, rng=None):
    
    rnd_state = np.random.RandomState(0)

    indices = np.arange(0, 369)
    rnd_state.shuffle(indices)
    val_mr  = indices[0:37]
    test_mr  = indices[37:74]

    indices = np.arange(0, 167)
    rnd_state.shuffle(indices)
    val_ct  = indices[0:17]  
    test_ct  = indices[17:34]   

    return _prepare_mr_ct(path_mr,
                          path_ct,
                          validation_indices_mr=val_mr,
                          test_indices_mr=test_mr,
                          validation_indices_ct=val_ct,
                          test_indices_ct=test_ct,
                          masked_fraction_mr=masked_fraction_mr, masked_fraction_ct=masked_fraction_ct,                                                            
                          drop_masked=drop_masked,
                          rng=rng)

def _prepare_mr_ct(path_mr,
                   path_ct,
                   validation_indices_mr,
                   test_indices_mr,
                   validation_indices_ct,
                   test_indices_ct,
                   masked_fraction_mr=0, masked_fraction_ct=0,
                   drop_masked=False, rng=None):
    """
    Convenience function to prepare brats data as multi_source_array objects,
    split into training and validation subsets.
    
    path_hgg (string) : Path of the h5py file containing the data
    masked_fraction (float) : The fraction in [0, 1.] of volumes in the 
        training set for which  to return segmentation masks as None
    drop_masked (bool) : Whether to omit volumes with "masked" segmentations.
    rng (numpy RandomState) : Random number generator.
    
    NOTE: A constant random seed (0) is always used to determine the training/
    validation/testing split. The rng passed for data preparation is used to 
    determine which labels to mask out (if any); if none is passed, the default
    uses a random seed of 0.
    
    Returns six arrays: healthy slices, sick slices, and segmentations for 
    the training and validation subsets.
    """
    
    if rng is None:
        rng = np.random.RandomState(0)
    
    # Assemble volumes and corresponding segmentations; split train/valid/test.
    volumes_h_mr = {'train': [], 'valid': [], 'test': []}
    volumes_sm_mr = {'train': [], 'valid': [], 'test': []}
    volumes_s_mr = {'train': []}
    volumes_m_mr = {'train': []}    
    volumes_h_ct = {'train': [], 'valid': [], 'test': []}
    volumes_sm_ct = {'train': [], 'valid': [], 'test': []}
    volumes_s_ct = {'train': []}
    volumes_m_ct = {'train': []}

    h5py_file_mr = h5py.File(path_mr, mode='r')
    h5py_file_ct = h5py.File(path_ct, mode='r')
    
    for idx, case_id in enumerate(h5py_file_mr.keys()):   # Per patient.
        #if idx > 10:
        #    continue
        f = h5py_file_mr[case_id]
        if idx in validation_indices_mr:
            split = 'valid'
            volumes_sm_mr[split].append(np.concatenate((f['flair_s'],f['segmentation']), axis=1))
            volumes_h_mr[split].append(f['flair_h'])            
        elif idx in test_indices_mr:
            split = 'test'        
            volumes_sm_mr[split].append(np.concatenate((f['flair_s'],f['segmentation']), axis=1))
            volumes_h_mr[split].append(f['flair_h'])                  
        else:
            split = 'train'
            volumes_s_mr[split].append(f['flair_s'])
            volumes_m_mr[split].append(f['segmentation'])            
            volumes_h_mr[split].append(f['flair_h'])

    for idx, case_id in enumerate(h5py_file_ct.keys()):   # Per patient.
        #if idx > 10:
        #    continue
        f = h5py_file_ct[case_id]
        if idx in validation_indices_ct:
            split = 'valid'
            volumes_sm_ct[split].append(np.concatenate((f['ct_s'],f['segmentation']), axis=1))
            volumes_h_ct[split].append(f['ct_h'])            
        elif idx in test_indices_ct:
            split = 'test'        
            volumes_sm_ct[split].append(np.concatenate((f['ct_s'],f['segmentation']), axis=1))
            volumes_h_ct[split].append(f['ct_h'])                  
        else:
            split = 'train'
            volumes_s_ct[split].append(f['ct_s'])
            volumes_m_ct[split].append(f['segmentation'])            
            volumes_h_ct[split].append(f['ct_h'])
    
    if masked_fraction_mr!=1:
        # mr
        # Volumes with these indices will either be dropped from the training
        # set or have their segmentations set to None.
        # 
        # The `masked_fraction` determines the maximal fraction of slices that
        # are to be thus removed. All or none of the slices are selected for 
        # each volume.
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_m_mr['train']])
        num_masked_slices = 0
        max_masked_slices = int(min(num_total_slices,
                                    num_total_slices*masked_fraction_mr+0.5))
        for i in rng.permutation(len(volumes_m_mr['train'])):
            num_slices = len(volumes_m_mr['train'][i])
            if num_slices>0 and num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            if num_slices+num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            masked_indices.append(i)
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices)+"flair slices are labeled across {} "
              "volumes ({:.1f}%).".format(len(volumes_m_mr['train'])-len(masked_indices),
                        100*(1-num_masked_slices/float(num_total_slices))))
    else : 
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_m_mr['train']])
        num_masked_slices = 0
        for i in rng.permutation(len(volumes_m_mr['train'])):
            num_slices = len(volumes_m_mr['train'][i])
            masked_indices.append(i)
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices,
                        )+"flair slices are labeled"
              )
    
    # Apply masking in one of two ways.
    # 
    # 1. Mask out the labels for volumes indexed with `masked_indices` by 
    # setting the segmentations volume as an array of `None`, with length 
    # equal to the number of slices in the volume.
    # 
    # OR if `drop_masked` is True:
    # 
    # 2. Remove all volumes indexed with `masked_indices`.
    volumes_sm_train = []

    for i in range(len(volumes_m_mr['train'])):
        if i in masked_indices:
            # Mask out or drop.
            if drop_masked:
                continue    # Drop.
            volumes_sm_train.append(np.concatenate((volumes_s_mr['train'][i],np.full(volumes_m_mr['train'][i].shape,None)), axis=1)) 
        else:
            # Keep.
            volumes_sm_train.append(np.concatenate((volumes_s_mr['train'][i],volumes_m_mr['train'][i]), axis=1))

    volumes_sm_mr['train'] = volumes_sm_train         


    if masked_fraction_ct!=1:
        # ct
        # Volumes with these indices will either be dropped from the training
        # set or have their segmentations set to None.
        # 
        # The `masked_fraction` determines the maximal fraction of slices that
        # are to be thus removed. All or none of the slices are selected for 
        # each volume.
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_m_ct['train']])
        num_masked_slices = 0
        max_masked_slices = int(min(num_total_slices,
                                    num_total_slices*masked_fraction_ct+0.5))
        for i in rng.permutation(len(volumes_m_ct['train'])):
            num_slices = len(volumes_m_ct['train'][i])
            if num_slices>0 and num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            if num_slices+num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            masked_indices.append(i)
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices)+"flair slices are labeled across {} "
              "volumes ({:.1f}%).".format(len(volumes_m_ct['train'])-len(masked_indices),
                        100*(1-num_masked_slices/float(num_total_slices))))
    else : 
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_m_ct['train']])
        num_masked_slices = 0
        for i in rng.permutation(len(volumes_m_ct['train'])):
            num_slices = len(volumes_m_ct['train'][i])
            masked_indices.append(i)
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices,
                        )+"ct slices are labeled"
              )
    
    # Apply masking in one of two ways.
    # 
    # 1. Mask out the labels for volumes indexed with `masked_indices` by 
    # setting the segmentations volume as an array of `None`, with length 
    # equal to the number of slices in the volume.
    # 
    # OR if `drop_masked` is True:
    # 
    # 2. Remove all volumes indexed with `masked_indices`.
    volumes_sm_train = []

    for i in range(len(volumes_m_ct['train'])):
        if i in masked_indices:
            # Mask out or drop.
            if drop_masked:
                continue    # Drop.
            volumes_sm_train.append(np.concatenate((volumes_s_ct['train'][i],np.full(volumes_m_ct['train'][i].shape,None)), axis=1)) 
        else:
            # Keep.
            volumes_sm_train.append(np.concatenate((volumes_s_ct['train'][i],volumes_m_ct['train'][i]), axis=1))

    volumes_sm_ct['train'] = volumes_sm_train        
    
    # Merge all arrays in each list of arrays.
    data = OrderedDict([('train', OrderedDict()),
                        ('valid', OrderedDict()),
                        ('test', OrderedDict())])
    for key in data.keys():
        # HACK: we may have a situation where the number of sick examples
        # is greater than the number of healthy. In that case, we should
        # duplicate the healthy set M times so that it has a bigger size
        # than the sick set.

        data[key]['h_mr']  = multi_source_array(volumes_h_mr[key])
        data[key]['sm_mr']  = multi_source_array(volumes_sm_mr[key])
        data[key]['h_ct']  = multi_source_array(volumes_h_ct[key])
        data[key]['sm_ct'] = multi_source_array(volumes_sm_ct[key])

    return data



def preprocessor_mr_ct(data_augmentation_kwargs=None, label_warp=None,
                       label_shift=None, label_dropout=0,
                       label_crop_rand=None, label_crop_rand2=None,
                       label_crop_left=None):
    """
    Preprocessor function to pass to a data_flow, for BRATS data.
    
    data_augmentation_kwargs : Dictionary of keyword arguments to pass to
        the data augmentation code (image_stack_random_transform).
    label_warp (float) : The sigma value of the spline warp applied to
        to the target label mask during training in order to corrupt it. Used
        for testing robustness to label noise.
    label_shift (int) : The number of pixels to shift all training target masks
        to the right.
    label_dropout (float) : The probability in [0, 1] of discarding a slice's
        segmentation mask.
    label_crop_rand (float) : Crop out a randomly sized rectangle out of every
        connected component of the mask. The minimum size of the rectangle is
        set as a fraction of the connected component's bounding box, in [0, 1].
    label_crop_rand2 (float) : Crop out a randomly sized rectangle out of every
        connected component of the mask. The mean size in each dimension is
        set as a fraction of the connected component's width/height, in [0, 1].
    label_crop_left (float) : If true, crop out the left fraction (in [0, 1]) 
        of every connected component of the mask.
    """
        
    def process_element(inputs):
        
        h_t1, sm_mr, h_t2, sm_ct = inputs

        # Retrieve 
        s_t1 = np.expand_dims(sm_mr[0,:,:], axis=0).astype(np.float32)
        m_t1 = np.expand_dims(sm_mr[1,:,:], axis=0)
        
        s_t2 = np.expand_dims(sm_ct[0,:,:], axis=0).astype(np.float32)
        m_t2 = np.expand_dims(sm_ct[1,:,:], axis=0)
        
        if m_t1[0,0,0]==None:
            m_t1=None        
        
        if m_t2[0,0,0]==None:
            m_t2=None
       
        # Float.
        h_t1 = h_t1.astype(np.float32)
        s_t1 = s_t1.astype(np.float32)
        h_t2 = h_t2.astype(np.float32)
        s_t2 = s_t2.astype(np.float32)
        
        # Data augmentation.
        
        if data_augmentation_kwargs=='nnunet':
            if h_t1 is not None:
                h_t1 = nnunet_transform(h_t1)
            if s_t1 is not None:
                _ = nnunet_transform(s_t1, m_t1)
            if m_t1 is not None:
                assert s_t1 is not None
                s_t1, m_t1 = _
            else:
                s_t1 = _
            if h_t2 is not None:
                h_t2 = nnunet_transform(h_t2)
            if s_t2 is not None:
                _ = nnunet_transform(s_t2, m_t2)
            if m_t2 is not None:
                assert s_t2 is not None
                s_t2, m_t2 = _
            else:
                s_t2 = _        
        
        
        elif data_augmentation_kwargs=='nnunet_default':
            if h_t1 is not None:
                h_t1 = nnunet_transform_default(h_t1)
            if s_t1 is not None:
                _ = nnunet_transform_default(s_t1, m_t1)
            if m_t1 is not None:
                assert s_t1 is not None
                s_t1, m_t1 = _
            else:
                s_t1 = _
            if h_t2 is not None:
                h_t2 = nnunet_transform_default(h_t2)
            if s_t2 is not None:
                _ = nnunet_transform_default(s_t2, m_t2)
            if m_t2 is not None:
                assert s_t2 is not None
                s_t2, m_t2 = _
            else:
                s_t2 = _                
        
        
        elif data_augmentation_kwargs is not None:
            if h_t1 is not None:
                h_t1 = image_random_transform(h_t1, **data_augmentation_kwargs,
                                           n_warp_threads=1)
            if s_t1 is not None:
                _ = image_random_transform(s_t1, m_t1, **data_augmentation_kwargs,
                                           n_warp_threads=1)
            if m_t1 is not None:
                assert s_t1 is not None
                s_t1, m_t1 = _
            else:
                s_t1 = _
            if h_t2 is not None:
                h_t2 = image_random_transform(h_t2, **data_augmentation_kwargs,
                                           n_warp_threads=1)
            if s_t2 is not None:
                _ = image_random_transform(s_t2, m_t2, **data_augmentation_kwargs,
                                           n_warp_threads=1)
            if m_t2 is not None:
                assert s_t2 is not None
                s_t2, m_t2 = _
            else:
                s_t2 = _
                
        
        # Remove distant outlier intensities.
        if h_t1 is not None:
            h_t1 = np.clip(h_t1, -1., 1.)
        if s_t1 is not None:
            s_t1 = np.clip(s_t1, -1., 1.)
        if h_t2 is not None:
            h_t2 = np.clip(h_t2, -1., 1.)
        if s_t2 is not None:
            s_t2 = np.clip(s_t2, -1., 1.)  
            
        return h_t1, s_t1, m_t1, h_t2, s_t2, m_t2
        
    def process_batch(batch):
        # Process every element.
        elements = []
        for i in range(len(batch[0])):
            elem = process_element([b[i] for b in batch])
            elements.append(elem)
        out_batch = list(zip(*elements))
        return out_batch
    
    return process_batch


