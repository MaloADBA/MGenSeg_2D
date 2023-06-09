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

def prepare_base(path,
                 modalities = ['t1', 't2'],
                  masked_fraction_1=0, masked_fraction_2=0, 
                  drop_masked=False, rng=None):
    # Random 20% data split (10% validation, 10% testing).
    rnd_state = np.random.RandomState(0)
    indices = np.arange(0, 369)
    rnd_state.shuffle(indices)
    val  = indices[0:37]
    test = indices[37:74]
    return _prepare_base(path,
                               modalities,
                               masked_fraction_1=masked_fraction_1,
                               masked_fraction_2=masked_fraction_2,
                               validation_indices=val,
                               testing_indices=test,
                               drop_masked=drop_masked,
                               rng=rng)

def _prepare_base(path, modalities, validation_indices,
                   testing_indices, masked_fraction_1=0, masked_fraction_2=0,
                   drop_masked=False, rng=None):
    """
    Convenience function to prepare multimodal brats data as multi_source_array objects for baselines,
    split into training and validation subsets.
    
    path (string) : Path of the h5py file containing the data
    modalities (list of strings) : ["source","target"] contrasts picked for experiment
    masked_fraction_2 (float) : The fraction in [0, 1.] of source volumes in the 
        training set for which  to return segmentation masks as None
    masked_fraction_1 (float) : The fraction in [0, 1.] of source volumes in the 
        training set for which  to return segmentation masks as None
    drop_masked (bool) : Whether to omit volumes with "masked" segmentations.
    rng (numpy RandomState) : Random number generator.
    
    NOTE: A constant random seed (0) is always used to determine the training/
    validation/testing split. The rng passed for data preparation is used to 
    determine which labels to mask out (if any); if none is passed, the default
    uses a random seed of 0.
    """
    
    if rng is None:
        rng = np.random.RandomState(0)
    
    # Assemble volumes and corresponding segmentations; split train/valid/test.
    volumes_h_1 = {'train': [], 'valid': [], 'test': []}
    volumes_s_1 = {'train': [], 'valid': [], 'test': []}
    volumes_h_2 = {'train': [], 'valid': [], 'test': []}
    volumes_s_2 = {'train': [], 'valid': [], 'test': []}
    volumes_m_1 = {'train': [], 'valid': [], 'test': []}
    volumes_m_2 = {'train': [], 'valid': [], 'test': []}
    indices_h_1 = {'train': [], 'valid': [], 'test': []}
    indices_s_1 = {'train': [], 'valid': [], 'test': []}
    indices_h_2 = {'train': [], 'valid': [], 'test': []}
    indices_s_2 = {'train': [], 'valid': [], 'test': []}
    
    volumes_h = {'train': [], 'valid': [], 'valid_1': [], 'valid_2': [], 'test': [], 'test_1': [], 'test_2': []}
    volumes_s = {'train': [], 'valid': [], 'valid_1': [], 'valid_2': [], 'test': [], 'test_1': [], 'test_2': []}
    volumes_m = {'train': [], 'valid': [], 'valid_1': [], 'valid_2': [], 'test': [], 'test_1': [], 'test_2': []}
    indices_h = {'train': [], 'valid': [], 'valid_1': [], 'valid_2': [], 'test': [], 'test_1': [], 'test_2': []}
    indices_s = {'train': [], 'valid': [], 'valid_1': [], 'valid_2': [], 'test': [], 'test_1': [], 'test_2': []}

    try:
        h5py_file = h5py.File(path, mode='r')
    except:
        print("Failed to open data: {}".format(path))
        raise
    indices = []
    case_identities = []
    for idx, case_id in enumerate(h5py_file.keys()):   # Per patient.
        f = h5py_file[case_id]
        if idx in validation_indices:
            split = 'valid'
            
            volumes_h[split].append(f['healthy_'+modalities[0]])
            volumes_s[split].append(f['sick_'+modalities[0]])
            volumes_m[split].append(f['segmentation'])
            indices_h[split].append(f['h_indices'])
            indices_s[split].append(f['s_indices'])
            volumes_h[split].append(f['healthy_'+modalities[1]])
            volumes_s[split].append(f['sick_'+modalities[1]])
            volumes_m[split].append(f['segmentation'])
            indices_h[split].append(f['h_indices'])
            indices_s[split].append(f['s_indices'])
            
            volumes_h['valid_1'].append(f['healthy_'+modalities[0]])
            volumes_s['valid_1'].append(f['sick_'+modalities[0]])
            volumes_m['valid_1'].append(f['segmentation'])
            indices_h['valid_1'].append(f['h_indices'])
            indices_s['valid_1'].append(f['s_indices'])
            
            volumes_h['valid_2'].append(f['healthy_'+modalities[1]])
            volumes_s['valid_2'].append(f['sick_'+modalities[1]])
            volumes_m['valid_2'].append(f['segmentation'])
            indices_h['valid_2'].append(f['h_indices'])
            indices_s['valid_2'].append(f['s_indices'])
            
        elif idx in testing_indices:
            split = 'test'
            
            volumes_h[split].append(f['healthy_'+modalities[0]])
            volumes_s[split].append(f['sick_'+modalities[0]])
            volumes_m[split].append(f['segmentation'])
            indices_h[split].append(f['h_indices'])
            indices_s[split].append(f['s_indices'])
            volumes_h[split].append(f['healthy_'+modalities[1]])
            volumes_s[split].append(f['sick_'+modalities[1]])
            volumes_m[split].append(f['segmentation'])
            indices_h[split].append(f['h_indices'])
            indices_s[split].append(f['s_indices'])
            
            volumes_h['test_1'].append(f['healthy_'+modalities[0]])
            volumes_s['test_1'].append(f['sick_'+modalities[0]])
            volumes_m['test_1'].append(f['segmentation'])
            indices_h['test_1'].append(f['h_indices'])
            indices_s['test_1'].append(f['s_indices'])
            
            volumes_h['test_2'].append(f['healthy_'+modalities[1]])
            volumes_s['test_2'].append(f['sick_'+modalities[1]])
            volumes_m['test_2'].append(f['segmentation'])
            indices_h['test_2'].append(f['h_indices'])
            indices_s['test_2'].append(f['s_indices'])
        else:
            split = 'train'
            volumes_h_1[split].append(f['healthy_'+modalities[0]])
            volumes_s_1[split].append(f['sick_'+modalities[0]])
            volumes_m_1[split].append(f['segmentation'])
            indices_h_1[split].append(f['h_indices'])
            indices_s_1[split].append(f['s_indices'])
            volumes_h_2[split].append(f['healthy_'+modalities[1]])
            volumes_s_2[split].append(f['sick_'+modalities[1]])
            volumes_m_2[split].append(f['segmentation'])
            indices_h_2[split].append(f['h_indices'])
            indices_s_2[split].append(f['s_indices'])

    
    if masked_fraction_1!=1:

        # Source volumes with these indices will either be dropped from the training
        # set or have their segmentations set to None.
        # 
        # The `masked_fraction` determines the maximal fraction of slices that
        # are to be thus removed. All or none of the slices are selected for 
        # each volume.
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_m_1['train']])
        num_masked_slices = 0
        max_masked_slices = int(min(num_total_slices,
                                    num_total_slices*masked_fraction_1+0.5))
        for i in rng.permutation(len(volumes_m_1['train'])):
            num_slices = len(volumes_m_1['train'][i])
            if num_slices>0 and num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            if num_slices+num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            masked_indices.append(i)
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices)+modalities[0]+" slices are labeled across {} "
              "volumes ({:.1f}%).".format(len(volumes_m_1['train'])-len(masked_indices),
                        100*(1-num_masked_slices/float(num_total_slices))))
    else : 
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_m_1['train']])
        num_masked_slices = 0
        for i in rng.permutation(len(volumes_m_1['train'])):
            num_slices = len(volumes_m_1['train'][i])
            masked_indices.append(i)
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices,
                        )+modalities[0]+" slices are labeled"
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
    volumes_h_train = []
    volumes_s_train = []
    volumes_m_train = []
    indices_h_train = []
    indices_s_train = []
    for i in range(len(volumes_m_1['train'])):
        if i in masked_indices:
            # Mask out or drop.
            if drop_masked:
                continue    # Drop.
            volumes_m_train.append(np.array([None]*len(volumes_m_1['train'][i])))
        else:
            # Keep.
            volumes_m_train.append(volumes_m_1['train'][i])
        volumes_h_train.append(volumes_h_1['train'][i])
        volumes_s_train.append(volumes_s_1['train'][i])
        indices_h_train.append(indices_h_1['train'][i])
        indices_s_train.append(indices_s_1['train'][i])
    
    if masked_fraction_2!=1:
        # Target volumes with these indices will either be dropped from the training
        # set or have their segmentations set to None.
        # 
        # The `masked_fraction` determines the maximal fraction of slices that
        # are to be thus removed. All or none of the slices are selected for 
        # each volume.
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_m_2['train']])
        num_masked_slices = 0
        max_masked_slices = int(min(num_total_slices,
                                    num_total_slices*masked_fraction_2+0.5))
        for i in rng.permutation(len(volumes_m_2['train'])):
            num_slices = len(volumes_m_2['train'][i])
            if num_slices>0 and num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            if num_slices+num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            masked_indices.append(i)
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices)+modalities[1]+" slices are labeled across {} "
              "volumes ({:.1f}%).".format(len(volumes_m_2['train'])-len(masked_indices),
                        100*(1-num_masked_slices/float(num_total_slices))))
    else : 
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_m_2['train']])
        num_masked_slices = 0
        for i in rng.permutation(len(volumes_m_2['train'])):
            masked_indices.append(i)
            num_slices = len(volumes_m_2['train'][i])
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices,
                        )+modalities[1]+" slices are labeled"
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
    for i in range(len(volumes_m_2['train'])):
        if i in masked_indices:
            # Mask out or drop.
            if drop_masked:
                continue    # Drop.
            volumes_m_train.append(np.array([None]*len(volumes_m_2['train'][i])))
        else:
            # Keep.
            volumes_m_train.append(volumes_m_2['train'][i])
        volumes_h_train.append(volumes_h_2['train'][i])
        volumes_s_train.append(volumes_s_2['train'][i])
        indices_h_train.append(indices_h_2['train'][i])
        indices_s_train.append(indices_s_2['train'][i])

    volumes_h['train'] = volumes_h_train
    volumes_s['train'] = volumes_s_train
    volumes_m['train'] = volumes_m_train
    indices_h['train'] = indices_h_train
    indices_s['train'] = indices_s_train
    
    # Merge all arrays in each list of arrays.
    data = OrderedDict([('train', OrderedDict()),
                        ('valid', OrderedDict()),
                        ('valid_1', OrderedDict()),
                        ('valid_2', OrderedDict()),
                        ('test', OrderedDict()),
                        ('test_1', OrderedDict()),
                        ('test_2', OrderedDict())])
    for key in data.keys():
        # HACK: we may have a situation where the number of sick examples
        # is greater than the number of healthy. In that case, we should
        # duplicate the healthy set M times so that it has a bigger size
        # than the sick set.
        m = 1
        len_h = sum([len(elem) for elem in volumes_h[key]])
        len_s = sum([len(elem) for elem in volumes_s[key]])
        if len_h < len_s:
            m = int(np.ceil(len_s / len_h))
        data[key]['h']  = multi_source_array(volumes_h[key]*m)
        data[key]['s']  = multi_source_array(volumes_s[key])
        data[key]['m']  = multi_source_array(volumes_m[key])
        data[key]['hi'] = multi_source_array(indices_h[key]*m)
        data[key]['si'] = multi_source_array(indices_s[key])
    return data



def prepare_mbrats(path,
                   modalities = ['t1', 't2'],
                   masked_fraction_1=0, masked_fraction_2=0, 
                   drop_masked=False, rng=None):
    # Random 20% data split (10% validation, 10% testing).
    rnd_state = np.random.RandomState(0)
    indices = np.arange(0, 369)
    rnd_state.shuffle(indices)
    val  = indices[0:37]
    test = indices[37:74]
    return _prepare_mbrats(path,
                          modalities,
                          masked_fraction_1=masked_fraction_1, 
                          masked_fraction_2=masked_fraction_2,
                          validation_indices=val,
                          testing_indices=test,
                          drop_masked=drop_masked,
                          rng=rng)

def _prepare_mbrats(path, modalities, validation_indices,
                   testing_indices, masked_fraction_1=0, masked_fraction_2=0,
                   drop_masked=False, rng=None):
    """
    Convenience function to prepare multimodal brats data as multi_source_array objects for MGenSeg,
    split into training and validation subsets.
    
    path (string) : Path of the h5py file containing the data
    modalities (list of strings) : ["source","target"] contrasts picked for experiment
    masked_fraction_2 (float) : The fraction in [0, 1.] of source volumes in the 
        training set for which  to return segmentation masks as None
    masked_fraction_1 (float) : The fraction in [0, 1.] of source volumes in the 
        training set for which  to return segmentation masks as None
    drop_masked (bool) : Whether to omit volumes with "masked" segmentations.
    rng (numpy RandomState) : Random number generator.
    
    NOTE: A constant random seed (0) is always used to determine the training/
    validation/testing split. The rng passed for data preparation is used to 
    determine which labels to mask out (if any); if none is passed, the default
    uses a random seed of 0.
    """
    
    if rng is None:
        rng = np.random.RandomState(0)
    
    # Assemble volumes and corresponding segmentations; split train/valid/test.
    volumes_h_1 = {'train': [], 'valid': [], 'test': []}
    volumes_s_1 = {'train': [], 'valid': [], 'test': []}
    volumes_h_2 = {'train': [], 'valid': [], 'test': []}
    volumes_s_2 = {'train': [], 'valid': [], 'test': []}
    volumes_m_1 = {'train': [], 'valid': [], 'test': []}
    volumes_m_2 = {'train': [], 'valid': [], 'test': []}
    indices_h_1 = {'train': [], 'valid': [], 'test': []}
    indices_s_1 = {'train': [], 'valid': [], 'test': []}
    indices_h_2 = {'train': [], 'valid': [], 'test': []}
    indices_s_2 = {'train': [], 'valid': [], 'test': []}
    try:
        h5py_file = h5py.File(path, mode='r')
    except:
        print("Failed to open data: {}".format(path))
        raise
    indices = []
    case_identities = []
    for idx, case_id in enumerate(h5py_file.keys()):   # Per patient.
        indices.append(idx)
        case_identities.append(case_id)
    c = list(zip(indices,  case_identities))
    # shuffle to dissociate same t1 and t2
    rng.shuffle(c)
    for idx, case_id in c:

        f = h5py_file[case_id]
        if idx in validation_indices:
            split = 'valid'
        elif idx in testing_indices:
            split = 'test'
        else:
            split = 'train'
        volumes_h_1[split].append(f['healthy_'+modalities[0]])
        volumes_s_1[split].append(f['sick_'+modalities[0]])
        volumes_m_1[split].append(f['segmentation'])
        indices_h_1[split].append(f['h_indices'])
        indices_s_1[split].append(f['s_indices'])
    
    rng.shuffle(c)
    for idx, case_id in c:

        f = h5py_file[case_id]
        if idx in validation_indices:
            split = 'valid'
        elif idx in testing_indices:
            split = 'test'
        else:
            split = 'train'
        volumes_h_2[split].append(f['healthy_'+modalities[1]])
        volumes_s_2[split].append(f['sick_'+modalities[1]])
        volumes_m_2[split].append(f['segmentation'])
        indices_h_2[split].append(f['h_indices'])
        indices_s_2[split].append(f['s_indices'])
    
    if masked_fraction_1!=1:
        # Source
        # Volumes with these indices will either be dropped from the training
        # set or have their segmentations set to None.
        # 
        # The `masked_fraction` determines the maximal fraction of slices that
        # are to be thus removed. All or none of the slices are selected for 
        # each volume.
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_m_1['train']])
        num_masked_slices = 0
        max_masked_slices = int(min(num_total_slices,
                                    num_total_slices*masked_fraction_1+0.5))
        for i in rng.permutation(len(volumes_m_1['train'])):
            num_slices = len(volumes_m_1['train'][i])
            if num_slices>0 and num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            if num_slices+num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            masked_indices.append(i)
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices)+modalities[0]+" slices are labeled across {} "
              "volumes ({:.1f}%).".format(len(volumes_m_1['train'])-len(masked_indices),
                        100*(1-num_masked_slices/float(num_total_slices))))
    else : 
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_m_1['train']])
        num_masked_slices = 0
        for i in rng.permutation(len(volumes_m_1['train'])):
            num_slices = len(volumes_m_1['train'][i])
            masked_indices.append(i)
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices,
                        )+modalities[0]+" slices are labeled"
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
    volumes_h_train = []
    volumes_s_train = []
    volumes_m_train = []
    indices_h_train = []
    indices_s_train = []
    for i in range(len(volumes_m_1['train'])):
        if i in masked_indices:
            # Mask out or drop.
            if drop_masked:
                continue    # Drop.
            volumes_m_train.append(np.array([None]*len(volumes_m_1['train'][i])))
        else:
            # Keep.
            volumes_m_train.append(volumes_m_1['train'][i])
        volumes_h_train.append(volumes_h_1['train'][i])
        volumes_s_train.append(volumes_s_1['train'][i])
        indices_h_train.append(indices_h_1['train'][i])
        indices_s_train.append(indices_s_1['train'][i])
    volumes_h_1['train'] = volumes_h_train
    volumes_s_1['train'] = volumes_s_train
    volumes_m_1['train'] = volumes_m_train
    indices_h_1['train'] = indices_h_train
    indices_s_1['train'] = indices_s_train

    if masked_fraction_2!=1:
        # Target
        # Volumes with these indices will either be dropped from the training
        # set or have their segmentations set to None.
        # 
        # The `masked_fraction` determines the maximal fraction of slices that
        # are to be thus removed. All or none of the slices are selected for 
        # each volume.
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_m_2['train']])
        num_masked_slices = 0
        max_masked_slices = int(min(num_total_slices,
                                    num_total_slices*masked_fraction_2+0.5))
        for i in rng.permutation(len(volumes_m_2['train'])):
            num_slices = len(volumes_m_2['train'][i])
            if num_slices>0 and num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            if num_slices+num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            masked_indices.append(i)
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices)+modalities[1]+" slices are labeled across {} "
              "volumes ({:.1f}%).".format(len(volumes_m_2['train'])-len(masked_indices),
                        100*(1-num_masked_slices/float(num_total_slices))))
    else : 
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_m_2['train']])
        num_masked_slices = 0
        for i in rng.permutation(len(volumes_m_2['train'])):
            masked_indices.append(i)
            num_slices = len(volumes_m_2['train'][i])
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices,
                        )+modalities[1]+" slices are labeled"
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
    volumes_h_train = []
    volumes_s_train = []
    volumes_m_train = []
    indices_h_train = []
    indices_s_train = []
    for i in range(len(volumes_m_2['train'])):
        if i in masked_indices:
            # Mask out or drop.
            if drop_masked:
                continue    # Drop.
            volumes_m_train.append(np.array([None]*len(volumes_m_2['train'][i])))
        else:
            # Keep.
            volumes_m_train.append(volumes_m_2['train'][i])
        volumes_h_train.append(volumes_h_2['train'][i])
        volumes_s_train.append(volumes_s_2['train'][i])
        indices_h_train.append(indices_h_2['train'][i])
        indices_s_train.append(indices_s_2['train'][i])
    volumes_h_2['train'] = volumes_h_train
    volumes_s_2['train'] = volumes_s_train
    volumes_m_2['train'] = volumes_m_train
    indices_h_2['train'] = indices_h_train
    indices_s_2['train'] = indices_s_train
    
    # Merge all arrays in each list of arrays.
    data = OrderedDict([('train', OrderedDict()),
                        ('valid', OrderedDict()),
                        ('test', OrderedDict())])
    for key in data.keys():
        # HACK: we may have a situation where the number of sick examples
        # is greater than the number of healthy. In that case, we should
        # duplicate the healthy set M times so that it has a bigger size
        # than the sick set.
        m = 1
        len_h = sum([len(elem) for elem in volumes_h_1[key]])
        len_s = sum([len(elem) for elem in volumes_s_1[key]])
        if len_h < len_s:
            m = int(np.ceil(len_s / len_h))
        data[key]['h_1']  = multi_source_array(volumes_h_1[key]*m)
        data[key]['s_1']  = multi_source_array(volumes_s_1[key])
        data[key]['m_1']  = multi_source_array(volumes_m_1[key])
        data[key]['hi_1'] = multi_source_array(indices_h_1[key]*m)
        data[key]['si_1'] = multi_source_array(indices_s_1[key])
        data[key]['h_2']  = multi_source_array(volumes_h_2[key]*m)
        data[key]['s_2']  = multi_source_array(volumes_s_2[key])
        data[key]['m_2']  = multi_source_array(volumes_m_2[key])
        data[key]['hi_2'] = multi_source_array(indices_h_2[key]*m)
        data[key]['si_2'] = multi_source_array(indices_s_2[key])
    return data



def preprocessor_mbrats(data_augmentation_kwargs=None, label_warp=None,
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
        
        h_t1, s_t1, m_t1, hi_t1, si_t1, h_t2, s_t2, m_t2, hi_t2, si_t2 = inputs
       
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
            
        return h_t1, s_t1, m_t1, hi_t1, si_t1, h_t2, s_t2, m_t2, hi_t2, si_t2
        
    def process_batch(batch):
        # Process every element.
        elements = []
        for i in range(len(batch[0])):
            elem = process_element([b[i] for b in batch])
            elements.append(elem)
        out_batch = list(zip(*elements))
        return out_batch
    
    return process_batch

def prepare_attnet(path, modalities, masked_fraction_1=0, masked_fraction_2=0, rng=None):
    # Random 20% data split (10% validation, 10% testing).
    rnd_state = np.random.RandomState(0)
    indices = np.arange(0, 369)
    rnd_state.shuffle(indices)
    val  = indices[0:37]
    test = indices[37:74]
    return _prepare_attnet(path,
                                 modalities=modalities,
                               validation_indices=val,
                               testing_indices=test,
                               masked_fraction_1=masked_fraction_1, masked_fraction_2=masked_fraction_2, 
                               rng=rng)

def _prepare_attnet(path, modalities, validation_indices, testing_indices, 
                          masked_fraction_1=0, masked_fraction_2=0, rng=None):
    """
    Convenience function to prepare multimodal brats data as multi_source_array objects for attent,
    split into training and validation subsets.
    
    path (string) : Path of the h5py file containing the data
    modalities (list of strings) : ["source","target"] contrasts picked for experiment
    masked_fraction_2 (float) : The fraction in [0, 1.] of source volumes in the 
        training set for which  to return segmentation masks as None
    masked_fraction_1 (float) : The fraction in [0, 1.] of source volumes in the 
        training set for which  to return segmentation masks as None
    drop_masked (bool) : Whether to omit volumes with "masked" segmentations.
    rng (numpy RandomState) : Random number generator.
    
    NOTE: A constant random seed (0) is always used to determine the training/
    validation/testing split. The rng passed for data preparation is used to 
    determine which labels to mask out (if any); if none is passed, the default
    uses a random seed of 0.
    """
    h5py_file = h5py.File(path, mode='r')
 
    if rng is None:
        rng = np.random.RandomState(0)
    
    # Assemble volumes and corresponding segmentations; split train/valid/test.
    
    volumes_1 = {'train': [], 'valid': [], 'test': []}
    volumes_2 = {'train': [], 'valid': [], 'test': []}
    volumes_translated_1 = {'train': [], 'valid': [], 'test': []}
    volumes_translated_2 = {'train': [], 'valid': [], 'test': []}
    volumes_m_1 = {'train': [], 'valid': [], 'test': []}
    volumes_m_2 = {'train': [], 'valid': [], 'test': []}
    
    indices = []
    case_identities = []
    for idx, case_id in enumerate(h5py_file.keys()):   # Per patient.
        indices.append(idx)
        case_identities.append(case_id)
    c = list(zip(indices,  case_identities))
    
    # shuffle to dissociate same t1 and t2
    rng.shuffle(c)
    for idx, case_id in c:
        f = h5py_file[case_id]
        if idx in validation_indices:
            split = 'valid'
        elif idx in testing_indices:
            split = 'test'
        else:
            split = 'train'
        volumes_1[split].append(f['sick_1'])
        volumes_m_1[split].append(f['segmentation'])
        volumes_translated_1[split].append(f['translated_sick_1'])
    
    rng.shuffle(c)
    for idx, case_id in c:
        f = h5py_file[case_id]
        if idx in validation_indices:
            split = 'valid'
        elif idx in testing_indices:
            split = 'test'
        else:
            split = 'train'
            
        volumes_2[split].append(f['sick_2'])
        volumes_translated_2[split].append(f['translated_sick_2'])
        volumes_m_2[split].append(f['segmentation'])

        
        
        
    if masked_fraction_1!=1:
        # t1
        # Volumes with these indices will either be dropped from the training
        # set or have their segmentations set to None.
        # 
        # The `masked_fraction` determines the maximal fraction of slices that
        # are to be thus removed. All or none of the slices are selected for 
        # each volume.
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_m_1['train']])
        num_masked_slices = 0
        max_masked_slices = int(min(num_total_slices,
                                    num_total_slices*masked_fraction_1+0.5))
        for i in rng.permutation(len(volumes_m_1['train'])):
            num_slices = len(volumes_m_1['train'][i])
            if num_slices>0 and num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            if num_slices+num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            masked_indices.append(i)
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices)+modalities[0]+" slices are labeled across {} "
              "volumes ({:.1f}%).".format(len(volumes_m_1['train'])-len(masked_indices),
                        100*(1-num_masked_slices/float(num_total_slices))))
    else : 
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_m_1['train']])
        num_masked_slices = 0
        for i in rng.permutation(len(volumes_m_1['train'])):
            num_slices = len(volumes_m_1['train'][i])
            masked_indices.append(i)
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices,
                        )+modalities[0]+" slices are labeled"
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
    volumes_1_train = []
    volumes_m_1_train = []
    volumes_translated_1_train = []

    for i in range(len(volumes_m_1['train'])):
        if i in masked_indices:
            volumes_m_1_train.append(np.array([None]*len(volumes_m_1['train'][i])))
        else:
            # Keep.
            volumes_m_1_train.append(volumes_m_1['train'][i])
        volumes_1_train.append(volumes_1['train'][i])
        volumes_translated_1_train.append(volumes_translated_1['train'][i])

    volumes_m_1['train'] = volumes_m_1_train
    volumes_1['train'] = volumes_1_train
    volumes_translated_1['train'] = volumes_translated_1_train

    if masked_fraction_2!=1:
        # t2
        # Volumes with these indices will either be dropped from the training
        # set or have their segmentations set to None.
        # 
        # The `masked_fraction` determines the maximal fraction of slices that
        # are to be thus removed. All or none of the slices are selected for 
        # each volume.
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_m_2['train']])
        num_masked_slices = 0
        max_masked_slices = int(min(num_total_slices,
                                    num_total_slices*masked_fraction_2+0.5))
        for i in rng.permutation(len(volumes_m_2['train'])):
            num_slices = len(volumes_m_2['train'][i])
            if num_slices>0 and num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            if num_slices+num_masked_slices >= max_masked_slices:
                continue    # Stop masking non-empty volumes (mask empty).
            masked_indices.append(i)
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices)+modalities[1]+" slices are labeled across {} "
              "volumes ({:.1f}%).".format(len(volumes_m_2['train'])-len(masked_indices),
                        100*(1-num_masked_slices/float(num_total_slices))))
    else : 
        masked_indices = []
        num_total_slices = sum([len(v) for v in volumes_m_2['train']])
        num_masked_slices = 0
        for i in rng.permutation(len(volumes_m_2['train'])):
            masked_indices.append(i)
            num_slices = len(volumes_m_2['train'][i])
            num_masked_slices += num_slices
        print("DEBUG: A total of {}/{} ".format(num_total_slices-num_masked_slices,
                        num_total_slices,
                        )+modalities[1]+" slices are labeled"
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
    volumes_2_train = []
    volumes_m_2_train = []
    volumes_translated_2_train = []

    for i in range(len(volumes_m_2['train'])):
        if i in masked_indices:
            volumes_m_2_train.append(np.array([None]*len(volumes_m_2['train'][i])))
        else:
            # Keep.
            volumes_m_2_train.append(volumes_m_2['train'][i])
        volumes_2_train.append(volumes_2['train'][i])
        volumes_translated_2_train.append(volumes_translated_2['train'][i])

    volumes_m_2['train'] = volumes_m_2_train
    volumes_2['train'] = volumes_2_train
    volumes_translated_2['train'] = volumes_translated_2_train 
        
    # Merge all arrays in each list of arrays.
    data = OrderedDict([('train', OrderedDict()),
                        ('valid', OrderedDict()),
                        ('test', OrderedDict())])
    
    for key in data.keys():
        
        if masked_fraction_1 > masked_fraction_2: #(source = 2)
            data[key]['s_1']  = multi_source_array(volumes_2[key])
            data[key]['m_1']  = multi_source_array(volumes_m_2[key])
            data[key]['translated_s_1']  = multi_source_array(volumes_translated_2[key])
            
            data[key]['s_2']  = multi_source_array(volumes_1[key])  
            data[key]['m_2']  = multi_source_array(volumes_m_1[key])
            data[key]['translated_s_2']  = multi_source_array(volumes_translated_1[key])
        
        else: #(source = 1)
            data[key]['s_1']  = multi_source_array(volumes_1[key])
            data[key]['m_1']  = multi_source_array(volumes_m_1[key])
            data[key]['translated_s_1']  = multi_source_array(volumes_translated_1[key])

            data[key]['s_2']  = multi_source_array(volumes_2[key])
            data[key]['m_2']  = multi_source_array(volumes_m_2[key])
            data[key]['translated_s_2']  = multi_source_array(volumes_translated_2[key])
            
    return data

def preprocessor_attnet(data_augmentation_kwargs=None, mgenseg=False):
    """
    Preprocessor function to pass to a data_flow, for BRATS data.
    
    data_augmentation_kwargs : Dictionary of keyword arguments to pass to
        the data augmentation code (image_stack_random_transform).
    """
        
    def process_element(inputs):
        s_t1, m_t1, s_t2, translated_s_t2, m_t2 = inputs
        
        # Float.
        s_t1 = s_t1.astype(np.float32)
        s_t2 = s_t2.astype(np.float32)
        translated_s_t2 = translated_s_t2.astype(np.float32)
        
        if mgenseg :
            # Data augmentation.
            if data_augmentation_kwargs is not None:
            
                if s_t2 is not None:
                    s_t2, m_t1 = image_random_transform(s_t2, m_t2, **data_augmentation_kwargs,
                                               n_warp_threads=1)

                    
                if translated_s_t2 is not None:
                    translated_s_t2, m_t2 = image_random_transform(translated_s_t2, m_t2, **data_augmentation_kwargs,
                                               n_warp_threads=1)
               
        else :
            # Data augmentation.
            if data_augmentation_kwargs is not None:
            
                if s_t1 is not None:
                    _ = image_random_transform(s_t1, m_t1, **data_augmentation_kwargs,
                                               n_warp_threads=1)
                if m_t1 is not None:
                    assert s_t1 is not None
                    s_t1, m_t1 = _
                else:
                    s_t1 = _
                    
                if translated_s_t2 is not None:
                    _ = image_random_transform(translated_s_t2, m_t2, **data_augmentation_kwargs,
                                               n_warp_threads=1)
                if m_t2 is not None:
                    assert translated_s_t2 is not None
                    translated_s_t2, m_t2 = _
                else:
                    translated_s_t2 = _              

        # Remove distant outlier intensities.
        if s_t1 is not None:
            s_t1 = np.clip(s_t1, -1., 1.)
        if s_t2 is not None:
            s_t2 = np.clip(s_t2, -1., 1.)  
        if translated_s_t2 is not None:
            translated_s_t2 = np.clip(translated_s_t2, -1., 1.)  
            
        return s_t1, m_t1, s_t2, translated_s_t2, m_t2
        
    def process_batch(batch):
        # Process every element.
        elements = []
        for i in range(len(batch[0])):
            elem = process_element([b[i] for b in batch])
            elements.append(elem)
        out_batch = list(zip(*elements))
        return out_batch
    
    return process_batch

def nnunet_transform(img, seg=None):
    # Based on `data_augmentation_insaneDA2.py` and on "data_aug_params"
    # extracted from the pretrained BRATS nnunet:
    #
    #"data_aug_params":
    #"{
    #'selected_data_channels': None,
    #'selected_seg_channels': [0],
    #'do_elastic': True,
    #'elastic_deform_alpha': (0.0, 900.0),
    #'elastic_deform_sigma': (9.0, 13.0),
    #'p_eldef': 0.3,
    #'do_scaling': True, 
    #'scale_range': (0.65, 1.6),
    #'independent_scale_factor_for_each_axis': True,
    #'p_independent_scale_per_axis': 0.3,
    #'p_scale': 0.3,
    #'do_rotation': True,
    #'rotation_x': (-0.5235987755982988, 0.5235987755982988),
    #'rotation_y': (-0.5235987755982988, 0.5235987755982988),
    #'rotation_z': (-0.5235987755982988, 0.5235987755982988),
    #'rotation_p_per_axis': 1,
    #'p_rot': 0.3,
    #'random_crop': False,
    #'random_crop_dist_to_border': None,
    #'do_gamma': True,
    #'gamma_retain_stats': True,
    #'gamma_range': (0.5, 1.6),
    #'p_gamma': 0.3,
    #'do_mirror': True,
    #'mirror_axes': (0, 1, 2),
    #'dummy_2D': False,
    #'mask_was_used_for_normalization': OrderedDict([(0, True), (1, True), (2, True), (3, True)]),
    #'border_mode_data': 'constant',
    #'all_segmentation_labels': None,
    #'move_last_seg_chanel_to_data': False,
    #'cascade_do_cascade_augmentations': False,
    #'cascade_random_binary_transform_p': 0.4,
    #'cascade_random_binary_transform_p_per_label': 1,
    #'cascade_random_binary_transform_size': (1, 8),
    #'cascade_remove_conn_comp_p': 0.2,
    #'cascade_remove_conn_comp_max_size_percent_threshold': 0.15,
    #'cascade_remove_conn_comp_fill_with_other_class_p': 0.0,
    #'do_additive_brightness': True,
    #'additive_brightness_p_per_sample': 0.3,
    #'additive_brightness_p_per_channel': 1,
    #'additive_brightness_mu': 0,
    #'additive_brightness_sigma': 0.2,
    #'num_threads': 24,
    #'num_cached_per_thread': 4,
    #'patch_size_for_spatialtransform': array([128, 128, 128]),
    #'eldef_deformation_scale': (0, 0.25)
    #}",
    #
    #
    # NOTE: scale has been reduced from (0.65, 1.6) to (0.9, 1.1) in order
    # to make sure that tumour is never removed from sick images.
    # 
    transforms = []
    transforms += [
        SpatialTransform_2(
            patch_size=None,
            do_elastic_deform=True,
            deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(-30/360*2*np.pi, 30/360*2*np.pi),
            angle_y=(-30/360*2*np.pi, 30/360*2*np.pi),
            angle_z=(-30/360*2*np.pi, 30/360*2*np.pi),
            do_scale=True,
            #scale=(0.65, 1.6),
            scale=(0.9, 1.1),
            border_mode_data='constant',
            border_cval_data=0,
            order_data=3,
            border_mode_seg='constant',
            border_cval_seg=0,
            order_seg=0,
            random_crop=False,
            p_el_per_sample=0.3,
            p_scale_per_sample=0.3,
            p_rot_per_sample=0.3,
            independent_scale_for_each_axis=False,
            p_independent_scale_per_axis=0.3
        )
    ]
    transforms += [GaussianNoiseTransform(p_per_sample=0.15)]
    transforms += [
        GaussianBlurTransform(
            (0.5, 1.5),
            different_sigma_per_channel=True,
            p_per_sample=0.2,
            p_per_channel=0.5
        )
    ]
    transforms += [
        BrightnessMultiplicativeTransform(
            multiplier_range=(0.70, 1.3),
            p_per_sample=0.15
        )
    ]
    transforms += [
        ContrastAugmentationTransform(
            contrast_range=(0.65, 1.5),
            p_per_sample=0.15
        )
    ]
    transforms += [
        SimulateLowResolutionTransform(
            zoom_range=(0.5, 1),
            per_channel=True,
            p_per_channel=0.5,
            order_downsample=0,
            order_upsample=3,
            p_per_sample=0.25,
            ignore_axes=None
        )
    ]
    transforms += [
        GammaTransform(     # This one really does appear twice...
            (0.5, 1.6),     # gamma_range
            True,
            True,
            retain_stats=True,
            p_per_sample=0.15   # Hardcoded.
        )
    ]
    transforms += [
        BrightnessTransform(
            0,      # additive_brightness_mu
            0.2,    # additive_brightness_sigma
            True,
            p_per_sample=0.3,
            p_per_channel=1
        )
    ]
    transforms += [
        GammaTransform(
            (0.5, 1.6),     # gamma_range
            False,
            True,
            retain_stats=True,
            p_per_sample=0.3    # Passed as param.
        )
    ]
    transforms += [MirrorTransform((1, 2))]  # mirror_axes
    full_transform = Compose(transforms)
    
    # Transform.
    img_input = img[None]
    seg_input = seg
    if seg is not None:
        seg_input = seg[None]
    out = full_transform(data=img_input, seg=seg_input)
    img_output = out['data'][0]
    if seg is None:
        return img_output
    seg_output = out['seg'][0]
    return img_output, seg_output

def nnunet_transform_default(img, seg=None):
    #
    #"data_aug_params": "{'selected_data_channels': None,
    # 'selected_seg_channels': [0],
    # 'do_elastic': False,
    # 'elastic_deform_alpha': (0.0, 200.0),
    # 'elastic_deform_sigma': (9.0, 13.0),
    # 'do_scaling': True,
    # 'scale_range': (0.7, 1.4),
    # 'do_rotation': True,
    # 'rotation_x': (-3.141592653589793, 3.141592653589793),
    # 'rotation_y': (-0.0, 0.0),
    # 'rotation_z': (-0.0, 0.0),
    # 'random_crop': False,
    # 'random_crop_dist_to_border':  # None,
    # 'do_gamma': True,
    # 'gamma_retain_stats': True,
    # 'gamma_range': (0.7, 1.5),
    # 'p_gamma': 0.3,
    # 'num_threads': 12,
    # 'num_cached_per_thread': 1,
    # 'do_mirror': True,
    # 'mirror_axes': (0, 1),
    # 'p_eldef': 0.2,
    # 'p_scale': 0.2,
    # 'p_rot': 0.2,
    # 'dummy_2D': False,
    # 'mask_was_used_for_normalization': OrderedDict([(0, True), (1, True), (2, True), (3, True)]),
    # 'all_segmentation_labels': None,
    # 'move_last_seg_chanel_to_data': False,
    # 'border_mode_data': 'constant',
    # 'cascade_do_cascade_augmentations': False,
    # 'cascade_random_binary_transform_p': 0.4,
    # 'cascade_random_binary_transform_size': (1, 8),
    # 'cascade_remove_conn_comp_p': 0.2,
    # 'cascade_remove_conn_comp_max_size_percent_threshold': 0.15,
    # 'cascade_remove_conn_comp_fill_with_other_class_p': 0.0,
    # 'independent_scale_factor_for_each_axis': False,
    # 'patch_size_for_spatialtransform': array([192, 160])}",

    transforms = []
    transforms += [SpatialTransform(
        None,
        patch_center_dist_from_border=None,
        do_elastic_deform=False,
        alpha=(0.0, 200.0),
        sigma=(9.0, 13.0),
        do_rotation=True, angle_x=(-0.2617993877991494, 0.2617993877991494), angle_y=(-0.0, 0.0),
        angle_z=(-0.0, 0.0), p_rot_per_axis=1,
        do_scale=True, scale=(0.9, 1.1),
        border_mode_data='constant', border_cval_data=0, order_data=3,
        border_mode_seg="constant", border_cval_seg=0,
        order_seg=0, random_crop=False, p_el_per_sample=0.2,
        p_scale_per_sample=0.2, p_rot_per_sample=0.2,
        independent_scale_for_each_axis=True
    )]
    transforms += [
        GaussianNoiseTransform(p_per_sample=0.1),
        GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                              p_per_channel=0.5),
        BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15)
    ]
    transforms += [(ContrastAugmentationTransform(p_per_sample=0.15)),
                   SimulateLowResolutionTransform(zoom_range=(0.5, 1),
                                                  per_channel=True, p_per_channel=0.5,
                                                  order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                  ignore_axes=None)]
    transforms += [
        GammaTransform((0.7, 1.5),
                       True,
                       True,
                       retain_stats=True,
                       p_per_sample=0.1)
    ]

    transforms += [
        GammaTransform((0.7, 1.5),
                       False,
                       True,
                       retain_stats=True,
                       p_per_sample=0.3)
    ]

    transforms += [
        MirrorTransform((0,1))
    ]

    # transforms += [
    #     MaskTransform(OrderedDict([(0, False)]), mask_idx_in_seg=0, set_outside_to=0)
    # ]

    # TODO; check whether delete or not
    # transforms += [
    #    RemoveLabelTransform(-1, 0)
    # ]
    # transforms += [
    #    RenameTransform('seg', 'target', True)
    # ]

    full_transform = Compose(transforms)

    # Transform.
    img_input = img[None]
    seg_input = seg
    if seg is not None:
        seg_input = seg[None]
    out = full_transform(data=img_input, seg=seg_input)
    img_output = out['data'][0]
    if seg is None:
        return img_output
    seg_output = out['seg'][0]
    return img_output, seg_output



def preprocessor_transunet(data_augmentation_kwargs=None, label_warp=None,
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
    
    def resize(image, size, interpolator=sitk.sitkLinear):
        sitk_image = sitk.GetImageFromArray(image)
        new_spacing = [x*y/z for x, y, z in zip(
                       sitk_image.GetSpacing(),
                       sitk_image.GetSize(),
                       size)]
        sitk_out = sitk.Resample(sitk_image,
                                 size,
                                 sitk.Transform(),
                                 interpolator,
                                 sitk_image.GetOrigin(),
                                 new_spacing,
                                 sitk_image.GetDirection(),
                                 0,
                                 sitk_image.GetPixelID())
        out = sitk.GetArrayFromImage(sitk_out)
        return out
        
    def process_element(inputs):
        h, s, m, hi, si = inputs
        
        # Set up rng.
        if m is not None:
            seed = abs(hash(m.data.tobytes()))//2**32
            rng = np.random.RandomState(seed)
        
        # Drop mask.
        if m is not None:
            if rng.choice([True, False], p=[label_dropout, 1-label_dropout]):
                m = None
        
        # Crop mask.
        if m is not None and (   label_crop_rand is not None
                              or label_crop_rand2 is not None
                              or label_crop_left is not None):
            m_out = m.copy()
            m_dilated = ndimage.morphology.binary_dilation(m)
            m_labeled, n_obj = ndimage.label(m_dilated)
            for bbox in ndimage.find_objects(m_labeled):
                _, row, col = bbox
                if label_crop_rand is not None:
                    r = int(label_crop_rand*(row.stop-row.start))
                    c = int(label_crop_rand*(col.stop-col.start))
                    row_a = rng.randint(row.start, row.stop+1-r)
                    row_b = rng.randint(row_a+r, row.stop+1)
                    col_a = rng.randint(col.start, col.stop+1-c)
                    col_b = rng.randint(col_a+c, col.stop+1)
                    m_out[:, row_a:row_b, col_a:col_b] = 0
                if label_crop_rand2 is not None:
                    def get_p(n):
                        mu = int(label_crop_rand2*n+0.5)     # mean
                        m = (12*mu-6*n)/(n*(n+1)*(n+2))     # slope
                        i = 1/(n+1)-m*n/2                   # intersection
                        p = np.array([max(x*m+i, 0) for x in range(n+1)])
                        p = p/p.sum()   # Precision errors can make p.sum() > 1
                        return p
                    width  = row.stop - row.start
                    height = col.stop - col.start
                    box_width  = rng.choice(range(width+1),  p=get_p(width))
                    box_height = rng.choice(range(height+1), p=get_p(height))
                    box_row_start = rng.randint(row.start,
                                                row.stop+1-box_width)
                    box_col_start = rng.randint(col.start,
                                                col.stop+1-box_height)
                    row_slice = slice(box_row_start, box_row_start+box_width)
                    col_slice = slice(box_col_start, box_col_start+box_height)
                    m_out[:, row_slice, col_slice] = 0
                if label_crop_left is not None:
                    crop_size = int(label_crop_left*(col.stop-col.start))
                    m_out[:, row, col.start:col.start+crop_size] = 0
            m = m_out
        
        # Float.
        h = h.astype(np.float32)
        s = s.astype(np.float32)
        
        # Data augmentation.
        if data_augmentation_kwargs is not None:
            if h is not None:
                h = image_random_transform(h, **data_augmentation_kwargs,
                                           n_warp_threads=1)
            if s is not None:
                _ = image_random_transform(s, m, **data_augmentation_kwargs,
                                           n_warp_threads=1)
            if m is not None:
                assert s is not None
                s, m = _
            else:
                s = _
        
        # Corrupt the mask by warping it.
        if label_warp is not None:
            if m is not None:
                m = image_random_transform(m,
                                           spline_warp=True,
                                           warp_sigma=label_warp,
                                           warp_grid_size=3,
                                           n_warp_threads=1)
        if label_shift is not None:
            if m is not None:
                m_shift = np.zeros(m.shape, dtype=m.dtype)
                m_shift[:,label_shift:,:] = m[:,:-label_shift,:]
                m = m_shift
        
        # Remove distant outlier intensities.
        if h is not None:
            h = np.clip(h, -1., 1.)
        if s is not None:
            s = np.clip(s, -1., 1.)

        h = np.pad(h, ((0,0),(8,8),(68,68)))
        
        s = np.pad(s, ((0,0),(8,8),(68,68)))
        
        if m is not None:
        
            m = np.pad(m, ((0,0),(8,8),(68,68)))
            m = (m > 0)*1
        
        return h, s, m, hi, si
        
    def process_batch(batch):
        # Process every element.
        elements = []
        for i in range(len(batch[0])):
            elem = process_element([b[i] for b in batch])
            elements.append(elem)
        out_batch = list(zip(*elements))
        return out_batch
    
    return process_batch
