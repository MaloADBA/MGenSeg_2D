from __future__ import (print_function,
                        division)
import argparse
import json
from utils.dispatch import (dispatch,
                            dispatch_argument_parser)


'''
Process arguments.
'''
def get_parser():
    parser = dispatch_argument_parser(description="AttNET Trans.")
    g_exp = parser.add_argument_group('Experiment')
    g_exp.add_argument('--data', type=str, default='./Extended GenSeg/T1-T2 dataset/')
    g_exp.add_argument('--path', type=str, default='./experiments')
    g_exp.add_argument('--model_from', type=str, default=None)
    g_exp.add_argument('--model_kwargs', type=json.loads, default=None)
    g_exp.add_argument('--weights_from', type=str, default=None)
    g_exp.add_argument('--weight_decay', type=float, default=1e-4)
    g_exp.add_argument('--source_modality', type=str, default='t1')
    g_exp.add_argument('--target_modality', type=str, default='t2')    
    g_exp_da = g_exp.add_mutually_exclusive_group()
    g_exp_da.add_argument('--augment_data', action='store_true')
    g_exp.add_argument('--batch_size_train', type=int, default=20)
    g_exp.add_argument('--batch_size_valid', type=int, default=20)
    g_exp.add_argument('--epochs', type=int, default=200)
    g_exp.add_argument('--learning_rate', type=json.loads, default=0.001)
    g_exp.add_argument('--opt_kwargs', type=json.loads, default=None)
    g_exp.add_argument('--optimizer', type=str, default='amsgrad')
    g_exp.add_argument('--n_vis', type=int, default=10)
    g_exp.add_argument('--nb_io_workers', type=int, default=1)
    g_exp.add_argument('--nb_proc_workers', type=int, default=1)
    g_exp.add_argument('--save_image_events', action='store_true',
                       help="Save images into tensorboard event files.")
    g_exp.add_argument('--init_seed', type=int, default=1234)
    g_exp.add_argument('--data_seed', type=int, default=0)
    g_exp.add_argument('--label_warp', type=float, default=None,
                       help="The sigma value of the spline warp applied to "
                            "to the target label mask during training in "
                            "order to corrupt it. Used for testing "
                            "robustness to label noise.")
    g_exp.add_argument('--label_shift', type=int, default=None,
                       help="The number of pixels to shift every training "
                            "target mask. Used for testing robustness to "
                            "label noise.")
    g_exp.add_argument('--label_dropout', type=float, default=0,
                       help="The probability of randomly dropping a reference "
                            "segmentation mask in the training set.")
    g_exp.add_argument('--label_permutation', type=float, default=0,
                       help="The fraction of training slices for which labels "
                            "are mismatched via permutation.")
    g_exp.add_argument('--label_crop_rand', type=float, default=None,
                       help="Crop out a randomly sized rectangle out of "
                            "every connected component of the mask during "
                            "training. The minimum size of the rectangle is "
                            "set as a fraction of the connected component's "
                            "bounding box.")
    g_exp.add_argument('--label_crop_rand2', type=float, default=None)
    g_exp.add_argument('--label_crop_left', type=float, default=None,
                       help="Crop out the left fraction of every connected "
                            "component of the mask during training.")
    return parser


def run(args):
    from collections import OrderedDict
    from functools import partial
    import os
    import re
    import shutil
    import subprocess
    import sys
    import warnings

    import numpy as np
    import torch
    from torch.autograd import Variable
    import ignite
    from ignite.engine import (Events,
                               Engine)
    from ignite.handlers import ModelCheckpoint

    from data_tools.io import data_flow
    from data_tools.data_augmentation import image_random_transform

    from utils.data.multimodal_brats import (prepare_mbrats,
                                  preprocessor_mbrats)
    from utils.data.common import (data_flow_sampler,
                                   permuted_view)

    from utils.experiment import experiment
    from utils.metrics import (batchwise_loss_accumulator,
                               dice_global)
    from utils.trackers import(image_logger,
                               scoring_function,
                               summary_tracker)

    from model import configs
    from model.attnet_translation import translation_model as trans_model


    # Disable buggy profiler.
    torch.backends.cudnn.benchmark = True
    
    # Set up experiment.
    experiment_state = experiment(args)
    args = experiment_state.args
    torch.manual_seed(args.init_seed)
    
    # Data augmentation settings.
    da_kwargs = {'rotation_range': 3.,
                 'zoom_range': 0.1,
                 'intensity_shift_range': 0.1,
                 'horizontal_flip': True,
                 'vertical_flip': True,
                 'fill_mode': 'reflect',
                 'spline_warp': True,
                 'warp_sigma': 5,
                 'warp_grid_size': 3}

    
    # Prepare data.
    data = prepare_mbrats(path=os.path.join(args.data, "data.h5"),
                              modalities = [args.source_modality, args.target_modality],                          
                              masked_fraction_1=0,
                              masked_fraction_2=0,
                              rng=np.random.RandomState(args.data_seed))

    get_data_list = lambda key : [data[key]['h_1'],
                                  data[key]['s_1'],
                                  data[key]['m_1'],
                                  data[key]['hi_1'],
                                  data[key]['si_1'],
                                  data[key]['h_2'],
                                  data[key]['s_2'],
                                  data[key]['m_2'],
                                  data[key]['hi_2'],
                                  data[key]['si_2']]
    loader = {
        'train': data_flow_sampler(get_data_list('train'),
                                   sample_random=True,
                                   batch_size=args.batch_size_train,
                                   preprocessor=preprocessor_mbrats(
                                       data_augmentation_kwargs=da_kwargs,
                                       label_warp=args.label_warp,
                                       label_shift=args.label_shift,
                                       label_dropout=args.label_dropout,
                                       label_crop_rand=args.label_crop_rand,
                                       label_crop_rand2=args.label_crop_rand2,
                                       label_crop_left=args.label_crop_left),
                                   nb_io_workers=args.nb_io_workers,
                                   nb_proc_workers=args.nb_proc_workers,
                                   rng=np.random.RandomState(args.init_seed)),
        'valid': data_flow_sampler(get_data_list('valid'),
                                   sample_random=True,
                                   batch_size=args.batch_size_valid,
                                   preprocessor=preprocessor_mbrats(),
                                   nb_io_workers=args.nb_io_workers,
                                   nb_proc_workers=args.nb_proc_workers,
                                   rng=np.random.RandomState(args.init_seed)),
        'test':  data_flow_sampler(get_data_list('test'),
                                   sample_random=True,
                                   batch_size=args.batch_size_valid,
                                   preprocessor=preprocessor_mbrats(),
                                   nb_io_workers=args.nb_io_workers,
                                   nb_proc_workers=args.nb_proc_workers,
                                   rng=np.random.RandomState(args.init_seed))}
    
    # Function to convert data to pytorch usable form.
    def prepare_batch(batch):
        h_1, s_1, m_1, hi_1, si_1, h_2, s_2, m_2, hi_2, si_2 = batch
        # Prepare for pytorch.
        s_1 = Variable(torch.from_numpy(np.array(s_1))).cuda()
        s_2 = Variable(torch.from_numpy(np.array(s_2))).cuda()
        return s_1, s_2
    
    # Helper for training/validation loops : detach variables from graph.
    def detach(x):
        detached = OrderedDict([(k, v.detach())
                                if isinstance(v, Variable)
                                else (k, v)
                                for k, v in x.items()])
        return detached
    
    # Training loop.
    def training_function(engine, batch):
        for model in experiment_state.model.values():
            model.train()
        S, T = prepare_batch(batch)

        outputs = experiment_state.model['G'](x_S=S, x_T=T, optimizer=experiment_state.optimizer)
        outputs = detach(outputs)

        return outputs
    
    # Validation loop.
    def validation_function(engine, batch):
        for model in experiment_state.model.values():
            model.eval()
        S, T = prepare_batch(batch)
        with torch.no_grad():
            outputs = experiment_state.model['G'](x_S=S, x_T=T, rng=engine.rng)
        outputs = detach(outputs)
        return outputs
    
    # Get engines.
    engines = {}
    engines['train'] = experiment_state.setup_engine(
                                            training_function,
                                            epoch_length=len(loader['train']))
    engines['valid'] = experiment_state.setup_engine(
                                            validation_function,
                                            prefix='val',
                                            epoch_length=len(loader['valid']))
    engines['test'] = experiment_state.setup_engine(
                                            validation_function,
                                            prefix='test',
                                            epoch_length=len(loader['test']))
    for key in ['valid', 'test']:
        engines[key].add_event_handler(
            Events.STARTED,
            lambda engine: setattr(engine, 'rng', np.random.RandomState(0)))
    
    
    # Set up metrics.
    metrics = {}
        
    for key in engines:
        metrics[key] = OrderedDict()

        metrics[key]['rec_S']  = batchwise_loss_accumulator(
                        output_transform=lambda x: x['l_rec_S'])
        metrics[key]['rec_T']  = batchwise_loss_accumulator(
                        output_transform=lambda x: x['l_rec_T'])
        metrics[key]['rec']  = batchwise_loss_accumulator(
                        output_transform=lambda x: x['l_rec'])        
        metrics[key]['G']    = batchwise_loss_accumulator(
                        output_transform=lambda x: x['l_G'])
        metrics[key]['S']    = batchwise_loss_accumulator(
                        output_transform=lambda x: x['l_DS'])
        metrics[key]['T']    = batchwise_loss_accumulator(
                        output_transform=lambda x: x['l_DT'])
        for name, m in metrics[key].items():
            m.attach(engines[key], name=name)

    # Set up validation.
    engines['train'].add_event_handler(Events.EPOCH_COMPLETED,
                             lambda _: engines['valid'].run(loader['valid']))
    
    # Set up model checkpointing.
    score_function = scoring_function('rec')
    experiment_state.setup_checkpoints(engines['train'], engines['valid'],
                                       score_function=score_function)
    
    # Set up tensorboard logging for losses.
    tracker = summary_tracker(experiment_state.experiment_path,
                              initial_epoch=experiment_state.get_epoch())
    tracker.attach(
        engine=engines["valid"],
        prefix="valid",
        output_transform=lambda x: dict([(k, v)
                                         for k, v in x.items()
                                         if k.startswith('l_')]),
        metric_keys=['G'])
    
    tracker.attach(
        engine=engines["train"],
        prefix="train",
        output_transform=lambda x: dict([(k, v)
                                         for k, v in x.items()
                                         if k.startswith('l_')]),
        metric_keys=['G'])
    
    # Set up image logging.
    def output_transform(output):
        transformed = OrderedDict()
        for k, v in output.items():
            if k.startswith('x_') and v is not None:
                k_new = k.replace('x_','')
                v_new = v.cpu().numpy()
                v_new = v_new[:,0]         
                transformed[k_new] = v_new
        return transformed
    
    save_image = image_logger(
        initial_epoch=experiment_state.get_epoch(),
        directory=os.path.join(experiment_state.experiment_path, "images"),
        summary_tracker=(tracker if args.save_image_events else None),
        num_vis=args.n_vis,
        suffix='Images',
        output_name='Images',
        output_transform=output_transform,
        fontsize=40)
    save_image.attach(engines['valid'])
    
    '''
    Train.
    '''
    engines['train'].run(loader['train'], max_epochs=args.epochs)

if __name__ == '__main__':
    parser = get_parser()
    dispatch(parser, run)

