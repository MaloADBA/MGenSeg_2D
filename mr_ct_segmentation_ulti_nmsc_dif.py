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
    parser = dispatch_argument_parser(description="BRATS seg.")
    g_exp = parser.add_argument_group('Experiment')
    g_exp.add_argument('--data', type=str, default='./Extended GenSeg/T1-T2 dataset/')
    g_exp.add_argument('--data_mr', type=str, default='./Extended GenSeg/T1-T2 dataset/')
    g_exp.add_argument('--data_ct', type=str, default='./Extended GenSeg/T1-T2 dataset/')    
    g_exp.add_argument('--path', type=str, default='./experiments')
    g_exp.add_argument('--model_from', type=str, default=None)
    g_exp.add_argument('--model_kwargs', type=json.loads, default=None)
    g_exp.add_argument('--weights_from', type=str, default=None)
    g_exp.add_argument('--weight_decay', type=float, default=1e-4)
    g_exp.add_argument('--labeled_fraction_source', type=float, default=0.01)
    g_exp.add_argument('--labeled_fraction_target', type=float, default=0)
    g_exp.add_argument('--yield_only_labeled', action='store_true')
    g_exp_da = g_exp.add_mutually_exclusive_group()
    g_exp_da.add_argument('--augment_data', action='store_true')
    g_exp_da.add_argument('--augment_data_nnunet', action='store_true')
    g_exp_da.add_argument('--augment_data_nnunet_default', action='store_true')
    g_exp_da.add_argument('--augment_data_nnunet_default_3d', action='store_true')
    g_exp.add_argument('--batch_size_train', type=int, default=20)
    g_exp.add_argument('--batch_size_valid', type=int, default=20)
    g_exp.add_argument('--epochs', type=int, default=200)
    g_exp.add_argument('--learning_rate', type=json.loads, default=0.001)
    g_exp.add_argument('--opt_kwargs', type=json.loads, default=None)
    g_exp.add_argument('--optimizer', type=str, default='amsgrad')
    g_exp.add_argument('--n_vis', type=int, default=20)
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

    from utils.data.multimodal_mr_ct_mgenseg import (prepare_mr_ct,
                                                    preprocessor_mr_ct)
    from utils.data.common import (data_flow_sampler,
                                   permuted_view)

    from utils.experiment import experiment
    from utils.metrics import (batchwise_loss_accumulator,
                               dice_global)
    from utils.trackers import(image_logger,
                               scoring_function,
                               summary_tracker)

    from model import configs
    from model.gen_segmentation_ulti_nmsc_dif import segmentation_model as model_ext


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
    data = prepare_mr_ct(path_mr = os.path.join(args.data_mr),
				                  path_ct = os.path.join(args.data_ct),    
                          masked_fraction_mr=1.-args.labeled_fraction_source,
                          masked_fraction_ct=1.-args.labeled_fraction_target,
                          drop_masked=args.yield_only_labeled,
                          rng=np.random.RandomState(args.data_seed))
    target_class = [1]

    get_data_list = lambda key : [data[key]['h_mr'],
                                  data[key]['sm_mr'],
                                  data[key]['h_ct'],
                                  data[key]['sm_ct']]
    loader = {
        'train': data_flow_sampler(get_data_list('train'),
                                   sample_random=True,
                                   batch_size=args.batch_size_train,
                                   preprocessor=preprocessor_mr_ct(
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
                                   preprocessor=preprocessor_mr_ct(),
                                   nb_io_workers=args.nb_io_workers,
                                   nb_proc_workers=args.nb_proc_workers,
                                   rng=np.random.RandomState(args.init_seed)),
        'test':  data_flow_sampler(get_data_list('test'),
                                   sample_random=True,
                                   batch_size=args.batch_size_valid,
                                   preprocessor=preprocessor_mr_ct(),
                                   nb_io_workers=args.nb_io_workers,
                                   nb_proc_workers=args.nb_proc_workers,
                                   rng=np.random.RandomState(args.init_seed))}
    
    # Function to convert data to pytorch usable form.
    def prepare_batch(batch, slice_conditional=False):
        h_1, s_1, m_1, h_2, s_2, m_2 = batch
        # Prepare for pytorch.
        h_1 = Variable(torch.from_numpy(np.array(h_1))).cuda()
        s_1 = Variable(torch.from_numpy(np.array(s_1))).cuda()
        h_2 = Variable(torch.from_numpy(np.array(h_2))).cuda()
        s_2 = Variable(torch.from_numpy(np.array(s_2))).cuda()
        # Provide slice index tuple if slice_conditional.
        return h_1, s_1, m_1, h_2, s_2, m_2
    
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
        SB, SA, SA_M, TB, TA, TA_M = prepare_batch(batch)

        outputs = experiment_state.model['G'](x_SA=SA, x_SB=SB, x_TA=TA, x_TB=TB, mask_S=SA_M, mask_T=TA_M,
                                         optimizer=experiment_state.optimizer)
        outputs = detach(outputs)
        
        # Drop images without labels, for visualization.
        indices = [i for i, m in enumerate(SA_M) if m is not None]
        for key in filter(lambda x: x.startswith('x_'), outputs.keys()):
            if outputs['x_SM'] is None:
                outputs[key] = None
            elif outputs[key] is not None and len(outputs)==len(SA_M):
                outputs[key] = outputs[key][indices]
        
        return outputs
    
    # Validation loop.
    def validation_function(engine, batch):
        for model in experiment_state.model.values():
            model.eval()
        SB, SA, SA_M, TB, TA, TA_M = prepare_batch(batch)
        with torch.no_grad():
            outputs = experiment_state.model['G'](x_SA=SA, x_SB=SB, x_TA=TA, x_TB=TB, mask_S=SA_M, mask_T=TA_M, rng=engine.rng)
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
    def dice_transform_all_TA(x):
        if x['x_TAM'] is not None and x['x_TAM'].size(1)!=1:
            return (1.-x['x_TAM'][:,0:1], x['x_TM'])  # 4 out; get whole lesion.
        return (x['x_TAM'], x['x_TM'])                # 1 out; get whole lesion.
    def dice_transform_all_SA(x):
        if x['x_SAM'] is not None and x['x_SAM'].size(1)!=1:
            return (1.-x['x_SAM'][:,0:1], x['x_SM'])  # 4 out; get whole lesion.
        return (x['x_SAM'], x['x_SM'])                # 1 out; get whole lesion.
    def dice_transform_all(x):
        SAM,SM=dice_transform_all_SA(x)
        TAM,TM=dice_transform_all_TA(x) 
        return (torch.cat((SAM,TAM),dim=0),torch.cat((SM,TM),dim=0))
        
    for key in engines:
        metrics[key] = OrderedDict()
        metrics[key]['dice_TA'] = dice_global(target_class=target_class,
                                           output_transform=dice_transform_all_TA)
        metrics[key]['dice_SA'] = dice_global(target_class=target_class,
                                           output_transform=dice_transform_all_SA)
        if key=='valid' or key=='test':
            metrics[key]['dice_global'] = dice_global(target_class=target_class,
                                               output_transform=dice_transform_all)
        
        metrics[key]['rec_img']  = batchwise_loss_accumulator(
                        output_transform=lambda x: x['l_rec_img'])
        metrics[key]['rec_features']  = batchwise_loss_accumulator(
                        output_transform=lambda x: x['l_rec_features'])
        metrics[key]['G']    = batchwise_loss_accumulator(
                        output_transform=lambda x: x['l_G'])
        metrics[key]['SA']    = batchwise_loss_accumulator(
                        output_transform=lambda x: x['l_DSA'])
        metrics[key]['SB']    = batchwise_loss_accumulator(
                        output_transform=lambda x: x['l_DSB'])
        metrics[key]['TA']    = batchwise_loss_accumulator(
                        output_transform=lambda x: x['l_DTA'])
        metrics[key]['TB']    = batchwise_loss_accumulator(
                        output_transform=lambda x: x['l_DTB'])
        metrics[key]['T']    = batchwise_loss_accumulator(
                        output_transform=lambda x: x['l_DT'])
        metrics[key]['S']    = batchwise_loss_accumulator(
                        output_transform=lambda x: x['l_DS'])

        for name, m in metrics[key].items():
            m.attach(engines[key], name=name)

    # Set up validation.
    engines['train'].add_event_handler(Events.EPOCH_COMPLETED,
                             lambda _: engines['valid'].run(loader['valid']))
    
    # Set up model checkpointing.
    score_function = scoring_function('dice_TA')
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
        metric_keys=['dice_TA']+['dice_SA']+['dice_global'])
    
    tracker.attach(
        engine=engines["train"],
        prefix="train",
        output_transform=lambda x: dict([(k, v)
                                         for k, v in x.items()
                                         if k.startswith('l_')]),
        metric_keys=['dice_TA']+['dice_SA'])
    
    # Set up image logging.
    def output_transform(output):
        transformed = OrderedDict()
        for k, v in output.items():
            if k.startswith('x_') and v is not None and v.dim()==4:
                k_new = k.replace('x_','')
                v_new = v.cpu().numpy()
                if k_new.startswith('SM') or k_new.startswith('TM') or k_new.startswith('SATM') or k_new.startswith('TAM') or k_new.startswith('SAM') or k_new.startswith('TASM'):
                    if v_new.shape[1]==1:
                        # 'M', or 'AM' with single class.
                        v_new = np.squeeze(v_new, axis=1)
                    else:
                        # 'AM' with multiple classes.
                        v_new = np.argmax(v_new, axis=1)
                        v_new = sum([(v_new==i+1)*c
                                     for i, c in enumerate(target_class)])
                    if output['x_'+k_new] is not None and                                                 output['x_'+k_new].shape[1]!=1:
                        # HACK (min and max values for correct vis)
                        v_new[:,0,0]  = max(target_class)
                        v_new[:,0,-1] = 0
                else:
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
    
    '''
    Test.
    '''
    print("\nTESTING\n")
    engines['test'].run(loader['test'])
    print("\nTESTING ON BEST CHECKPOINT\n")
    experiment_state.load_best_state()
    engines['test'].run(loader['test'])


if __name__ == '__main__':
    parser = get_parser()
    dispatch(parser, run)

