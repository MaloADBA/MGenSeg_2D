from collections import (defaultdict,
                         OrderedDict)
from contextlib import nullcontext
import functools
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.cuda.amp import autocast
import torch.nn.functional as F
from fcn_maker.loss import dice_loss
from .common.network.basic import grad_norm
from .common.losses import (bce,
                            cce,
                            dist_ratio_mse_abs,
                            gan_objective,
                            mae,
                            mse)
from .common.mine import mine

def clear_grad(optimizer):
    # Sets `grad` to None instead of zeroing it.
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad = None


def _reduce(loss):
    # Reduce torch tensors with dim > 1 with a mean on all but the first
    # (batch) dimension. Else, return as is.
    def _mean(x):
        if not isinstance(x, torch.Tensor) or x.dim()<=1:
            return x
        else:
            return x.view(x.size(0), -1).mean(1)
    if not hasattr(loss, '__len__'): loss = [loss]
    if len(loss)==0: return 0
    return sum([_mean(v) for v in loss])


def autocast_if_needed():
    # Decorator. If the method's object has a scaler, use the pytorch
    # autocast context; else, run the method without any context.
    def decorator(method):
        @functools.wraps(method)
        def context_wrapper(cls, *args, **kwargs):
            return method(cls, *args, **kwargs)
        return context_wrapper
    return decorator

class segmentation_model(nn.Module):
    """
    Interface wrapper around the `DataParallel` parts of the model.
    """
    def __init__(self, segmenter, loss_seg=None, rng=None):
        super(segmentation_model, self).__init__()

        kwargs = OrderedDict((
            ('rng',               rng if rng else np.random.RandomState()),
            ('segmenter',    segmenter),
            ('loss_seg',          loss_seg)
        ))

        for key, val in kwargs.items():
            setattr(self, key, val)
        
        # Module to compute all network outputs (except discriminator) on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_forward = ['segmenter', 'rng']
        kwargs_forward = dict([(key, val) for key, val in kwargs.items()
                               if key in keys_forward])
        self._forward = _forward(**kwargs_forward)
    
    def _autocast_if_needed(self):
        return nullcontext()
    
    def forward(self, x_V, x_M, optimizer=None, rng=None):
        # Compute gradients and update?
        do_updates_bool = True if optimizer is not None else False
        
        # Apply scaler for gradient backprop if it is passed.
        def backward(loss):
            return loss.backward()
        
        # Apply scaler for optimizer step if it is passed.
        def step(optimizer):
            optimizer.step()
        
        # Compute all outputs.
        with torch.set_grad_enabled(do_updates_bool):
            with self._autocast_if_needed():
                visible= self._forward(x_V=x_V, rng=rng)

        # Prepare a mask Tensor without None entries.
        mask_VM_packed = x_AM_packed = None
        mask_indices = []
        if x_M is not None:
            mask_indices = [i for i, m in enumerate(x_M) if m is not None]
            mask_packed = np.array([x_M[i] for i in mask_indices], dtype=np.float32)
            mask_packed = Variable(torch.from_numpy(mask_packed))
            mask_VM_packed = mask_packed.cuda()

        # Segment.
        x_AM_packed = visible['x_VM'][mask_indices]
        loss_seg = self.loss_seg(x_AM_packed, mask_VM_packed)

        mask_AM_packed = ((visible['x_VM'][mask_indices] > 0.5)*1)                        
        
        if do_updates_bool and isinstance(loss_seg, torch.Tensor):
            clear_grad(optimizer['G'])
            with self._autocast_if_needed():
                _loss = loss_seg.mean()
            backward(_loss)

            step(optimizer['G'])
            gradnorm_G = grad_norm(self)
        
        # Compile ouputs.
        outputs = OrderedDict()
        outputs['x_M'] = mask_VM_packed
        outputs['x_A'] = x_V[mask_indices]
        outputs['x_AM'] = mask_AM_packed
        return outputs


class _forward(nn.Module):
    def __init__(self, segmenter, rng=None):
        super(_forward, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.segmenter          = segmenter

   
    @autocast_if_needed()
    def forward(self, x_V, rng=None):

        x_VM = self.segmenter(x_V)

        # Compile outputs and return.
        visible = {'x_VM': x_VM}
   
        return visible



