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
import math

def label2onenot(labels, dim):
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

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
            if cls.scaler is not None:
                with torch.cuda.amp.autocast():
                    return method(cls, *args, **kwargs)
            return method(cls, *args, **kwargs)
        return context_wrapper
    return decorator

class segmentation_model(nn.Module):
    """
    Interface wrapper around the `DataParallel` parts of the model.
    """
    def __init__(self, generator, disc, classifieur, loss_seg=None,
                 num_disc_updates=1, lambda_seg=1, lambda_disc=1, lambda_id=1, lambda_class=1, lambda_gp=1,
                 scaler=None, debug_ac_gan=False, rng=None, relativistic=False, grad_penalty=None):
        super(segmentation_model, self).__init__()

        self.lambda_class = lambda_class
        self.lambda_id = lambda_id
        self.lambda_gp = lambda_gp
        self.lambda_seg = lambda_seg
        self.lambda_disc = lambda_disc
        
        self.gan_objective = gan_objective('hinge',
                                           relativistic=relativistic,
                                           grad_penalty_real=grad_penalty,
                                           grad_penalty_fake=None,
                                           grad_penalty_mean=0)
        self.generator = generator
        self.scaler = scaler
        self.rng = rng
        self.debug_ac_gan = debug_ac_gan
        self.loss_seg = loss_seg
        self.separate_networks = OrderedDict((
            ('disc',            disc),
            ('classifieur',            classifieur)
            ))

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).cuda()
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)
    
    def _autocast_if_needed(self):
        # If a scaler is passed, use pytorch gradient autocasting. Else,
        # just use a null context that does nothing.
        if self.scaler is not None:
            context = torch.cuda.amp.autocast()
        else:
            context = nullcontext()
        return context
    
    def forward(self, x_S, x_T, mask_S=None, mask_T=None, optimizer=None, rng=None):
        # Compute gradients and update?
        do_updates_bool = True if optimizer is not None else False
        
        # Apply scaler for gradient backprop if it is passed.
        def backward(loss):
            if self.scaler is not None:
                return self.scaler.scale(loss).backward()
            return loss.backward()
        
        # Apply scaler for optimizer step if it is passed.
        def step(optimizer):
            if self.scaler is not None:
                self.scaler.step(optimizer)
            else:
                optimizer.step()
                
        # Compute labels.
        labels_org = torch.tensor(np.array([0 for i in range(x_S.shape[0])]+[1 for i in range(x_S.shape[0])]))
        rand_idx = torch.randperm(labels_org.size(0))
        labels_trg = labels_org[rand_idx]

        vec_trg = label2onenot(labels_trg, 2)
        vec_org = label2onenot(labels_org, 2)

        labels_org = labels_org.cuda()
        labels_trg = labels_trg.cuda()
        vec_org = vec_org.cuda()
        vec_trg = vec_trg.cuda()
        
        x_real = torch.cat([x_S, x_T], dim=0)
        
        disc = self.separate_networks['disc']
        classifieur = self.separate_networks['classifieur']

        ## compute outputs
        
        # original-to-target domain.
        y_seg, x_fake = self.generator(x_real, vec_trg)
       
        # Target-to-original domain.
        y_rec, x_rec = self.generator(x_fake, vec_org)

        
                
        ## train discriminator

        out_cls = classifieur(disc.main(x_real))

        d_loss_cls = F.cross_entropy(out_cls, labels_org)

        disc_loss =  self.gan_objective.D(disc,
                                     fake=x_fake.detach(),
                                     real=x_real,
                                     kwargs_real=None,
                                     kwargs_fake=None,
                                     scaler=None)
        
        d_loss_gp = 0.

        # Backward and optimize.
        d_loss =  self.lambda_disc*_reduce(disc_loss) + self.lambda_class * d_loss_cls + self.lambda_gp * d_loss_gp
        
        if do_updates_bool:
            clear_grad(optimizer['D'])
            backward(d_loss)
            step(optimizer['D'])

              
        
        ## train generator

        out_cls = classifieur(disc.main(x_fake))
        g_loss_cls = F.cross_entropy(out_cls, labels_trg) 

        gen_loss =  self.gan_objective.G(disc,
                                     fake=x_fake,
                                     real=x_real,
                                     kwargs_real=None,
                                     kwargs_fake=None)
        
        mask_S_packed = mask_T_packed = None

        mask_S_packed = Variable(torch.from_numpy((np.array(mask_S, dtype=np.float32)>0)*1))
        if torch.cuda.device_count()==1:
            # `DataParallel` does not respect `output_device` when
            # there is only one GPU. So it returns outputs on GPU rather
            # than CPU, as requested. When this happens, putting mask
            # on GPU allows all values to stay on one device.
            mask_S_packed = mask_S_packed.cuda()
            
        mask_T_packed = Variable(torch.from_numpy((np.array(mask_T, dtype=np.float32)>0)*1))
        if torch.cuda.device_count()==1:
            # `DataParallel` does not respect `output_device` when
            # there is only one GPU. So it returns outputs on GPU rather
            # than CPU, as requested. When this happens, putting mask
            # on GPU allows all values to stay on one device.
            mask_T_packed = mask_T_packed.cuda()
        
        mask_packed = torch.cat([mask_S_packed, mask_T_packed], dim=0)
        
        loss_seg = 0.
        loss_seg = self.loss_seg(y_seg, mask_packed)
   
        g_loss_rec = torch.mean(torch.abs(x_real - x_rec))
        g_loss_shape = self.loss_seg(y_rec, mask_packed)

        # Backward and optimize.
        g_loss = self.lambda_disc*_reduce(gen_loss) + self.lambda_class * g_loss_cls + self.lambda_seg * loss_seg + self.lambda_id * g_loss_rec + self.lambda_seg * g_loss_shape
        

        if do_updates_bool :
            if 'S' in optimizer:
                clear_grad(optimizer['S'])
            clear_grad(optimizer['G'])
            backward(g_loss)
            if self.scaler is not None:
                self.scaler.unscale_(optimizer['G'])
                if 'S' in optimizer:
                    self.scaler.unscale_(optimizer['S'])
            step(optimizer['G'])
            if 'S' in optimizer:
                step(optimizer['S'])
        
        
        # Update scaler.
        if self.scaler is not None and do_updates_bool:
            self.scaler.update()
        
        binary_M = ((y_seg > 0.5)*1)
        binary_M_bis = ((y_rec > 0.5)*1)
        
        # Compile ouputs.
        outputs = OrderedDict()
        outputs['x_SM'] = mask_S_packed
        outputs['x_S'] = x_S
        outputs['x_SM_pred'] = binary_M[:x_S.shape[0]]        
        outputs['x_S_trans'] = x_fake[:x_S.shape[0]]
        outputs['x_SM_pred_bis'] = binary_M_bis[:x_S.shape[0]]           
        outputs['x_S_rec'] = x_rec[:x_S.shape[0]]
        outputs['x_TM'] = mask_T_packed
        outputs['x_T'] = x_T
        outputs['x_TM_pred'] = binary_M[x_S.shape[0]:]
        outputs['x_T_trans'] = x_fake[x_S.shape[0]:]
        outputs['x_TM_pred_bis'] = binary_M_bis[x_S.shape[0]:]
        outputs['x_T_rec'] = x_rec[x_S.shape[0]:]
        outputs['l_D']  = d_loss
        outputs['l_G']  = g_loss
        outputs['l_seg']  = loss_seg
        outputs['l_shape']  = g_loss_shape
        outputs['l_class']  = g_loss_cls
        outputs['l_rec']  = g_loss_rec
        return outputs
        
        
