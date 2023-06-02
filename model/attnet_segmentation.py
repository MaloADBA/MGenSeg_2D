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

def prob2entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    公式（2）
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

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
    def __init__(self, segmentor, disc_main, disc_aux, loss_seg=None, loss_gan='hinge',
                 num_disc_updates=1, lambda_disc_main=1, lambda_disc_aux=1, lambda_seg_main=1, lambda_seg_aux=1,
                 scaler=None, debug_ac_gan=False, rng=None, relativistic=False, grad_penalty=None):
        super(segmentation_model, self).__init__()
        lambdas = OrderedDict((
            ('lambda_disc_main',        lambda_disc_main),
            ('lambda_disc_aux',        lambda_disc_aux),
            ('lambda_seg_main',        lambda_seg_main),
            ('lambda_seg_aux',        lambda_seg_aux)
            ))
        kwargs = OrderedDict((
            ('rng',               rng if rng else np.random.RandomState()),
            ('scaler',            scaler),
            ('segmentor',    segmentor),
            ('loss_seg',          loss_seg),
            ('loss_gan',          loss_gan),
            ('num_disc_updates',  num_disc_updates),
            ('relativistic',      relativistic),
            ('grad_penalty',      grad_penalty),
            ('gan_objective',     gan_objective(loss_gan,
                                                relativistic=relativistic,
                                                grad_penalty_real=grad_penalty,
                                                grad_penalty_fake=None,
                                                grad_penalty_mean=0)),
            ('debug_ac_gan',      debug_ac_gan)
            ))
        self.separate_networks = OrderedDict((
            ('disc_main',             disc_main),
            ('disc_aux',             disc_aux)
            ))
        kwargs.update(lambdas)
        for key, val in kwargs.items():
            setattr(self, key, val)

        # Separate networks not stored directly as attributes.
        # -> Separate parameters, separate optimizers.
        kwargs.update(self.separate_networks)
        
        # Module to compute all network outputs (except discriminator) on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_forward = ['segmentor', 'scaler', 'rng']
        kwargs_forward = dict([(key, val) for key, val in kwargs.items()
                               if key in keys_forward])
        self._forward = _forward(**kwargs_forward, **lambdas)
        if torch.cuda.device_count()>1:
            self._forward = nn.DataParallel(self._forward, output_device=-1)
        
        # Module to compute discriminator losses on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_D = ['gan_objective', 'disc_main', 'disc_aux', 'scaler', 'debug_ac_gan']
        kwargs_D = dict([(key, val) for key, val in kwargs.items()
                         if key in keys_D])
        self._loss_D = _loss_D(**kwargs_D, **lambdas)
        if torch.cuda.device_count()>1:
            self._loss_D = nn.DataParallel(self._loss_D, output_device=-1)
        
        # Module to compute generator updates on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_G = ['gan_objective', 'disc_main', 'disc_aux', 'scaler', 'debug_ac_gan']
        kwargs_G = dict([(key, val) for key, val in kwargs.items()
                         if key in keys_G])
        self._loss_G = _loss_G(**kwargs_G, **lambdas)
        if torch.cuda.device_count()>1:
            self._loss_G = nn.DataParallel(self._loss_G, output_device=-1)
    
    def _autocast_if_needed(self):
        # If a scaler is passed, use pytorch gradient autocasting. Else,
        # just use a null context that does nothing.
        if self.scaler is not None:
            context = torch.cuda.amp.autocast()
        else:
            context = nullcontext()
        return context
    
    def forward(self, x_S, x_T, x_TS, mask_S=None, mask_T=None, optimizer=None, rng=None):
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
        
        # Compute all outputs.
        with torch.set_grad_enabled(do_updates_bool):
            with self._autocast_if_needed():
                visible, hidden = self._forward(x_S=x_S, x_T=x_T, x_TS=x_TS, rng=rng)
        #####
        # Evaluate discriminator loss and update.
        loss_disc = defaultdict(int)
        loss_D = gradnorm_D = 0
        
        for i in range(self.num_disc_updates):
            # Evaluate.
            with torch.set_grad_enabled(do_updates_bool):
                with self._autocast_if_needed():
                    loss_disc = self._loss_D(x_SM_main=visible['x_SM_main'], x_TSM_main=visible['x_TSM_main'],
                                             x_SM_aux=hidden['x_SM_aux'], x_TSM_aux=hidden['x_TSM_aux'])
                    loss_D = _reduce(loss_disc.values())
            # Update discriminator
            disc_main = self.separate_networks['disc_main']
            disc_aux = self.separate_networks['disc_aux']
            if do_updates_bool:
                clear_grad(optimizer['D'])
                with self._autocast_if_needed():
                    _loss = loss_D.mean()
                backward(_loss)
                step(optimizer['D'])
                gradnorm_D = grad_norm(disc_main)+grad_norm(disc_aux)

        
        # Evaluate generator losses.
        gradnorm_G = 0
        with torch.set_grad_enabled(do_updates_bool):
            with self._autocast_if_needed():
                losses_G = self._loss_G(x_SM_main=visible['x_SM_main'], x_TSM_main=visible['x_TSM_main'],
                                            x_SM_aux=hidden['x_SM_aux'], x_TSM_aux=hidden['x_TSM_aux'])
  
        mask_S_packed = mask_T_packed = x_SM_main_packed = x_TSM_main_packed = x_SM_aux_packed = x_TSM_aux_packed = None

        if mask_S is not None:
            # Prepare a mask Tensor without None entries.
            mask_S_indices = [i for i, m in enumerate(mask_S) if m is not None]
            mask_S_packed = np.array([(mask_S[i]>0)*1 for i in mask_S_indices], dtype=np.float32)
            mask_S_packed = Variable(torch.from_numpy(mask_S_packed))
            if torch.cuda.device_count()==1:
                # `DataParallel` does not respect `output_device` when
                # there is only one GPU. So it returns outputs on GPU rather
                # than CPU, as requested. When this happens, putting mask
                # on GPU allows all values to stay on one device.
                mask_S_packed = mask_S_packed.cuda()
                
        if mask_T is not None:
            # Prepare a mask Tensor without None entries.
            mask_T_indices = [i for i, m in enumerate(mask_T) if m is not None]
            mask_T_packed = np.array([(mask_T[i]>0)*1 for i in mask_T_indices], dtype=np.float32)
            mask_T_packed = Variable(torch.from_numpy(mask_T_packed))
            if torch.cuda.device_count()==1:
                # `DataParallel` does not respect `output_device` when
                # there is only one GPU. So it returns outputs on GPU rather
                # than CPU, as requested. When this happens, putting mask
                # on GPU allows all values to stay on one device.
                mask_T_packed = mask_T_packed.cuda()    
                
        loss_seg_SM_main = None
        loss_seg_SM_aux = None
                
        if mask_T_packed is not None and len(mask_T_packed):
                with self._autocast_if_needed():
                    
                    x_TSM_main_packed = visible['x_TSM_main'][mask_T_indices]
                    x_SM_main_packed = visible['x_SM_main'][mask_S_indices]
                    
                    x_TSM_aux_packed = hidden['x_TSM_aux'][mask_T_indices]
                    x_SM_aux_packed = hidden['x_SM_aux'][mask_S_indices]
                    
                    loss_seg_SM_main = self.lambda_seg_main*self.loss_seg(torch.cat((x_SM_main_packed,x_TSM_main_packed), dim=0), torch.cat((mask_S_packed,mask_T_packed), dim=0)).mean()
                        
                    loss_seg_SM_aux = self.lambda_seg_aux*self.loss_seg(torch.cat((x_SM_aux_packed,x_TSM_aux_packed), dim=0), torch.cat((mask_S_packed,mask_T_packed), dim=0)).mean()
                                        
                    x_TSM_main_packed = ((visible['x_TSM_main'][mask_T_indices] > 0.5)*1)
                    x_SM_main_packed = ((visible['x_SM_main'][mask_S_indices] > 0.5)*1)
                    

        
        if not (mask_T_packed is not None and len(mask_T_packed)) and len(mask_S_packed):
            with self._autocast_if_needed():
                
                    x_SM_main_packed = visible['x_SM_main'][mask_S_indices]
                    
                    x_SM_aux_packed = hidden['x_SM_aux'][mask_S_indices]
                    
                    loss_seg_SM_main = self.lambda_seg_main*self.loss_seg(x_SM_main_packed, mask_S_packed)
                    loss_seg_SM_aux = self.lambda_seg_aux*self.loss_seg(x_SM_aux_packed, mask_S_packed)
                                        
                    x_SM_main_packed = ((visible['x_SM_main'][mask_S_indices] > 0.5)*1)
    

        # Include segmentation loss with generator losses and update.
        with self._autocast_if_needed():
            if loss_seg_SM_main is not None:
                losses_G['l_seg'] = _reduce([loss_seg_SM_main, loss_seg_SM_aux]).mean()
                losses_G['l_G'] += losses_G['l_seg']
            else :
                losses_G['l_seg'] = None
            loss_G = losses_G['l_G']
        if do_updates_bool and isinstance(loss_G, torch.Tensor):
            if 'S' in optimizer:
                clear_grad(optimizer['S'])
            clear_grad(optimizer['G'])
            with self._autocast_if_needed():
                _loss = loss_G.mean()
            backward(_loss)
            if self.scaler is not None:
                self.scaler.unscale_(optimizer['G'])
                if 'S' in optimizer:
                    self.scaler.unscale_(optimizer['S'])

            step(optimizer['G'])
            if 'S' in optimizer:
                step(optimizer['S'])
            gradnorm_G = grad_norm(self)
        
        # Unscale norm.
        if self.scaler is not None and do_updates_bool:
            gradnorm_D /= self.scaler.get_scale()
            gradnorm_G /= self.scaler.get_scale()
        
        # Update scaler.
        if self.scaler is not None and do_updates_bool:
            self.scaler.update()
            
        S_pred_main = prob2entropy(torch.cat([visible['x_SM_main'],1-visible['x_SM_main']],dim=1))
        TS_pred_main = prob2entropy(torch.cat([visible['x_TSM_main'],1-visible['x_TSM_main']],dim=1))
        
        entropy_SM_main = torch.sum(S_pred_main, dim=1, keepdim=True)
        entropy_TSM_main = torch.sum(TS_pred_main, dim=1, keepdim=True)
        
        # Compile ouputs.
        outputs = OrderedDict()
        outputs['x_SM'] = mask_S_packed
        outputs['x_S'] = visible['x_S']
        outputs['x_SM_entropy'] = entropy_SM_main
        outputs['x_SM_main'] = x_SM_main_packed
        outputs['x_TM'] = mask_T_packed
        outputs['x_T'] = visible['x_T']
        outputs['x_TS'] = visible['x_TS']
        outputs['x_TSM_entropy'] = entropy_TSM_main
        outputs['x_TSM_main'] = x_TSM_main_packed
        outputs.update(losses_G)
        outputs['l_D']  = loss_D
        outputs['l_D_main'] = _reduce([loss_disc['main']])
        outputs['l_D_aux'] = _reduce([loss_disc['aux']])
        outputs['l_gradnorm_D'] = gradnorm_D
        outputs['l_gradnorm_G'] = gradnorm_G
        return outputs


class _forward(nn.Module):
    def __init__(self, segmentor, lambda_seg_main=1, lambda_seg_aux=1, lambda_disc_main=1, lambda_disc_aux=1, scaler=None, rng=None):
        super(_forward, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.segmentor          = segmentor
        self.scaler             = scaler

   
    @autocast_if_needed()
    def forward(self, x_S, x_T, x_TS, rng=None):

        x_SM_aux, x_SM_main = self.segmentor(x_S)
        x_TSM_aux, x_TSM_main = self.segmentor(x_TS)

        # Compile outputs and return.
        visible = OrderedDict((
            ('x_S',           x_S),
            ('x_SM_main',           x_SM_main),
            ('x_T',          x_T),
            ('x_TS',          x_TS),
            ('x_TSM_main',           x_TSM_main)
            ))
        hidden = OrderedDict((
            ('x_SM_aux',           x_SM_aux),
            ('x_TSM_aux',           x_TSM_aux)
            ))
   
        return visible, hidden


class _loss_D(nn.Module):
    def __init__(self, gan_objective, disc_main, disc_aux, lambda_seg_main=1, lambda_seg_aux=1, 
                 scaler=None, lambda_disc_main=1, lambda_disc_aux=1, debug_ac_gan=False):
        super(_loss_D, self).__init__()
        self._gan               = gan_objective
        self.scaler             = scaler
        self.lambda_disc_main        = lambda_disc_main
        self.lambda_disc_aux        = lambda_disc_aux
        self.debug_ac_gan       = debug_ac_gan
        self.net = {'disc_main'    : disc_main,
                    'disc_aux'    : disc_aux}  # Separate params.
    
    @autocast_if_needed()
    def forward(self, x_SM_main, x_TSM_main, x_SM_aux, x_TSM_aux):

        if isinstance(x_SM_main, list):
            x_SM_main = [x.detach() for x in x_SM_main]
        else:
            x_SM_main = x_SM_main.detach()
            
        if isinstance(x_SM_aux, list):
            x_SM_aux = [x.detach() for x in x_SM_aux]
        else:
            x_SM_aux = x_SM_aux.detach()
            
        if isinstance(x_TSM_main, list):
            x_TSM_main = [x.detach() for x in x_TSM_main]
        else:
            x_TSM_main = x_TSM_main.detach()
            
        if isinstance(x_TSM_aux, list):
            x_TSM_aux = [x.detach() for x in x_TSM_aux]
        else:
            x_TSM_aux = x_TSM_aux.detach()
            
        # Discriminators.
        kwargs_real = None 
        kwargs_fake = None 
        loss_disc = OrderedDict()
        
        S_pred_main = prob2entropy(torch.cat([x_SM_main,1-x_SM_main],dim=1))
        TS_pred_main = prob2entropy(torch.cat([x_TSM_main,1-x_TSM_main],dim=1))
        
        loss_disc['main'] = 0.5*self._gan.D(self.net['disc_main'],
                         fake=TS_pred_main,
                         real=S_pred_main,
                         kwargs_real=kwargs_real,
                         kwargs_fake=kwargs_fake,
                         scaler=self.scaler)       
        
            
        S_pred_aux = torch.cat([x_SM_aux,1-x_SM_aux],dim=1)
        TS_pred_aux = torch.cat([x_TSM_aux,1-x_TSM_aux],dim=1)
        
        loss_disc['aux'] = 0.5*self._gan.D(self.net['disc_aux'],
                                 fake=TS_pred_aux,
                                 real=S_pred_aux,
                                 kwargs_real=kwargs_real,
                                 kwargs_fake=kwargs_fake,
                                 scaler=self.scaler)

        return loss_disc


class _loss_G(nn.Module):
    def __init__(self, gan_objective, disc_main, disc_aux, lambda_seg_main=1, lambda_seg_aux=1,
                 scaler=None, lambda_disc_main=1, lambda_disc_aux=1, debug_ac_gan=False):
        super(_loss_G, self).__init__()
        self._gan               = gan_objective
        self.scaler             = scaler
        self.lambda_disc_main        = lambda_disc_main
        self.lambda_disc_aux        = lambda_disc_aux
        self.debug_ac_gan       = debug_ac_gan
        self.net = {'disc_main'    : disc_main,
                    'disc_aux'    : disc_aux}  # Separate params.
    
    @autocast_if_needed()
    def forward(self, x_SM_main, x_TSM_main, x_SM_aux, x_TSM_aux):
                                     

        # Generator loss.
        loss_gen = defaultdict(int)
        kwargs_real = None
        kwargs_fake = None  
        
        S_pred_main = prob2entropy(torch.cat([x_SM_main,1-x_SM_main],dim=1))
        TS_pred_main = prob2entropy(torch.cat([x_TSM_main,1-x_TSM_main],dim=1)) 
        
        S_pred_aux = torch.cat([x_SM_aux,1-x_SM_aux],dim=1)
        TS_pred_aux = torch.cat([x_TSM_aux,1-x_TSM_aux],dim=1)
        
        loss_gen['main'] = self.lambda_disc_main*self._gan.G(self.net['disc_main'],
                     fake=TS_pred_main,
                     real=S_pred_main,
                     kwargs_real=kwargs_real,
                     kwargs_fake=kwargs_fake)
        loss_gen['aux'] = self.lambda_disc_aux*self._gan.G(self.net['disc_aux'],
                     fake=TS_pred_aux,
                     real=S_pred_aux,
                     kwargs_real=kwargs_real,
                     kwargs_fake=kwargs_fake)

        
            
        # All generator losses combined.
        loss_G = _reduce(loss_gen.values())
        
        # Compile outputs and return.
        losses = OrderedDict((
            ('l_G',           loss_G),
            ('l_gen_main',      _reduce([loss_gen['main']])),
            ('l_gen_aux',      _reduce([loss_gen['aux']]))
            ))
        return losses

