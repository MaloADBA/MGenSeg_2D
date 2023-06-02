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
            if cls.scaler is not None:
                with torch.cuda.amp.autocast():
                    return method(cls, *args, **kwargs)
            return method(cls, *args, **kwargs)
        return context_wrapper
    return decorator

class translation_model(nn.Module):
    """
    Interface wrapper around the `DataParallel` parts of the model.
    """
    def __init__(self, generator_S, generator_T, disc_S, disc_T, loss_rec=mae, loss_gan='hinge',
                 num_disc_updates=1, lambda_disc=1, lambda_cyc=1, scaler=None, 
                 debug_ac_gan=False, rng=None, relativistic=False, grad_penalty=None):
        super(translation_model, self).__init__()
        lambdas = OrderedDict((
            ('lambda_disc',        lambda_disc),
            ('lambda_cyc',        lambda_cyc)
            ))
        kwargs = OrderedDict((
            ('rng',               rng if rng else np.random.RandomState()),
            ('scaler',            scaler),
            ('generator_S',    generator_S),
            ('generator_T',  generator_T),
            ('loss_rec',          loss_rec),
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
            ('disc_S',             disc_S),
            ('disc_T',             disc_T)
            ))
        kwargs.update(lambdas)
        for key, val in kwargs.items():
            setattr(self, key, val)

        # Separate networks not stored directly as attributes.
        # -> Separate parameters, separate optimizers.
        kwargs.update(self.separate_networks)
        
        # Module to compute all network outputs (except discriminator) on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_forward = ['generator_S', 'generator_T', 'scaler', 'rng']
        kwargs_forward = dict([(key, val) for key, val in kwargs.items()
                               if key in keys_forward])
        self._forward = _forward(**kwargs_forward, **lambdas)
        if torch.cuda.device_count()>1:
            self._forward = nn.DataParallel(self._forward, output_device=-1)
        
        # Module to compute discriminator losses on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_D = ['gan_objective', 'disc_S', 'disc_T', 'scaler', 'debug_ac_gan']
        kwargs_D = dict([(key, val) for key, val in kwargs.items()
                         if key in keys_D])
        self._loss_D = _loss_D(**kwargs_D, **lambdas)
        if torch.cuda.device_count()>1:
            self._loss_D = nn.DataParallel(self._loss_D, output_device=-1)
        
        # Module to compute generator updates on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_G = ['gan_objective', 'disc_S', 'disc_T', 'scaler', 'loss_rec', 'debug_ac_gan']
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
    
    def forward(self, x_S, x_T, optimizer=None, rng=None):
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
                visible = self._forward(x_S=x_S, x_T=x_T, rng=rng)
        #####
        # Evaluate discriminator loss and update.
        loss_disc = defaultdict(int)
        loss_D = gradnorm_D = 0
        if self.lambda_disc:
            
            for i in range(self.num_disc_updates):
                # Evaluate.
                with torch.set_grad_enabled(do_updates_bool):
                    with self._autocast_if_needed():
                        loss_disc = self._loss_D(
                            x_S=x_S, x_T=x_T,
                            out_ST=visible['x_ST'], out_TS=visible['x_TS'])
                        loss_D = _reduce(loss_disc.values())
                # Update discriminator
                disc_S = self.separate_networks['disc_S']
                disc_T = self.separate_networks['disc_T']
                if do_updates_bool:
                    clear_grad(optimizer['D'])
                    with self._autocast_if_needed():
                        _loss = loss_D.mean()
                    backward(_loss)
                    step(optimizer['D'])
                    gradnorm_D_trans = grad_norm(disc_T)+grad_norm(disc_S)

        
        # Evaluate generator losses.
        gradnorm_G = 0
        with torch.set_grad_enabled(do_updates_bool):
            with self._autocast_if_needed():
                losses_G = self._loss_G(x_S=x_S, x_T=x_T,
                                        x_ST=visible['x_ST'], x_TS=visible['x_TS'],
                                        x_STS=visible['x_STS'], x_TST=visible['x_TST'])

        
        # Include segmentation loss with generator losses and update.
        with self._autocast_if_needed():
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
        
        # Compile ouputs.
        outputs = OrderedDict()
        outputs.update(visible)
        outputs.update(losses_G)
        outputs['l_D']  = loss_D
        outputs['l_DS'] = _reduce([loss_disc['S']])
        outputs['l_DT'] = _reduce([loss_disc['T']])
        outputs['l_gradnorm_D'] = gradnorm_D
        outputs['l_gradnorm_G'] = gradnorm_G
        return outputs


class _forward(nn.Module):
    def __init__(self, generator_S, generator_T, scaler=None,
                 lambda_disc=1, lambda_cyc=1, rng=None):
        super(_forward, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.generator_S        = generator_S
        self.generator_T        = generator_T
        self.scaler             = scaler
        self.lambda_disc        = lambda_disc
        self.lambda_cyc         = lambda_cyc
   
    @autocast_if_needed()
    def forward(self, x_S, x_T, rng=None):

        # T pathway (target)
        x_TS = self.generator_S(x_T)
        x_TST = self.generator_T(x_TS)
        
        # S pathway (source)
        x_ST = self.generator_T(x_S)
        x_STS = self.generator_S(x_ST)
        
        # Compile outputs and return.
        visible = OrderedDict((
            
            
            ('x_S',           x_S), 
            ('x_ST',          x_ST),
            ('x_STS',          x_STS),
            ('x_T',           x_T),
            ('x_TS',          x_TS),
            ('x_TST',          x_TST)
            ))
   
        return visible


class _loss_D(nn.Module):
    def __init__(self, gan_objective, disc_T, disc_S, 
                 scaler=None, lambda_disc=1, lambda_cyc=1, debug_ac_gan=False):
        super(_loss_D, self).__init__()
        self._gan               = gan_objective
        self.scaler             = scaler
        self.lambda_disc        = lambda_disc
        self.lambda_cyc        = lambda_cyc
        self.debug_ac_gan       = debug_ac_gan
        self.net = {'disc_T'    : disc_T,
                    'disc_S'    : disc_S}  # Separate params.
    
    @autocast_if_needed()
    def forward(self, x_S, x_T,
                out_ST, out_TS):

        
        if isinstance(out_ST, list):
            out_ST = [x.detach() for x in out_ST]
        else:
            out_ST = out_ST.detach()
            
        if isinstance(out_TS, list):
            out_TS = [x.detach() for x in out_TS]
        else:
            out_TS = out_TS.detach()
        
        # Discriminators.
        kwargs_real = None 
        kwargs_fake = None 
        loss_disc = OrderedDict()
        if self.lambda_disc :
            loss_disc['S'] = self.lambda_disc*self._gan.D(self.net['disc_S'],
                                     fake=out_TS,
                                     real=x_S,
                                     kwargs_real=kwargs_real,
                                     kwargs_fake=kwargs_fake,
                                     scaler=self.scaler)
            
            loss_disc['T'] = self.lambda_disc*self._gan.D(self.net['disc_T'],
                                     fake=out_ST,
                                     real=x_T,
                                     kwargs_real=kwargs_real,
                                     kwargs_fake=kwargs_fake,
                                     scaler=self.scaler)

        return loss_disc


class _loss_G(nn.Module):
    def __init__(self, gan_objective, disc_T, disc_S, scaler=None,
                 loss_rec=mae, lambda_disc=1, lambda_cyc=1, debug_ac_gan=False):
        super(_loss_G, self).__init__()
        self._gan               = gan_objective
        self.scaler             = scaler
        self.loss_rec           = loss_rec
        self.lambda_disc        = lambda_disc
        self.lambda_cyc         = lambda_cyc
        self.debug_ac_gan       = debug_ac_gan
        self.net = {'disc_T'    : disc_T,
                    'disc_S'    : disc_S}  # Separate params.
    
    @autocast_if_needed()
    def forward(self, x_S, x_T,
                x_ST, x_STS, x_TS, x_TST):
                                     

        # Generator loss.
        loss_gen = defaultdict(int)
        kwargs_real = None
        kwargs_fake = None  
              
        if self.lambda_disc:
            loss_gen['S'] = self.lambda_disc*self._gan.G(self.net['disc_S'],
                         fake=x_TS,
                         real=x_S,
                         kwargs_real=kwargs_real,
                         kwargs_fake=kwargs_fake)
            loss_gen['T'] = self.lambda_disc*self._gan.G(self.net['disc_T'],
                         fake=x_ST,
                         real=x_T,
                         kwargs_real=kwargs_real,
                         kwargs_fake=kwargs_fake)

        
        # Reconstruction loss.
        loss_rec = defaultdict(int) 
            
        if self.lambda_cyc:
            loss_rec['x_STS'] = self.lambda_cyc*self.loss_rec(x_S, x_STS)
            loss_rec['x_TST'] = self.lambda_cyc*self.loss_rec(x_T, x_TST)
            
        # All generator losses combined.
        loss_G = ( _reduce(loss_gen.values())
                  +_reduce(loss_rec.values()))
        
        # Compile outputs and return.
        losses = OrderedDict((
            ('l_G',           loss_G),
            ('l_gen_S',      _reduce([loss_gen['S']])),
            ('l_gen_T',      _reduce([loss_gen['T']])),
            ('l_rec_S',         _reduce([loss_rec['x_STS']])),
            ('l_rec_T',         _reduce([loss_rec['x_TST']])),
            ('l_rec',      _reduce([loss_rec['x_TST'], loss_rec['x_STS']]))
            ))
        return losses
