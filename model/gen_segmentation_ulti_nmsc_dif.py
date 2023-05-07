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


def _cce(p, t):
    # Cross entropy loss that can handle multi-scale classifiers.
    # 
    # If p is a list, process every element (and reduce to batch dim).
    # Each tensor is reduced by `mean` and reduced tensors are averaged
    # together.
    # (For multi-scale classifiers. Each scale is given equal weight.)
    if not isinstance(p, torch.Tensor):
        return sum([_cce(elem, t) for elem in p])/float(len(p))
    # Convert target list to torch tensor (batch_size, 1, 1, 1).
    t = torch.Tensor(t).reshape(-1,1,1).expand(-1,p.size(2),p.size(3)).long()
    if p.is_cuda:
        t = t.to(p.device)
    # Cross-entropy.
    out = F.cross_entropy(p, t)
    # Return if no dimensions beyond batch dim.
    if out.dim()<=1:
        return out
    # Else, return after reducing to batch dim.
    return out.view(out.size(0), -1).mean(1)    # Reduce to batch dim.


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
    def __init__(self, encoder_source, encoder_target, decoder_common_source, decoder_common_target, decoder_residual_source, decoder_residual_target, segmenter,
                 decoder_source, decoder_target,
                 disc_SA, disc_SB, disc_TA, disc_TB, disc_S, disc_T, shape_sample,  scaler=None, loss_rec=mae, 
                 loss_seg=None, loss_gan='hinge',
                 num_disc_updates=1, relativistic=False, grad_penalty=None,
                 disc_clip_norm=None,gen_clip_norm=None,  
                 lambda_disc=1, lambda_x_id=1, lambda_z_id=1, lambda_seg=1, 
                 lambda_mod_x_id=1, lambda_mod_z_id=1, lambda_mod_cyc=1, lambda_mod_disc=1,
                 debug_ac_gan=False, rng=None):
        super(segmentation_model, self).__init__()
        lambdas = OrderedDict((
            ('lambda_disc',       lambda_disc),
            ('lambda_x_id',       lambda_x_id),
            ('lambda_z_id',       lambda_z_id),
            ('lambda_seg',        lambda_seg),
            ('lambda_mod_x_id',       lambda_mod_x_id),
            ('lambda_mod_z_id',       lambda_mod_z_id),
            ('lambda_mod_disc',        lambda_mod_disc),
            ('lambda_mod_cyc',        lambda_mod_cyc)
            ))
        kwargs = OrderedDict((
            ('rng',               rng if rng else np.random.RandomState()),
            ('encoder_source',           encoder_source),
            ('decoder_common_source',    decoder_common_source),
            ('encoder_target',           encoder_target),
            ('decoder_common_target',    decoder_common_target),            
            ('decoder_residual_target',  decoder_residual_target),
            ('decoder_residual_source',  decoder_residual_source),
            ('decoder_source',  decoder_source),
            ('decoder_target',  decoder_target),
            ('shape_sample',      shape_sample),
            ('scaler',            scaler),
            ('loss_rec',          loss_rec),
            ('loss_seg',          loss_seg if loss_seg else dice_loss()),
            ('loss_gan',          loss_gan),
            ('num_disc_updates',  num_disc_updates),
            ('relativistic',      relativistic),
            ('grad_penalty',      grad_penalty),
            ('gen_clip_norm',     gen_clip_norm),
            ('disc_clip_norm',    disc_clip_norm),
            ('gan_objective',     gan_objective(loss_gan,
                                                relativistic=relativistic,
                                                grad_penalty_real=grad_penalty,
                                                grad_penalty_fake=None,
                                                grad_penalty_mean=0)),
            ('debug_ac_gan',      debug_ac_gan)
            ))
        self.separate_networks = OrderedDict((
            ('segmenter',         segmenter),
            ('disc_SA',            disc_SA),
            ('disc_SB',            disc_SB),
            ('disc_TA',            disc_TA),
            ('disc_TB',            disc_TB),
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
        keys_forward = ['encoder_source', 'decoder_common_source', 'encoder_target', 'decoder_common_target', 
                        'decoder_residual_source', 'decoder_residual_target',
                        'decoder_source', 'decoder_target','segmenter', 
                        'shape_sample',
                        'scaler', 'rng']
        kwargs_forward = dict([(key, val) for key, val in kwargs.items()
                               if key in keys_forward])
        self._forward = _forward(**kwargs_forward, **lambdas)
        if torch.cuda.device_count()>1:
            self._forward = nn.DataParallel(self._forward, output_device=-1)
        
        # Module to compute discriminator losses on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_D = ['gan_objective', 'disc_SA', 'disc_SB', 'disc_TA', 'disc_TB',
                  'disc_S', 'disc_T', 'scaler', 'debug_ac_gan']
        kwargs_D = dict([(key, val) for key, val in kwargs.items()
                         if key in keys_D])
        self._loss_D = _loss_D(**kwargs_D, **lambdas)
        if torch.cuda.device_count()>1:
            self._loss_D = nn.DataParallel(self._loss_D, output_device=-1)
        
        # Module to compute generator updates on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_G = ['gan_objective', 'disc_SA', 'disc_SB', 'disc_TA', 'disc_TB',
                  'disc_S', 'disc_T', 'scaler', 'loss_rec', 'debug_ac_gan']
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
    
    def forward(self, x_SA, x_SB, x_TA, x_TB, mask_S=None, mask_T=None, optimizer=None, rng=None):
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
                visible, hidden, intermediates = self._forward(x_SA=x_SA, x_SB=x_SB, x_TA=x_TA, x_TB=x_TB,
                                                               rng=rng)
        #####
        # Evaluate discriminator loss and update.
        loss_disc = defaultdict(int)
        loss_D = gradnorm_D = 0
        if self.lambda_disc or self.lambda_mod_disc:
            if intermediates['x_SASB_list'] is None:
                out_SASB = visible['x_SASB']
            else:
                out_SASB = intermediates['x_SASB_list']+[visible['x_SASB']]
                
            if intermediates['x_SAT_list'] is None:
                out_SAT = visible['x_SAT']
            else:
                out_SAT = intermediates['x_SAT_list']+[visible['x_SAT']]
                
            if intermediates['x_TATB_list'] is None:
                out_TATB = visible['x_TATB']
            else:
                out_TATB = intermediates['x_TATB_list']+[visible['x_TATB']]
                
            if intermediates['x_TAS_list'] is None:
                out_TAS = visible['x_TAS']
            else:
                out_TAS = intermediates['x_TAS_list']+[visible['x_TAS']]
                
            if intermediates['x_SBSA_list'] is None:
                out_SBSA = visible['x_SBSA']
            else:
                out_SBSA = intermediates['x_SBSA_list']+[visible['x_SBSA']]
                
            if intermediates['x_SBT_list'] is None:
                out_SBT = visible['x_SBT']
            else:
                out_SBT = intermediates['x_SBT_list']+[visible['x_SBT']]
                
                
            if intermediates['x_TBTA_list'] is None:
                out_TBTA = visible['x_TBTA']
            else:
                out_TBTA = intermediates['x_TBTA_list']+[visible['x_TBTA']]
                
            if intermediates['x_TBS_list'] is None:
                out_TBS = visible['x_TBS']
            else:
                out_TBS = intermediates['x_TBS_list']+[visible['x_TBS']]    
                
                
            for i in range(self.num_disc_updates):
                # Evaluate.
                with torch.set_grad_enabled(do_updates_bool):
                    with self._autocast_if_needed():
                        loss_disc = self._loss_D(
                            x_SA=x_SA, x_SB=x_SB, x_TA=x_TA, x_TB=x_TB, 
                            out_SASB=out_SASB, out_SAT=out_SAT,
                            out_TATB=out_TATB, out_TAS=out_TAS,
                            out_SBSA=out_SBSA, out_SBT=out_SBT, 
                            out_TBTA=out_TBTA, out_TBS=out_TBS)
                        loss_D = _reduce(loss_disc.values())
                # Update discriminator
                disc_SA = self.separate_networks['disc_SA']
                disc_SB = self.separate_networks['disc_SB']
                disc_TA = self.separate_networks['disc_TA']
                disc_TB = self.separate_networks['disc_TB']    
                disc_S = self.separate_networks['disc_S']
                disc_T = self.separate_networks['disc_T']
                if do_updates_bool:
                    clear_grad(optimizer['D'])
                    with self._autocast_if_needed():
                        _loss = loss_D.mean()
                    backward(_loss)
                    if self.disc_clip_norm:
                        if self.scaler is not None:
                            self.scaler.unscale_(optimizer['D'])
                        nn.utils.clip_grad_norm_(disc_SA.parameters(),
                                                 max_norm=self.disc_clip_norm)
                        nn.utils.clip_grad_norm_(disc_SB.parameters(),
                                                 max_norm=self.disc_clip_norm)
                        nn.utils.clip_grad_norm_(disc_TA.parameters(),
                                                 max_norm=self.disc_clip_norm)
                        nn.utils.clip_grad_norm_(disc_TB.parameters(),
                                                 max_norm=self.disc_clip_norm)
                        nn.utils.clip_grad_norm_(disc_T.parameters(),
                                                 max_norm=self.disc_clip_norm)
                        nn.utils.clip_grad_norm_(disc_S.parameters(),
                                                 max_norm=self.disc_clip_norm)
                    step(optimizer['D'])
                    gradnorm_D_specific = grad_norm(disc_SA)+grad_norm(disc_SB)+grad_norm(disc_TB)+grad_norm(disc_TA)
                    gradnorm_D_trans = grad_norm(disc_T)+grad_norm(disc_S)

        
        # Evaluate generator losses.
        gradnorm_G = 0
        with torch.set_grad_enabled(do_updates_bool):
            with self._autocast_if_needed():
                losses_G = self._loss_G(x_SA=x_SA, x_SB=x_SB, x_TA=x_TA, x_TB=x_TB, 
                                        x_SASB=visible['x_SASB'], x_SASA=visible['x_SASA'], 
                                        x_SAS=visible['x_SAS'], x_SAT=visible['x_SAT'],
                                        x_TATB=visible['x_TATB'], x_TATA=visible['x_TATA'], 
                                        x_TAT=visible['x_TAT'], x_TAS=visible['x_TAS'],
                                        x_TBTB=visible['x_TBTB'], x_TBTA=visible['x_TBTA'], 
                                        x_TBT=visible['x_TBT'], x_TBS=visible['x_TBS'],
                                        x_SBSB=visible['x_SBSB'], x_SBSA=visible['x_SBSA'], 
                                        x_SBS=visible['x_SBS'], x_SBT=visible['x_SBT'],
                                        x_SBTS=visible['x_SBTS'], x_SATS=visible['x_SATS'],
                                        x_TBST=visible['x_TBST'], x_TAST=visible['x_TAST'],
                                        
                                        c_SA=hidden['c_SA'], u_SA=hidden['u_SA'], 
                                        c_SASB=hidden['c_SASB'], c_SASA=hidden['c_SASA'], u_SASA=hidden['u_SASA'],
                                        s_SAS=hidden['s_SAS'], s_SAT=hidden['s_SAT'],
                                        
                                        c_TA=hidden['c_TA'], u_TA=hidden['u_TA'], 
                                        c_TATB=hidden['c_TATB'], c_TATA=hidden['c_TATA'], u_TATA=hidden['u_TATA'],
                                        s_TAT=hidden['s_TAT'], s_TAS=hidden['s_TAS'],
                                        
                                        c_TB=hidden['c_TB'], u_TB=hidden['u_TB'],
                                        u_TB_sampled=hidden['u_TB_sampled'],
                                        c_TBTB=hidden['c_TBTB'], c_TBTA=hidden['c_TBTA'], u_TBTA=hidden['u_TBTA'],
                                        s_TBT=hidden['s_TBT'], s_TBS=hidden['s_TBS'],
                                        
                                        c_SB=hidden['c_SB'], u_SB=hidden['u_SB'],
                                        u_SB_sampled=hidden['u_SB_sampled'],
                                        c_SBSB=hidden['c_SBSB'], c_SBSA=hidden['c_SBSA'], u_SBSA=hidden['u_SBSA'],
                                        s_SBS=hidden['s_SBS'], s_SBT=hidden['s_SBT'])

        
        # Compute segmentation loss outside of DataParallel modules,
        # avoiding various issues:
        # - scatter of small batch sizes can lead to empty tensors
        # - tracking mask indices is very messy
        # - Dice loss reduced before being returned; then, averaged over GPUs
        mask_S_packed = mask_T_packed = x_SAM_packed = x_TAM_packed = x_SATM_packed = x_TASM_packed = None
        if mask_S is not None:
            # Prepare a mask Tensor without None entries.
            mask_S_indices = [i for i, m in enumerate(mask_S) if m is not None]
            mask_S_packed = np.array([(mask_S[i]>0)*1 for i in mask_S_indices])
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
            mask_T_packed = np.array([(mask_T[i]>0)*1 for i in mask_T_indices])
            mask_T_packed = Variable(torch.from_numpy(mask_T_packed))
            if torch.cuda.device_count()==1:
                # `DataParallel` does not respect `output_device` when
                # there is only one GPU. So it returns outputs on GPU rather
                # than CPU, as requested. When this happens, putting mask
                # on GPU allows all values to stay on one device.
                mask_T_packed = mask_T_packed.cuda()

        loss_seg_trans_M = 0.
        loss_seg_M = 0.
        
        if self.lambda_seg and mask_T_packed is not None and len(mask_T_packed):
            if self.lambda_seg and mask_S_packed is not None and len(mask_S_packed):
                with self._autocast_if_needed():
                    
                    x_TASM_packed = visible['x_TASM'][mask_T_indices]
                    x_TAM_packed = visible['x_TAM'][mask_T_indices]
                    
                    x_SATM_packed = visible['x_SATM'][mask_S_indices]
                    x_SAM_packed = visible['x_SAM'][mask_S_indices]
                    
                    loss_seg_trans_M = self.lambda_seg*self.loss_seg(x_SATM_packed, mask_S_packed)+self.lambda_seg*self.loss_seg(x_TASM_packed, mask_T_packed)
                    
                    loss_seg_M = self.lambda_seg*self.loss_seg(x_SAM_packed, mask_S_packed)+self.lambda_seg*self.loss_seg(x_TAM_packed, mask_T_packed)
                    
                    x_SATM_packed = ((visible['x_SATM'][mask_S_indices] > 0.5)*1)
                    x_SAM_packed = ((visible['x_SAM'][mask_S_indices] > 0.5)*1)
                    x_TAM_packed = ((visible['x_TAM'][mask_T_indices] > 0.5)*1)
                    x_TASM_packed = ((visible['x_TASM'][mask_T_indices] > 0.5)*1)
                
            else :
                with self._autocast_if_needed():
                
                    x_TASM_packed = visible['x_TASM'][mask_T_indices]
                    x_TAM_packed = visible['x_TAM'][mask_T_indices]
                           
                    loss_seg_trans_M = self.lambda_seg*self.loss_seg(x_TASM_packed,mask_T_packed)
                    
                    loss_seg_M = self.lambda_seg*self.loss_seg(x_TAM_packed,mask_T_packed)
                    
                    x_TAM_packed = ((visible['x_TAM'][mask_T_indices] > 0.5)*1)
                    x_TASM_packed = ((visible['x_TASM'][mask_T_indices] > 0.5)*1)        
        
        if not (self.lambda_seg and mask_T_packed is not None and len(mask_T_packed)):
            with self._autocast_if_needed():
                
                    x_SATM_packed = visible['x_SATM'][mask_S_indices]
                    x_SAM_packed = visible['x_SAM'][mask_S_indices]
                           
                    loss_seg_trans_M = self.lambda_seg*self.loss_seg(x_SATM_packed,mask_S_packed)
                    
                    loss_seg_M = self.lambda_seg*self.loss_seg(x_SAM_packed,mask_S_packed)
                    
                    x_SAM_packed = ((visible['x_SAM'][mask_S_indices] > 0.5)*1)
                    x_SATM_packed = ((visible['x_SATM'][mask_S_indices] > 0.5)*1)  
        
        # Include segmentation loss with generator losses and update.
        with self._autocast_if_needed():
            losses_G['l_seg'] = _reduce([loss_seg_M, loss_seg_trans_M])
            losses_G['l_G'] += losses_G['l_seg']
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
            if self.gen_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.parameters(),
                                         max_norm=self.gen_clip_norm)
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
        outputs['x_SM'] = mask_S_packed
        outputs['x_TM'] = mask_T_packed
        outputs.update(visible)
        outputs['x_SAM'] = x_SAM_packed
        outputs['x_SATM'] = x_SATM_packed
        outputs['x_TAM'] = x_TAM_packed
        outputs['x_TASM'] = x_TASM_packed
        outputs.update(losses_G)
        outputs['l_D']  = loss_D
        outputs['l_DSA'] = _reduce([loss_disc['SA']])
        outputs['l_DTA'] = _reduce([loss_disc['TA']])
        outputs['l_DSB'] = _reduce([loss_disc['SB']])
        outputs['l_DTB'] = _reduce([loss_disc['TB']])
        outputs['l_DS'] = _reduce([loss_disc['S1'], loss_disc['S2']])
        outputs['l_DT'] = _reduce([loss_disc['T1'], loss_disc['T2']])
        outputs['l_gradnorm_D'] = gradnorm_D
        outputs['l_gradnorm_G'] = gradnorm_G
        
        return outputs


class _forward(nn.Module):
    def __init__(self, encoder_source, decoder_common_source, encoder_target, decoder_common_target, 
                 decoder_residual_source, decoder_residual_target, segmenter,
                 decoder_source, decoder_target,
                 shape_sample, decoder_autoencode=None, scaler=None,
                 lambda_disc=1, lambda_x_id=1, lambda_z_id=1, lambda_seg=1, 
                 lambda_mod_x_id=1, lambda_mod_z_id=1, lambda_mod_cyc=1, lambda_mod_disc=1,
                 rng=None):
        super(_forward, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.encoder_source            = encoder_source
        self.decoder_common_source     = decoder_common_source
        self.encoder_target            = encoder_target
        self.decoder_common_target     = decoder_common_target        
        self.decoder_residual_source   = decoder_residual_source
        self.decoder_residual_target   = decoder_residual_target
        self.decoder_source     = decoder_source 
        self.decoder_target     = decoder_target
        self.segmenter          = [segmenter]   # Separate params.
        self.shape_sample       = shape_sample
        self.decoder_autoencode = decoder_autoencode
        self.scaler             = scaler
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_seg         = lambda_seg
        self.lambda_mod_disc    = lambda_mod_disc
        self.lambda_mod_x_id    = lambda_mod_x_id
        self.lambda_mod_z_id    = lambda_mod_z_id
        self.lambda_mod_cyc     = lambda_mod_cyc
    
    def _z_sample(self, batch_size, rng=None):
        if rng is None:
            rng = self.rng
        sample = rng.randn(batch_size, *self.shape_sample).astype(np.float32)
        ret = Variable(torch.from_numpy(sample))
        ret = ret.to(torch.cuda.current_device())
        return ret

    
    @autocast_if_needed()
    def forward(self, x_SA, x_SB, x_TA, x_TB, rng=None):
        
        batch_size = len(x_SA)
        
        
        # Helper function for summing either two tensors or pairs of tensors
        # across two lists of tensors.
        def add(a, b):
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                return a+b
            else:
                assert not isinstance(a, torch.Tensor)
                assert not isinstance(b, torch.Tensor)
                return [elem_a+elem_b for elem_a, elem_b in zip(a, b)]
        
        # Helper function to split an output into the final image output
        # tensor and a list of intermediate tensors.
        def unpack(x):
            x_list = None
            if not isinstance(x, torch.Tensor):
                x, x_list = x[-1], x[:-1]
            return x, x_list
        
        # SA pathway (sick source)
        s_SA, skip_SA = self.encoder_source(x_SA)
        info_SA = {'skip_info': skip_SA}
        c_SA, u_SA = torch.split(s_SA, [s_SA.size(1)-self.shape_sample[0], self.shape_sample[0]], dim=1)

        x_SA_residual, skip_SAM = self.decoder_residual_source(torch.cat([c_SA, u_SA], dim=1), **info_SA)

        x_SASB, _ = self.decoder_common_source(c_SA, **info_SA)
        x_SASA = add(x_SASB, x_SA_residual)
        
        s_SASB, _ = self.encoder_source(x_SASB)
        c_SASB, _ = torch.split(s_SASB, [s_SASB.size(1)-self.shape_sample[0], self.shape_sample[0]], dim=1)
        s_SASA, _ = self.encoder_source(x_SASA)
        c_SASA, u_SASA = torch.split(s_SASA, [s_SASA.size(1)-self.shape_sample[0], self.shape_sample[0]], dim=1)
        
        x_SAS, _ = self.decoder_source(s_SA, **info_SA)
        s_SAS, _ = self.encoder_source(x_SAS)
        
        x_SAT, _ = self.decoder_target(s_SA, **info_SA)
        
        s_SAT, skip_SAT = self.encoder_target(x_SAT)
        info_SAT = {'skip_info': skip_SAT}
        
        c_SAT, u_SAT = torch.split(s_SAT, [s_SAT.size(1)-self.shape_sample[0], self.shape_sample[0]], dim=1)
        x_SAT_residual, skip_SATM = self.decoder_residual_target(torch.cat([c_SAT, u_SAT], dim=1), **info_SAT)  
        
        x_SATS, _ = self.decoder_source(torch.cat([c_SAT, u_SAT], dim=1), **info_SAT)

        # Unpack.
        x_SASB, x_SASB_list = unpack(x_SASB)
        x_SASA, x_SASA_list = unpack(x_SASA)
        x_SAS, x_SAS_list = unpack(x_SAS)
        x_SAT, x_SAT_list = unpack(x_SAT)
        x_SATS, x_SATS_list = unpack(x_SATS)
        x_SA_residual, _= unpack(x_SA_residual)
        x_SAT_residual, _= unpack(x_SAT_residual)
        
        # TA pathway (sick target)
        s_TA, skip_TA = self.encoder_target(x_TA)
        info_TA = {'skip_info': skip_TA}
        c_TA, u_TA = torch.split(s_TA, [s_TA.size(1)-self.shape_sample[0], self.shape_sample[0]], dim=1)
        
        
        x_TA_residual, skip_TAM = self.decoder_residual_target(torch.cat([c_TA, u_TA], dim=1), **info_TA)
        x_TATB, _ = self.decoder_common_target(c_TA, **info_TA)
        x_TATA = add(x_TATB, x_TA_residual)
        
        s_TATB, _ = self.encoder_target(x_TATB)
        c_TATB, _ = torch.split(s_TATB, [s_TATB.size(1)-self.shape_sample[0], self.shape_sample[0]], dim=1)
        s_TATA, _ = self.encoder_target(x_TATA)
        c_TATA, u_TATA = torch.split(s_TATA, [s_TATA.size(1)-self.shape_sample[0], self.shape_sample[0]], dim=1)
        
        x_TAT, _ = self.decoder_target(s_TA, **info_TA)
        s_TAT, _ = self.encoder_target(x_TAT)
        
        x_TAS, _ = self.decoder_source(s_TA, **info_TA)
        
        s_TAS, skip_TAS = self.encoder_source(x_TAS)
        info_TAS = {'skip_info': skip_TAS}

        c_TAS, u_TAS = torch.split(s_TAS, [s_TAS.size(1)-self.shape_sample[0], self.shape_sample[0]], dim=1) 
        x_TAS_residual, skip_TASM = self.decoder_residual_source(torch.cat([c_TAS, u_TAS], dim=1), **info_TAS)
        
        x_TAST, _ = self.decoder_target(torch.cat([c_TAS, u_TAS], dim=1), **info_TAS)
        
        # Unpack.
        x_TATB, x_TATB_list = unpack(x_TATB)
        x_TATA, x_TATA_list = unpack(x_TATA)
        x_TAT, x_TAT_list = unpack(x_TAT)
        x_TAS, x_TAS_list = unpack(x_TAS)
        x_TAST, x_TAST_list = unpack(x_TAST)
        x_TA_residual, _= unpack(x_TA_residual)
        x_TAS_residual, _= unpack(x_TAS_residual)
        
        # SB pathway (healthy source)
        s_SB, skip_SB = self.encoder_source(x_SB)
        info_SB = {'skip_info': skip_SB}
        
        u_SB_sampled = self._z_sample(batch_size, rng=rng)
        c_SB, u_SB  = torch.split(s_SB, [s_SB.size(1)-self.shape_sample[0], self.shape_sample[0]], dim=1)
        
        x_SB_residual, _ = self.decoder_residual_source(torch.cat([c_SB, u_SB_sampled], dim=1), **info_SB)
        x_SBSB, _ = self.decoder_common_source(c_SB, **info_SB)
        x_SBSA = add(x_SBSB, x_SB_residual)
        
        s_SBSB, _ = self.encoder_source(x_SBSB)
        c_SBSB, _ = torch.split(s_SBSB, [s_SBSB.size(1)-self.shape_sample[0], self.shape_sample[0]], dim=1)
        s_SBSA, _ = self.encoder_source(x_SBSA)
        c_SBSA, u_SBSA = torch.split(s_SBSA, [s_SBSA.size(1)-self.shape_sample[0], self.shape_sample[0]], dim=1)
        
        x_SBS, _ = self.decoder_source(s_SB, **info_SB)
        s_SBS, _ = self.encoder_source(x_SBS)
        
        x_SBT, _ = self.decoder_target(s_SB, **info_SB)
        
        s_SBT, skip_SBT = self.encoder_target(x_SBT)
        info_SBT = {'skip_info': skip_SBT}
        
        c_SBT, u_SBT = torch.split(s_SBT, [s_SBT.size(1)-self.shape_sample[0], self.shape_sample[0]], dim=1)        
        x_SBTS, _ = self.decoder_source(torch.cat([c_SBT, u_SBT], dim=1), **info_SBT)        

        # Unpack.
        x_SBSB, x_SBSB_list = unpack(x_SBSB)
        x_SBSA, x_SBSA_list = unpack(x_SBSA)
        x_SBS, x_SBS_list = unpack(x_SBS)
        x_SBT, x_SBT_list = unpack(x_SBT)
        x_SBTS, x_SBTS_list = unpack(x_SBTS)
        x_SB_residual, _ = unpack(x_SB_residual)

        # TB pathway (healthy target)
        s_TB, skip_TB = self.encoder_target(x_TB)
        info_TB = {'skip_info': skip_TB}
        
        u_TB_sampled = self._z_sample(batch_size, rng=rng)
        c_TB, u_TB  = torch.split(s_TB, [s_TB.size(1)-self.shape_sample[0], self.shape_sample[0]], dim=1)
        
        x_TB_residual, _ = self.decoder_residual_target(torch.cat([c_TB, u_TB_sampled], dim=1), **info_TB)
        x_TBTB, _ = self.decoder_common_target(c_TB, **info_TB)
        x_TBTA = add(x_TBTB, x_TB_residual)
        
        s_TBTB, _ = self.encoder_target(x_TBTB)
        c_TBTB, _ = torch.split(s_TBTB, [s_TBTB.size(1)-self.shape_sample[0], self.shape_sample[0]], dim=1)
        s_TBTA, _ = self.encoder_target(x_TBTA)
        c_TBTA, u_TBTA = torch.split(s_TBTA, [s_TBTA.size(1)-self.shape_sample[0], self.shape_sample[0]], dim=1)
        
        x_TBT, _ = self.decoder_target(s_TB, **info_TB)
        s_TBT, _ = self.encoder_target(x_TBT)
        
        x_TBS, _ = self.decoder_source(s_TB, **info_TB)
        
        s_TBS, skip_TBS = self.encoder_source(x_TBS)
        info_TBS = {'skip_info': skip_TBS}
        
        c_TBS, u_TBS = torch.split(s_TBS, [s_TBS.size(1)-self.shape_sample[0], self.shape_sample[0]], dim=1)        
        x_TBST, _ = self.decoder_target(torch.cat([c_TBS, u_TBS], dim=1), **info_TBS)        

        # Unpack.
        x_TBTB, x_TBTB_list = unpack(x_TBTB)
        x_TBTA, x_TBTA_list = unpack(x_TBTA)
        x_TBT, x_TBT_list = unpack(x_TBT)
        x_TBS, x_TBS_list = unpack(x_TBS)
        x_TBST, x_TBST_list = unpack(x_TBST)
        x_TB_residual, _ = unpack(x_TB_residual)

        
        
        # Segment.
        x_SAM = x_SATM = x_TAM = x_TASM = None
        if self.lambda_seg:
            if self.segmenter[0] is not None:
                x_SAM = self.segmenter[0](s_SA, skip_info=skip_SAM)
                x_SATAM = self.segmenter[0](s_SATA, skip_info=skip_SATAM)
                x_TAM = self.segmenter[0](s_TA, skip_info=skip_TAM)
                x_TASAM = self.segmenter[0](s_TASA, skip_info=skip_TASAM)
            else:
                # Re-use residual decoder in mode 2.
                info_SAM = {'skip_info': skip_SAM}
                info_SATM = {'skip_info': skip_SATM}
                info_TAM = {'skip_info': skip_TAM}
                info_TASM = {'skip_info': skip_TASM}
                x_SAM = self.decoder_residual_source(torch.cat([c_SA, u_SA], dim=1), **info_SAM, mode=1)
                x_SATM = self.decoder_residual_target(torch.cat([c_SAT, u_SAT], dim=1), **info_SATM, mode=1)
                x_TAM = self.decoder_residual_target(torch.cat([c_TA, u_TA], dim=1), **info_TAM, mode=1)
                x_TASM = self.decoder_residual_source(torch.cat([c_TAS, u_TAS], dim=1), **info_TASM, mode=1)
                x_SAM, _ = unpack(x_SAM)
                x_SATM, _ = unpack(x_SATM)
                x_TAM, _ = unpack(x_TAM)
                x_TASM, _ = unpack(x_TASM)
        
        
        # Compile outputs and return.
        visible = OrderedDict((
            
            
            ('x_SA',           x_SA),
            ('x_SASB',          x_SASB),
            ('x_SA_residual', x_SA_residual),
            ('x_SAM',          x_SAM),
            ('x_SASA',          x_SASA),
            ('x_SAS',          x_SAS),
            ('x_SAT',          x_SAT),
            ('x_SAT_residual', x_SAT_residual),
            ('x_SATM',        x_SATM),
            ('x_SATS',          x_SATS),
            
            ('x_TA',           x_TA),
            ('x_TATB',          x_TATB),
            ('x_TA_residual', x_TA_residual),
            ('x_TAM',          x_TAM),
            ('x_TATA',          x_TATA),
            ('x_TAT',          x_TAT),
            ('x_TAS',         x_TAS),
            ('x_TAS_residual', x_TAS_residual),
            ('x_TASM',        x_TASM),            
            ('x_TAST',          x_TAST),
            
            ('x_TB',           x_TB),
            ('x_TBTB',          x_TBTB),
            ('x_TB_residual', x_TB_residual),
            ('x_TBTA',          x_TBTA),
            ('x_TBT',          x_TBT),
            ('x_TBS',           x_TBS),
            ('x_TBST',          x_TBST),
            
            ('x_SB',           x_SB),
            ('x_SBSB',          x_SBSB),
            ('x_SB_residual', x_SB_residual),
            ('x_SBSA',          x_SBSA),
            ('x_SBS',          x_SBS),
            ('x_SBT',          x_SBT),
            ('x_SBTS',          x_SBTS)
            ))
        hidden = OrderedDict((
            ('c_SA',          c_SA),
            ('u_SA',          u_SA),
            ('c_SASB',          c_SASB),
            ('c_SASA',          c_SASA),
            ('u_SASA',          u_SASA),
            ('s_SAS',          s_SAS),
            ('s_SAT',          s_SAT[:,s_SAT.size(1)-self.shape_sample[0]:]),
            
            ('c_TA',          c_TA),
            ('u_TA',          u_TA),
            ('c_TATB',          c_TATB),
            ('c_TATA',          c_TATA),
            ('u_TATA',          u_TATA),
            ('s_TAT',          s_TAT),
            ('s_TAS',          s_TAS[:,s_TAS.size(1)-self.shape_sample[0]:]),
            
            ('c_SB',          c_SB),
            ('u_SB',          u_SB),
            ('u_SB_sampled',          u_SB_sampled),
            ('c_SBSB',          c_SBSB),
            ('c_SBSA',          c_SBSA),
            ('u_SBSA',          u_SBSA),
            ('s_SBS',          s_SBS),
            ('s_SBT',          s_SBT[:,s_SBT.size(1)-self.shape_sample[0]:]),
            
            ('c_TB',          c_TB),
            ('u_TB',          u_TB),
            ('u_TB_sampled',          u_TB_sampled),
            ('c_TBTB',          c_TBTB),
            ('c_TBTA',          c_TBTA),
            ('u_TBTA',          u_TBTA),
            ('s_TBT',          s_TBT),
            ('s_TBS',          s_TBS[:,s_TBS.size(1)-self.shape_sample[0]:])
            ))
        intermediates = OrderedDict((
            ('x_SASB_list',     x_SASB_list),
            ('x_SAT_list',     x_SAT_list),
            ('x_TATB_list',     x_TATB_list),
            ('x_TAS_list',     x_TAS_list),
            ('x_SBSA_list',     x_SBSA_list),
            ('x_SBT_list',     x_SBT_list),
            ('x_TBTA_list',     x_TBTA_list),
            ('x_TBS_list',     x_TBS_list)
            ))
        return visible, hidden, intermediates


class _loss_D(nn.Module):
    def __init__(self, gan_objective, disc_SA, disc_SB, disc_TA, disc_TB, disc_T, disc_S, 
                 scaler=None, lambda_disc=1, lambda_x_id=1, lambda_z_id=1, lambda_seg=1, 
                 lambda_mod_x_id=1, lambda_mod_z_id=1, lambda_mod_cyc=1, lambda_mod_disc=1, debug_ac_gan=False):
        super(_loss_D, self).__init__()
        self._gan               = gan_objective
        self.scaler             = scaler
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_seg         = lambda_seg
        self.lambda_mod_disc    = lambda_mod_disc
        self.lambda_mod_x_id    = lambda_mod_x_id
        self.lambda_mod_z_id    = lambda_mod_z_id
        self.lambda_mod_cyc     = lambda_mod_cyc
        self.debug_ac_gan       = debug_ac_gan
        self.net = {'disc_SA'    : disc_SA,
                    'disc_SB'    : disc_SB,
                    'disc_TA'    : disc_TA,
                    'disc_TB'    : disc_TB,
                    'disc_T'    : disc_T,
                    'disc_S'    : disc_S}  # Separate params.
    
    @autocast_if_needed()
    def forward(self, x_SA, x_SB, x_TA, x_TB, 
                out_SASB, out_SAT,
                out_TATB, out_TAS,
                out_SBSA, out_SBT,
                out_TBTA, out_TBS):

        # Detach all tensors; updating discriminator, not generator.
            
        if isinstance(out_SASB, list):
            out_SASB = [x.detach() for x in out_SASB]
        else:
            out_SASB = out_SASB.detach()

            
        if isinstance(out_SAT, list):
            out_SAT = [x.detach() for x in out_SAT]
        else:
            out_SAT = out_SAT.detach()
            
        if isinstance(out_TATB, list):
            out_TATB = [x.detach() for x in out_TATB]
        else:
            out_TATB = out_TATB.detach()

            
        if isinstance(out_TAS, list):
            out_TAS = [x.detach() for x in out_TAS]
        else:
            out_TAS = out_TAS.detach()
            

        if isinstance(out_SBSA, list):
            out_SBSA = [x.detach() for x in out_SBSA]
        else:
            out_SBSA = out_SBSA.detach()

            
        if isinstance(out_SBT, list):
            out_SBT = [x.detach() for x in out_SBT]
        else:
            out_SBT = out_SBT.detach()
            

        if isinstance(out_TBTA, list):
            out_TBTA = [x.detach() for x in out_TBTA]
        else:
            out_TBTA = out_TBTA.detach()
            
        if isinstance(out_TBS, list):
            out_TBS = [x.detach() for x in out_TBS]
        else:
            out_TBS = out_TBS.detach()
        
        # Discriminators.
        kwargs_real = None 
        kwargs_fake = None 
        loss_disc = OrderedDict()
        if self.lambda_disc or self.lambda_mod_disc:
            loss_disc['SA'] = self.lambda_disc*self._gan.D(self.net['disc_SA'],
                                     fake=out_SBSA,
                                     real=x_SA,
                                     kwargs_real=kwargs_real,
                                     kwargs_fake=kwargs_fake,
                                     scaler=self.scaler)
            
            loss_disc['TA'] = self.lambda_disc*self._gan.D(self.net['disc_TA'],
                                     fake=out_TBTA,
                                     real=x_TA,
                                     kwargs_real=kwargs_real,
                                     kwargs_fake=kwargs_fake,
                                     scaler=self.scaler)
            
            loss_disc['TB'] = self.lambda_disc*self._gan.D(self.net['disc_TB'],
                                     fake=out_TATB,
                                     real=x_TB,
                                     kwargs_real=kwargs_real,
                                     kwargs_fake=kwargs_fake,
                                     scaler=self.scaler)
         
            loss_disc['SB'] = self.lambda_disc*self._gan.D(self.net['disc_SB'],
                                      fake=out_SASB,
                                      real=x_SB,
                                      kwargs_real=kwargs_real,
                                      kwargs_fake=kwargs_fake,
                                      scaler=self.scaler)
        if self.lambda_mod_disc or self.lambda_disc:    
            loss_disc['S1'] = self.lambda_mod_disc*self._gan.D(self.net['disc_S'],
                                      fake=out_TAS,
                                      real=x_SA,
                                      kwargs_real=kwargs_real,
                                      kwargs_fake=kwargs_fake,
                                      scaler=self.scaler)
            
            loss_disc['S2'] = self.lambda_mod_disc*self._gan.D(self.net['disc_S'],
                                      fake=out_TBS,
                                      real=x_SB,
                                      kwargs_real=kwargs_real,
                                      kwargs_fake=kwargs_fake,
                                      scaler=self.scaler)
          
            loss_disc['T1'] = self.lambda_mod_disc*self._gan.D(self.net['disc_T'],
                                      fake=out_SAT,
                                      real=x_TA,
                                      kwargs_real=kwargs_real,
                                      kwargs_fake=kwargs_fake,
                                      scaler=self.scaler)
            
            loss_disc['T2'] = self.lambda_mod_disc*self._gan.D(self.net['disc_T'],
                                      fake=out_SBT,
                                      real=x_TB,
                                      kwargs_real=kwargs_real,
                                      kwargs_fake=kwargs_fake,
                                      scaler=self.scaler)
            
        return loss_disc


class _loss_G(nn.Module):
    def __init__(self, gan_objective, disc_SA, disc_SB, disc_TA, disc_TB, disc_T, disc_S, scaler=None,
                 loss_rec=mae, lambda_disc=1, lambda_x_id=1, lambda_z_id=1, lambda_seg=1, 
                 lambda_mod_x_id=1, lambda_mod_z_id=1, lambda_mod_cyc=1, lambda_mod_disc=1,
                 debug_ac_gan=False):
        super(_loss_G, self).__init__()
        self._gan               = gan_objective
        self.scaler             = scaler
        self.loss_rec           = loss_rec
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_seg         = lambda_seg
        self.lambda_mod_disc    = lambda_mod_disc
        self.lambda_mod_x_id    = lambda_mod_x_id
        self.lambda_mod_z_id    = lambda_mod_z_id
        self.lambda_mod_cyc     = lambda_mod_cyc
        self.debug_ac_gan       = debug_ac_gan
        self.net = {'disc_SA'    : disc_SA,
                    'disc_SB'    : disc_SB,
                    'disc_TA'    : disc_TA,
                    'disc_TB'    : disc_TB,
                    'disc_T'    : disc_T,
                    'disc_S'    : disc_S}  # Separate params.
    
    @autocast_if_needed()
    def forward(self, x_SA, x_SB, x_TA, x_TB, 
                x_SASB, x_SASA, x_SAS, x_SAT, x_SATS,
                x_TATB, x_TATA, x_TAT, x_TAS, x_TAST,
                x_TBTB, x_TBTA, x_TBT, x_TBS, x_TBST,
                x_SBSB, x_SBSA, x_SBS, x_SBT, x_SBTS,
                c_SA, u_SA, c_SASB, c_SASA, u_SASA,
                s_SAS, s_SAT,
                c_TA, u_TA, c_TATB, c_TATA, u_TATA,
                s_TAT, s_TAS,
                c_TB, u_TB, u_TB_sampled, c_TBTB, c_TBTA, u_TBTA,
                s_TBT, s_TBS,
                c_SB, u_SB, u_SB_sampled, c_SBSB, c_SBSA, u_SBSA,
                s_SBS, s_SBT):
                                     

        # Generator loss.
        loss_gen = defaultdict(int)
        kwargs_real = None
        kwargs_fake = None  
              
        if self.lambda_disc:
            loss_gen['SA'] = self.lambda_disc*self._gan.G(self.net['disc_SA'],
                         fake=x_SBSA,
                         real=x_SA,
                         kwargs_real=kwargs_real,
                         kwargs_fake=kwargs_fake)
            loss_gen['TA'] = self.lambda_disc*self._gan.G(self.net['disc_TA'],
                         fake=x_TBTA,
                         real=x_TA,
                         kwargs_real=kwargs_real,
                         kwargs_fake=kwargs_fake)
            loss_gen['TB'] = self.lambda_disc*self._gan.G(self.net['disc_TB'],
                         fake=x_TATB,
                         real=x_TB,
                         kwargs_real=kwargs_real,
                         kwargs_fake=kwargs_fake)
            loss_gen['SB'] = self.lambda_disc*self._gan.G(self.net['disc_SB'],
                          fake=x_SASB,
                          real=x_SB,
                          kwargs_real=kwargs_real,
                          kwargs_fake=kwargs_fake)
                          
        if self.lambda_mod_disc:
            loss_gen['S1'] = self.lambda_mod_disc*self._gan.G(self.net['disc_S'],
                          fake=x_TAS,
                          real=x_SA,
                          kwargs_real=kwargs_real,
                          kwargs_fake=kwargs_fake)
            loss_gen['S2'] = self.lambda_mod_disc*self._gan.G(self.net['disc_S'],
                          fake=x_TBS,
                          real=x_SB,
                          kwargs_real=kwargs_real,
                          kwargs_fake=kwargs_fake)
            loss_gen['T1'] = self.lambda_mod_disc*self._gan.G(self.net['disc_T'],
                          fake=x_SAT,
                          real=x_TA,
                          kwargs_real=kwargs_real,
                          kwargs_fake=kwargs_fake)
            loss_gen['T2'] = self.lambda_mod_disc*self._gan.G(self.net['disc_T'],
                          fake=x_SBT,
                          real=x_TB,
                          kwargs_real=kwargs_real,
                          kwargs_fake=kwargs_fake)

        
        # Reconstruction loss.
        loss_rec = defaultdict(int)
        if self.lambda_x_id:
            loss_rec['x_SASA'] = self.lambda_x_id*self.loss_rec(x_SA, x_SASA)
            loss_rec['x_TATA'] = self.lambda_x_id*self.loss_rec(x_TA, x_TATA)
            loss_rec['x_SBSB'] = self.lambda_x_id*self.loss_rec(x_SB, x_SBSB)
            loss_rec['x_TBTB'] = self.lambda_x_id*self.loss_rec(x_TB, x_TBTB)

        if self.lambda_mod_x_id:
            loss_rec['x_SAS'] = self.lambda_mod_x_id*self.loss_rec(x_SA, x_SAS)
            loss_rec['x_TAT'] = self.lambda_mod_x_id*self.loss_rec(x_TA, x_TAT)
            loss_rec['x_SBS'] = self.lambda_mod_x_id*self.loss_rec(x_SB, x_SBS)
            loss_rec['x_TBT'] = self.lambda_mod_x_id*self.loss_rec(x_TB, x_TBT)
            
        if self.lambda_z_id:
            loss_rec['SASB'] = self.lambda_z_id*self.loss_rec(c_SA, c_SASB)
            loss_rec['SASA'] = self.lambda_z_id*self.loss_rec(torch.cat([c_SA, u_SA], dim=1), torch.cat([c_SASA, u_SASA], dim=1))
            loss_rec['TATB'] = self.lambda_z_id*self.loss_rec(c_TA, c_TATB)
            loss_rec['TATA'] = self.lambda_z_id*self.loss_rec(torch.cat([c_TA, u_TA], dim=1), torch.cat([c_TATA, u_TATA], dim=1))
            loss_rec['TBTB'] = self.lambda_z_id*self.loss_rec(c_TB, c_TBTB)
            loss_rec['TBTA'] = self.lambda_z_id*self.loss_rec(torch.cat([c_TB, u_TB_sampled], dim=1), torch.cat([c_TBTA, u_TBTA], dim=1))
            loss_rec['SBSB'] = self.lambda_z_id*self.loss_rec(c_SB, c_SBSB)
            loss_rec['SBSA'] = self.lambda_z_id*self.loss_rec(torch.cat([c_SB, u_SB_sampled], dim=1), torch.cat([c_SBSA, u_SBSA], dim=1))

        if self.lambda_mod_z_id and self.lambda_mod_x_id:  
            loss_rec['SAS'] = self.lambda_mod_z_id*self.loss_rec(torch.cat([c_SA, u_SA], dim=1), s_SAS)
            loss_rec['TAT'] = self.lambda_mod_z_id*self.loss_rec(torch.cat([c_TA, u_TA], dim=1), s_TAT)
            loss_rec['TBT'] = self.lambda_mod_z_id*self.loss_rec(torch.cat([c_TB, u_TB], dim=1), s_TBT)
            loss_rec['SBS'] = self.lambda_mod_z_id*self.loss_rec(torch.cat([c_SB, u_SB], dim=1), s_SBS)
                        
        if self.lambda_mod_z_id:    
            loss_rec['SAT'] = self.lambda_mod_z_id*self.loss_rec(u_SA, s_SAT) 
            loss_rec['TAS'] = self.lambda_mod_z_id*self.loss_rec(u_TA, s_TAS)    
            loss_rec['TBS'] = self.lambda_mod_z_id*self.loss_rec(u_TB, s_TBS) 
            loss_rec['SBT'] = self.lambda_mod_z_id*self.loss_rec(u_SB, s_SBT)    
            
        if self.lambda_mod_cyc:
            loss_rec['x_SATS'] = self.lambda_mod_cyc*self.loss_rec(x_SA, x_SATS)
            loss_rec['x_TAST'] = self.lambda_mod_cyc*self.loss_rec(x_TA, x_TAST)
            loss_rec['x_SBTS'] = self.lambda_mod_cyc*self.loss_rec(x_SB, x_SBTS)
            loss_rec['x_TBST'] = self.lambda_mod_cyc*self.loss_rec(x_TB, x_TBST)
            
        # All generator losses combined.
        loss_G = ( _reduce(loss_gen.values())
                  +_reduce(loss_rec.values()))
        
        # Compile outputs and return.
        losses = OrderedDict((
            ('l_G',           loss_G),
            ('l_gen_SA',      _reduce([loss_gen['SA']])),
            ('l_gen_TA',      _reduce([loss_gen['TA']])),
            ('l_gen_SB',      _reduce([loss_gen['SB']])),
            ('l_gen_TB',      _reduce([loss_gen['TB']])),
            ('l_gen_S',      _reduce([loss_gen['S1'], loss_gen['S2']])),
            ('l_gen_T',      _reduce([loss_gen['T1'], loss_gen['T2']])),
            ('l_rec_img',         _reduce([loss_rec['x_SASA'], loss_rec['x_TATA'], loss_rec['x_SBSB'], loss_rec['x_TBTB'],
                                           loss_rec['x_SAS'], loss_rec['x_TAT'], loss_rec['x_SBS'], loss_rec['x_TBT'],
                                           loss_rec['x_SATS'], loss_rec['x_TAST'], loss_rec['x_SBTS'], loss_rec['x_TBST']])),
            ('l_rec_features',       _reduce([loss_rec['SASB'], loss_rec['SASA'], loss_rec['SAS'], loss_rec['SAT'],
                                              loss_rec['TATB'], loss_rec['TATA'], loss_rec['TAT'], loss_rec['TAS'],
                                              loss_rec['SBSA'], loss_rec['SBSB'], loss_rec['SBS'], loss_rec['SBT'],
                                              loss_rec['TBTB'], loss_rec['TBTA'], loss_rec['TBT'], loss_rec['TBS']]))
            ))
        return losses

