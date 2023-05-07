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

def clear_grad(optimizer):
    # Sets `grad` to None instead of zeroing it.
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad = None

class MIND(torch.nn.Module):
    def __init__(self, non_local_region_size=9, patch_size=7, neighbor_size=3, gaussian_patch_sigma=3.0):
        super(MIND, self).__init__()
        self.nl_size = non_local_region_size
        self.p_size = patch_size
        self.n_size = neighbor_size
        self.sigma2 = gaussian_patch_sigma * gaussian_patch_sigma

        # calc shifted images in non local region
        self.image_shifter = torch.nn.Conv2d(in_channels=1, out_channels=self.nl_size * self.nl_size,
                                             kernel_size=(self.nl_size, self.nl_size),
                                             stride=1, padding=((self.nl_size-1)//2, (self.nl_size-1)//2),
                                             dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.nl_size*self.nl_size):
            t = torch.zeros((1, self.nl_size, self.nl_size))
            t[0, i % self.nl_size, i//self.nl_size] = 1
            self.image_shifter.weight.data[i] = t

        # patch summation
        self.summation_patcher = torch.nn.Conv2d(in_channels=self.nl_size*self.nl_size, out_channels=self.nl_size*self.nl_size,
                                                 kernel_size=(self.p_size, self.p_size),
                                                 stride=1, padding=((self.p_size-1)//2, (self.p_size-1)//2),
                                                 dilation=1, groups=self.nl_size*self.nl_size, bias=False, padding_mode='zeros')

        for i in range(self.nl_size*self.nl_size):
            # gaussian kernel
            t = torch.zeros((1, self.p_size, self.p_size))
            cx = (self.p_size-1)//2
            cy = (self.p_size-1)//2
            for j in range(self.p_size * self.p_size):
                x = j % self.p_size
                y = j//self.p_size
                d2 = torch.norm(torch.tensor([x-cx, y-cy]).float(), 2)
                t[0, x, y] = math.exp(-d2 / self.sigma2)

            self.summation_patcher.weight.data[i] = t

        # neighbor images
        self.neighbors = torch.nn.Conv2d(in_channels=1, out_channels=self.n_size*self.n_size,
                                         kernel_size=(self.n_size, self.n_size),
                                         stride=1, padding=((self.n_size-1)//2, (self.n_size-1)//2),
                                         dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.n_size*self.n_size):
            t = torch.zeros((1, self.n_size, self.n_size))
            t[0, i % self.n_size, i//self.n_size] = 1
            self.neighbors.weight.data[i] = t


        # neighbor patcher
        self.neighbor_summation_patcher = torch.nn.Conv2d(in_channels=self.n_size*self.n_size, out_channels=self.n_size * self.n_size,
                                                          kernel_size=(self.p_size, self.p_size),
                                                          stride=1, padding=((self.p_size-1)//2, (self.p_size-1)//2),
                                                          dilation=1, groups=self.n_size*self.n_size, bias=False, padding_mode='zeros')

        for i in range(self.n_size*self.n_size):
            t = torch.ones((1, self.p_size, self.p_size))
            self.neighbor_summation_patcher.weight.data[i] = t

    def forward(self, orig):
        assert(len(orig.shape) == 4)
        assert(orig.shape[1] == 1)

        # get original image channel stack
        orig_stack = torch.stack([orig.squeeze(dim=1) for i in range(self.nl_size*self.nl_size)], dim=1)

        # get shifted images
        shifted = self.image_shifter(orig)

        # get image diff
        diff_images = shifted - orig_stack

        # diff's L2 norm
        Dx_alpha = self.summation_patcher(torch.pow(diff_images, 2.0))

        # calc neighbor's variance
        neighbor_images = self.neighbor_summation_patcher(self.neighbors(orig))
        Vx = neighbor_images.var(dim=1).unsqueeze(dim=1)

        # output mind
        nume = torch.exp(-Dx_alpha / (Vx + 1e-8))
        denomi = nume.sum(dim=1).unsqueeze(dim=1)
        mind = nume / denomi
        return mind                
                
class MINDLoss(torch.nn.Module):
    def __init__(self, non_local_region_size=9, patch_size=7, neighbor_size=3, gaussian_patch_sigma=3.0):
        super(MINDLoss, self).__init__()
        self.nl_size = non_local_region_size
        self.MIND = MIND(non_local_region_size=non_local_region_size,
                         patch_size=patch_size,
                         neighbor_size=neighbor_size,
                         gaussian_patch_sigma=gaussian_patch_sigma)

    def forward(self, input, target):
        in_mind = self.MIND(input)
        tar_mind = self.MIND(target)
        mind_diff = in_mind - tar_mind
        l1 =torch.norm(mind_diff, 1)
        return l1/(input.shape[2] * input.shape[3] * self.nl_size * self.nl_size)                
                
class PatchNCELoss(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool
        self.batch_size=batch_size

    def forward(self, feat_q, feat_k):

        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        batch_dim_for_bmm = self.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / 0.07

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss                
                


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
    def __init__(self, generator_S, disc_S, disc_T, segmentor, generator_T, loss_seg=None, loss_gan='hinge',
                 num_disc_updates=1, batch_size=1, lambda_seg=1, lambda_disc=1, lambda_id=1, lambda_contraste=1, lambda_anatomy=1,
                 scaler=None, debug_ac_gan=False, rng=None, relativistic=False, grad_penalty=None):
        super(segmentation_model, self).__init__()

        lambdas = OrderedDict((
            ('lambda_seg',             lambda_seg),
            ('lambda_disc',        lambda_disc),
            ('lambda_id',        lambda_id),
            ('lambda_contraste',        lambda_contraste),
            ('lambda_anatomy',        lambda_anatomy)
            ))
        kwargs = OrderedDict((
            ('rng',               rng if rng else np.random.RandomState()),
            ('scaler',            scaler),
            ('segmentor',    segmentor),
            ('generator_S',    generator_S),
            ('loss_seg',          loss_seg),
            ('loss_gan',          loss_gan),
            ('generator_T',    generator_T),
            #('feature_extractor',          feature_extractor),
            ('num_disc_updates',  num_disc_updates),
            ('relativistic',      relativistic),
            ('grad_penalty',      grad_penalty),
            ('gan_objective',     gan_objective(loss_gan,
                                                relativistic=relativistic,
                                                grad_penalty_real=grad_penalty,
                                                grad_penalty_fake=None,
                                                grad_penalty_mean=0)),
            ('debug_ac_gan',      debug_ac_gan),
            ('criterion_NCE', [PatchNCELoss(batch_size).cuda() for i in [0,4,8,12,16]]),
            ('criterion_MIND', MINDLoss(non_local_region_size=9, patch_size=7, neighbor_size=3, gaussian_patch_sigma=3.0).cuda())
            ))
        self.separate_networks = OrderedDict([
            ('disc_S',             disc_S),
            ('disc_T',             disc_T)
            ])
            
        kwargs.update(lambdas)
        for key, val in kwargs.items():
            setattr(self, key, val)

        # Separate networks not stored directly as attributes.
        # -> Separate parameters, separate optimizers.
        kwargs.update(self.separate_networks)
        
        # Module to compute all network outputs (except discriminator) on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_forward = ['segmentor', 'generator_S', 'generator_T', 'scaler', 'rng']
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
        keys_G = ['gan_objective', 'disc_S', 'disc_T', 'criterion_NCE', 'criterion_MIND', 'scaler', 'debug_ac_gan']
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
        
        # Compute all outputs.
        with torch.set_grad_enabled(do_updates_bool):
            with self._autocast_if_needed():
                visible = self._forward(x_S=x_S, x_T=x_T, rng=rng)
        #####
        # Evaluate discriminator loss and update.
        loss_disc = defaultdict(int)
        loss_D = gradnorm_D = 0
        
        for i in range(self.num_disc_updates):
            # Evaluate.
            with torch.set_grad_enabled(do_updates_bool):
                with self._autocast_if_needed():
                    loss_disc = self._loss_D(out_ST=visible['x_ST'], x_T=visible['x_T'], out_TS=visible['x_TS'], x_S=visible['x_S'])
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
                gradnorm_D = grad_norm(disc_S)+grad_norm(disc_T)

        
        # Evaluate generator losses.
        gradnorm_G = 0
        with torch.set_grad_enabled(do_updates_bool):
            with self._autocast_if_needed():
                losses_G = self._loss_G(x_TT=visible['x_TT'], x_ST=visible['x_ST'], x_STS=visible['x_STS'], x_SS=visible['x_SS'],
                                        x_T=visible['x_T'], x_S=visible['x_S'], x_TST=visible['x_TST'], x_TS=visible['x_TS'])
  

        
        mask_S_packed = mask_T_packed = x_STM_packed = x_TM_packed = None
        
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
                
        loss_seg = 0.
        
        if mask_T_packed is not None and len(mask_T_packed):
                with self._autocast_if_needed():
                    
                    x_STM_packed = visible['x_STM'][mask_S_indices]
                    x_TM_packed = visible['x_TM'][mask_T_indices]
                    
                    loss_seg = self.lambda_seg*self.loss_seg(torch.cat((x_STM_packed,x_TM_packed), dim=0),torch.cat((mask_S_packed,mask_T_packed), dim=0))
                    
                    
                    x_STM_packed = ((visible['x_STM'][mask_S_indices] > 0.5)*1)
                    x_TM_packed = ((visible['x_TM'][mask_T_indices] > 0.5)*1)
     
        
        if not (mask_T_packed is not None and len(mask_T_packed)):
            with self._autocast_if_needed():
                
                    x_STM_packed = visible['x_STM'][mask_S_indices]
                           
                    loss_seg = self.lambda_seg*self.loss_seg(x_STM_packed, mask_S_packed)
                    
                    x_STM_packed = ((visible['x_STM'][mask_S_indices] > 0.5)*1)

        # Include segmentation loss with generator losses and update.
        with self._autocast_if_needed():
            losses_G['l_seg'] = _reduce([loss_seg])
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
        outputs['x_S'] = visible['x_S']
        outputs['x_ST'] = visible['x_ST']
        outputs['x_STS'] = visible['x_STS']
        outputs['x_SS'] = visible['x_SS']
        outputs['x_SM_pred'] = x_STM_packed
        outputs['x_TM'] = mask_T_packed
        outputs['x_T'] = visible['x_T']
        outputs['x_TS'] = visible['x_TS']
        outputs['x_TST'] = visible['x_TST']
        outputs['x_TT'] = visible['x_TT']
        outputs['x_TM_pred'] = x_TM_packed
        outputs.update(losses_G)
        outputs['l_D']  = loss_D
        outputs['l_gradnorm_D'] = gradnorm_D
        outputs['l_gradnorm_G'] = gradnorm_G
        return outputs


class _forward(nn.Module):
    def __init__(self, segmentor, generator_S, generator_T, lambda_seg=1, lambda_disc=1, lambda_id=1, lambda_anatomy=1, lambda_contraste=1, scaler=None, rng=None):
        super(_forward, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.segmentor          = segmentor
        self.generator_S          = generator_S   
        self.generator_T          = generator_T                                      
        self.scaler             = scaler

   
    @autocast_if_needed()
    def forward(self, x_S, x_T, rng=None):

        x_ST = self.generator_T(x_S)
        x_STS = self.generator_S(x_ST)
        x_SS = self.generator_S(x_S)   
        x_TS = self.generator_S(x_T)  
        x_TST = self.generator_T(x_TS)
        x_TT = self.generator_T(x_T)        
        x_STM = self.segmentor(x_ST)
        x_TM = self.segmentor(x_T)                                          

        # Compile outputs and return.
        visible = OrderedDict((
            ('x_S',           x_S),
            ('x_ST',           x_ST),
            ('x_STS',           x_STS),
            ('x_SS',           x_SS),
            ('x_STM',           x_STM),
            ('x_T',          x_T),
            ('x_TS',           x_TS),
            ('x_TST',           x_TST),
            ('x_TT',           x_TT),
            ('x_TM',           x_TM)
            ))
   
        return visible


class _loss_D(nn.Module):
    def __init__(self, gan_objective, disc_S, disc_T, lambda_seg=1, lambda_disc=1, lambda_id=1, lambda_anatomy=1, lambda_contraste=1,
                 scaler=None, debug_ac_gan=False):
        super(_loss_D, self).__init__()
        self._gan               = gan_objective
        self.scaler             = scaler
        self.lambda_disc        = lambda_disc
        self.debug_ac_gan       = debug_ac_gan
        self.net = {'disc_S'    : disc_S, 'disc_T' : disc_T}  # Separate params.
    
    @autocast_if_needed()
    def forward(self, out_ST, x_T, out_TS, x_S):

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
        
        loss_disc['disc_S'] = self._gan.D(self.net['disc_S'],
                         fake=out_TS,
                         real=x_S,
                         kwargs_real=kwargs_real,
                         kwargs_fake=kwargs_fake,
                         scaler=self.scaler)     
        loss_disc['disc_T'] = self._gan.D(self.net['disc_T'],
                 fake=out_ST,
                 real=x_T,
                 kwargs_real=kwargs_real,
                 kwargs_fake=kwargs_fake,
                 scaler=self.scaler)  

        return loss_disc


class _loss_G(nn.Module):
    def __init__(self, gan_objective, criterion_NCE, criterion_MIND, disc_S, disc_T, lambda_seg=1, lambda_disc=1, lambda_id=1, lambda_anatomy=1, lambda_contraste=1,
                 scaler=None, debug_ac_gan=False):
        super(_loss_G, self).__init__()
        self._gan               = gan_objective
        self.scaler             = scaler
        self.criterion_NCE      = criterion_NCE
        self.criterion_MIND     = criterion_MIND                                       
        self.lambda_disc        = lambda_disc
        self.lambda_id          = lambda_id
        self.lambda_anatomy     = lambda_anatomy
        self.lambda_contraste   = lambda_contraste                                    
        self.debug_ac_gan       = debug_ac_gan
        self.net = {'disc_S'    : disc_S, 'disc_T' : disc_T}  # Separate params.
    
    @autocast_if_needed()
    def forward(self, x_S, x_T, x_ST, x_STS, x_SS, x_TS, x_TST, x_TT):
                                     
        # Generator loss.
        loss_gen = defaultdict(int)
        kwargs_real = None
        kwargs_fake = None                                       
        
        def Cor_CoeLoss(y_pred, y_target):
            x = y_pred
            y = y_target
            x_var = x - torch.mean(x)
            y_var = y - torch.mean(y)
            r_num = torch.sum(x_var * y_var)
            r_den = torch.sqrt(torch.sum(x_var ** 2)) * torch.sqrt(torch.sum(y_var ** 2))
            r = r_num / r_den
            return 1 - r**2    # abslute constrain                                           
      
        loss_gen['T'] = self.lambda_disc*self._gan.G(self.net['disc_T'],
                     fake=x_ST,
                     real=x_T,
                     kwargs_real=kwargs_real,
                     kwargs_fake=kwargs_fake)
        loss_gen['S'] = self.lambda_disc*self._gan.G(self.net['disc_S'],
                     fake=x_TS,
                     real=x_S,
                     kwargs_real=kwargs_real,
                     kwargs_fake=kwargs_fake)           
        
        loss_gen['rec_TT'] = self.lambda_id*mse(x_T, x_TT)  
        loss_gen['rec_SS'] = self.lambda_id*mse(x_S, x_SS) 
        
        loss_gen['rec_STS'] = self.lambda_contraste*mse(x_S, x_STS)  
        loss_gen['rec_TST'] = self.lambda_contraste*mse(x_T, x_TST)       
                                                   
        loss_gen['MIND'] = 0.#self.lambda_anatomy*self.criterion_MIND(x_S, x_ST)+self.lambda_anatomy*self.criterion_MIND(x_T, x_TS)
                                                   
        loss_gen['Corr'] =  self.lambda_anatomy*Cor_CoeLoss(x_S, x_ST)+self.lambda_anatomy*Cor_CoeLoss(x_T, x_TS)
                                                   
        # All generator losses combined.
        loss_G = ( _reduce(loss_gen.values()))
        
        # Compile outputs and return.
        losses = OrderedDict((
            ('l_G',           loss_G),
            ('l_rec',      _reduce([loss_gen['rec_TT'], loss_gen['rec_SS']])),
            ('l_cyc',      _reduce([loss_gen['rec_STS'], loss_gen['rec_TST']])),
            ('l_AC',      _reduce([loss_gen['MIND'], loss_gen['Corr']]))
            ))
        return losses

