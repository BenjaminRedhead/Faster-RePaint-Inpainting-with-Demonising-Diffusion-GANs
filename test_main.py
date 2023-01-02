# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import os
import argparse
import torch as th
import numpy as np
import torch.nn.functional as F
import time
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util
from score_sde.models.ncsnpp_generator_adagn import NCSNpp

# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402

def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


def main(conf: conf_mgt.Default_Conf, args):

    print("Start", conf['name'])

    device = 'cuda:1'


    _,  diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )


    model = NCSNpp(args).to(device)
    ckpt = th.load('./model/netG_{}.pth'.format(args.epoch_id), map_location=device)

    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    model.load_state_dict(ckpt)
    model.eval()

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
        )

        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None



    def extract(input, t, shape):
        out = th.gather(input, 0, t)
        reshape = [shape[0]] + [1] * (len(shape) - 1)
        out = out.reshape(*reshape)

        return out

    def var_func_vp(t, beta_min, beta_max):
        log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
        var = 1. - torch.exp(2. * log_mean_coeff)
        return var

    def var_func_geometric(t, beta_min, beta_max):
        return beta_min * ((beta_max / beta_min) ** t)

    def sample_posterior(coefficients, x_0,x_t, t):
        
        def q_posterior(x_0, x_t, t):
            mean = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
            )
            var = extract(coefficients.posterior_variance, t, x_t.shape)
            log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
            return mean, var, log_var_clipped
        
    
        def p_sample(x_0, x_t, t):
            mean, _, log_var = q_posterior(x_0, x_t, t)
            
            noise = th.randn_like(x_t)
            
            nonzero_mask = (1 - (t == 0).type(th.float32))

            return mean + nonzero_mask[:,None,None,None] * th.exp(0.5 * log_var) * noise
                
        sample_x_pos = p_sample(x_0, x_t, t)
        
        return sample_x_pos
    
    to_range_0_1 = lambda x: (x + 1.) / 2.


    def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
        x = x_init
        with th.no_grad():
            for i in reversed(range(n_time)):
                t = th.full((x.size(0),), i, dtype=th.int64).to(x.device)
                
                t_time = t
                latent_z = th.randn(x.size(0), opt.nz, device=x.device)#.to(x.device)
                x_0 = generator(x, t_time, latent_z)
                x_new = sample_posterior(coefficients, x_0, x, t)
                x = x_new.detach()

    def get_time_schedule(args, device):
        n_timestep = args.num_timesteps
        eps_small = 1e-3
        t = np.arange(0, n_timestep + 1, dtype=np.float64)
        t = t / n_timestep
        t = th.from_numpy(t) * (1. - eps_small)  + eps_small
        return t.to(device)

    def get_sigma_schedule(args, device):
        n_timestep = args.num_timesteps
        beta_min = args.beta_min
        beta_max = args.beta_max
        eps_small = 1e-3
    
        t = np.arange(0, n_timestep + 1, dtype=np.float64)
        t = t / n_timestep
        t = th.from_numpy(t) * (1. - eps_small) + eps_small
        
        if args.use_geometric:
            var = var_func_geometric(t, beta_min, beta_max)
        else:
            var = var_func_vp(t, beta_min, beta_max)
        alpha_bars = 1.0 - var
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        
        first = th.tensor(1e-8)
        betas = th.cat((first[None], betas)).to(device)
        betas = betas.type(th.float32)
        sigmas = betas**0.5
        a_s = th.sqrt(1-betas)
        return sigmas, a_s, betas

    class Posterior_Coefficients():
        def __init__(self, args, device):
            
            _, _, self.betas = get_sigma_schedule(args, device=device)
            
            #we don't need the zeros
            self.betas = self.betas.type(th.float32)[1:]
            
            self.alphas = 1 - self.betas
            self.alphas_cumprod = th.cumprod(self.alphas, 0)
            self.alphas_cumprod_prev = th.cat(
                                        (th.tensor([1.], dtype=th.float32,device=device), self.alphas_cumprod[:-1]), 0
                                            )               
            self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
            
            self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
            self.sqrt_recip_alphas_cumprod = th.rsqrt(self.alphas_cumprod)
            self.sqrt_recipm1_alphas_cumprod = th.sqrt(1 / self.alphas_cumprod - 1)
            
            self.posterior_mean_coef1 = (self.betas * th.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
            self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * th.sqrt(self.alphas) / (1 - self.alphas_cumprod))
            
            self.posterior_log_variance_clipped = th.log(self.posterior_variance.clamp(min=1e-20))
        
    def model_fn(x, t, y=None, gt=None, **kwargs):
        T = get_time_schedule(args, device)
        pos_coeff = Posterior_Coefficients(args, device)
        x_t_1 = th.randn(args.batch_size, args.num_channels,args.image_size, args.image_size).to(device)
        fake_sample = sample_from_model(pos_coeff, model, args.num_timesteps, x_t_1,T,  args)
        fake_sample = to_range_0_1(fake_sample)
        return fake_sample
    

    print("sampling...")
    all_images = []

    dset = 'eval'

    eval_name = conf.get_default_eval_name()

    dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    for batch in iter(dl):

        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}

        model_kwargs["gt"] = batch['GT']

        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=device
            )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )


        result = sample_fn(
            model_fn,
            (batch_size, 3, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf
        )
        srs = toU8(result['sample'])
        gts = toU8(result['gt'])
        lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                   th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))

        gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))

        conf.eval_imswrite(
            srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
            img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)

    print("sampling complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int,default=1000)
    parser.add_argument('--num_channels', type=int, default=3,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')
    
    
    parser.add_argument('--num_channels_dae', type=int, default=128,
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    
    #geenrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    
    
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=200, help='sample generating batch size')
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    main(conf_arg,args)
