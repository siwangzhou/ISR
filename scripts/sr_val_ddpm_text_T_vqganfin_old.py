"""make variations of input image"""

import argparse, os
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # 向上一级
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ldm.util import instantiate_from_config
import math
import copy
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization

import time


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        section_counts = [int(x) for x in section_counts.split(",")]  # [250,]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image",
        default="inputs/user_upload",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/user_upload",
    )
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=1000,
        help="number of ddpm sampling steps",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,      # 2
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stableSRNew/v2-finetune_text_T_512.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--vqgan_ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/epoch=000011.ckpt",
        help="path to checkpoint of VQGAN model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="input size",
    )
    parser.add_argument(
        "--dec_w",
        type=float,
        default=0.5,
        help="weight for combining VQGAN and Diffusion",
    )
    parser.add_argument(
        "--colorfix_type",
        type=str,
        default="nofix",
        help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
    )

    opt = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print('>>>>>>>>>>color correction>>>>>>>>>>>')
    if opt.colorfix_type == 'adain':
        print('Use adain color correction')
    elif opt.colorfix_type == 'wavelet':
        print('Use wavelet color correction')
    else:
        print('No color correction')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
    vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
    vq_model = vq_model.to(device)
    vq_model.decoder.fusion_w = opt.dec_w

    seed_everything(opt.seed)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(opt.input_size),
        torchvision.transforms.CenterCrop(opt.input_size),
    ])

    transform_LR = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(128),
        # torchvision.transforms.CenterCrop(128),
    ])

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.to(device)
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    img_list_ori = os.listdir(opt.init_img)
    img_list = copy.deepcopy(img_list_ori)
    init_image_list = []

    init_image_LR_list = []

    for item in img_list_ori:
        if os.path.exists(os.path.join(outpath, item)):
            img_list.remove(item)
            continue
        cur_image = load_img(os.path.join(opt.init_img, item))

        cur_image_LR = transform_LR(cur_image)
        cur_image_LR = cur_image_LR.clamp(-1, 1)
        init_image_LR_list.append(cur_image_LR)

        cur_image = transform(cur_image)
        cur_image = cur_image.clamp(-1, 1)
        init_image_list.append(cur_image)
    init_image_list = torch.cat(init_image_list, dim=0)
    niters = math.ceil(init_image_list.size(0) / batch_size)
    init_image_list = init_image_list.chunk(niters)

    init_image_LR_list = torch.cat(init_image_LR_list, dim=0)
    niters_LR = math.ceil(init_image_LR_list.size(0) / batch_size)
    init_image_LR_list = init_image_LR_list.chunk(niters_LR)

    model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
                            linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
    model.num_timesteps = 1000

    use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps]))
    last_alpha_cumprod = 1.0
    new_betas = []
    timestep_map = []
    for i, alpha_cumprod in enumerate(model.alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
            timestep_map.append(i)
    new_betas = [beta.data.cpu().numpy() for beta in new_betas]
    model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
    model.num_timesteps = 1000
    model.ori_timesteps = list(use_timesteps)
    model.ori_timesteps.sort()
    model = model.to(device)
    param_list = []
    untrain_paramlist = []
    name_list = []
    for k, v in model.named_parameters():
        if 'spade' in k or 'structcond_stage_model' in k:
            param_list.append(v)
        else:
            name_list.append(k)
            untrain_paramlist.append(v)
    trainable_params = sum(p.numel() for p in param_list)
    untrainable_params = sum(p.numel() for p in untrain_paramlist)
    print(name_list)
    print(trainable_params)
    print(untrainable_params)
    param_list = []
    untrain_paramlist = []
    for k, v in vq_model.named_parameters():
        if 'fusion_layer' in k:
            param_list.append(v)
        elif 'loss' not in k:
            untrain_paramlist.append(v)
    trainable_params += sum(p.numel() for p in param_list)
    print(trainable_params)
    print(untrainable_params)
    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    # 添加代码
    total_images = sum(x.size(0) for x in init_image_list)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for n in trange(niters, desc="Sampling"):
                    init_image = init_image_list[n].to(device)

                    init_image_LR = init_image_LR_list[n].to(device)

                    init_latent_generator, enc_fea_lq = vq_model.encode(init_image)
                    init_latent = model.get_first_stage_encoding(init_latent_generator)
                    text_init = [''] * init_image.size(0)
                    semantic_c = model.cond_stage_model(text_init)
                    samples = model.sample(model=model, y_pixel=init_image_LR, vq_model=vq_model, enc_fea_lq=enc_fea_lq, cond=semantic_c,  # 文本/语义条件
                                              struct_cond=init_latent,
                                              batch_size=init_image.size(0),
                                              timesteps=opt.ddpm_steps,
                                              time_replace=opt.ddpm_steps,
                                              x_T=None,
                                              return_intermediates=False     # 修改代码True->False
                                              )
                    x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)

                    if opt.colorfix_type == 'adain':
                        x_samples = adaptive_instance_normalization(x_samples, init_image)
                    elif opt.colorfix_type == 'wavelet':
                        x_samples = wavelet_reconstruction(x_samples, init_image)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    for i in range(init_image.size(0)):
                        img_name = img_list.pop(0)
                        basename = os.path.splitext(os.path.basename(img_name))[0]
                        x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(outpath, basename + '.png'))

    # 添加代码
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    print(f"Total inference time: {elapsed:.4f}s")
    print(f"Average time per image: {elapsed / total_images:.4f}s")
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"Peak GPU memory: {peak_memory:.2f} GB")

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()

    """快速测试
    python scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt weights/stablesr_000117.ckpt --vqgan_ckpt weights/vqgan_cfw_00011.ckpt --init-img test_img --outdir experiment/test_img --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain
    
    RealSRVal_crop128:
    python scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt weights/stablesr_000117.ckpt --vqgan_ckpt weights/vqgan_cfw_00011.ckpt --init-img test_img/RealSRVal_crop128/test_LR --outdir experiment/RealSRVal_v1 --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain
    
    python scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt weights/stablesr_000117.ckpt --vqgan_ckpt weights/vqgan_cfw_00011.ckpt --init-img test_img/RealSRVal_crop128/test_LR --outdir experiment/RealSRVal_test --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain
    A_baseline：
    python scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt weights/stablesr_000117.ckpt --vqgan_ckpt weights/vqgan_cfw_00011.ckpt --init-img test_img/RealSRVal_crop128/test_LR_A --outdir experiment/RealSRVal_A_v1 --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain
    
    training:
    python scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt logs/2025-11-23T15-36-44_experiments_train/checkpoints/epoch=000006.ckpt --vqgan_ckpt weights/vqgan_cfw_00011.ckpt --init-img test_img/RealSRVal_crop128/test_LR_A --outdir experiment/train_RealSRVal_A_v1 --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain
    
    # autoDL
    python scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt weights/epoch=000003-v1.ckpt --vqgan_ckpt weights/vqgan_cfw_00011.ckpt --init-img test_img/RealSRVal_crop128/test_LR_A --outdir experiment/test --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain 
    """
