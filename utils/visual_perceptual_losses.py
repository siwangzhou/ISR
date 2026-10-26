import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms
import numpy as np
import cv2
from typing import Tuple, Optional

from utils import sam_model_registry, SamAutomaticMaskGenerator


# ---- VGG feature extractor (frozen) ----
class VGG16FeatureExtractor(nn.Module):
    """ Extract features from several VGG16 layers for perceptual loss.
    Uses pretrained vgg16.features; returns list of feature maps. """
    def __init__(self, device='cpu', layer_ids=(3, 8, 15, 22)):
        """ layer_ids: indices of layers in vgg.features to extract (conv layers)
        defaults pick a mix of shallow->deep layers (adapt as needed) """
        super().__init__()
        vgg_pre = models.vgg16(pretrained=True).features.to(device).eval()
        for p in vgg_pre.parameters():
            p.requires_grad = False
        self.vgg_features = vgg_pre
        self.layer_ids = set(layer_ids)
        self.device = device

    def forward(self, x: torch.Tensor):
        """
        x: Bx3xHxW (float in [0,1] and already normalized to ImageNet mean/std if needed)
        returns: list of feature maps corresponding to layer_ids
        """
        out = x
        feats = []
        for i, layer in enumerate(self.vgg_features):
            out = layer(out)
            if i in self.layer_ids:
                feats.append(out)
        return feats

# ---- helper: normalize for VGG (ImageNet) ----
imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])

def normalize_for_vgg(x: torch.Tensor) -> torch.Tensor:
    """ x: Bx3xHxW float tensor in [0,1] returns normalized tensor for VGG input """
    mean = imagenet_mean.to(x.device).view(1,3,1,1)
    std = imagenet_std.to(x.device).view(1,3,1,1)
    return (x - mean) / std


class VisualGuidance:
    """Visual guidance using frequency domain analysis with FFT"""
    def __init__(self):
        pass

    def compute_visual_loss(self, hr_image, sr_image):
        """
        Compute visual loss based on frequency domain analysis
        Args:
            hr_image: High-resolution reference image (B, C, H, W)
            sr_image: Super-resolved image (B, C, H, W)
        Returns: Visual loss value
        """
        # Convert to grayscale for frequency analysis if needed
        if hr_image.shape[1] == 3:
            hr_gray = torch.mean(hr_image, dim=1, keepdim=True)
            sr_gray = torch.mean(sr_image, dim=1, keepdim=True)
        else:
            hr_gray = hr_image
            sr_gray = sr_image
        batch_losses = []
        for b in range(hr_gray.shape[0]):
            # Get single image from batch
            hr_single = hr_gray[b, 0] # (H, W)
            sr_single = sr_gray[b, 0] # (H, W)
            # Apply FFT to transform to frequency domain
            # F(I_HR) and F(I_SR) in equation (10)
            fft_hr = torch.fft.fft2(hr_single)
            fft_sr = torch.fft.fft2(sr_single)
            # Shift zero-frequency component to center
            # I_HR^shift and I_SR^shift
            fft_hr_shift = torch.fft.fftshift(fft_hr)
            fft_sr_shift = torch.fft.fftshift(fft_sr)
            # Compute magnitude spectra with logarithmic compression
            # M_HR and M_SR with log(1 + |I_shift|)
            magnitude_hr = torch.log(1 + torch.abs(fft_hr_shift))
            magnitude_sr = torch.log(1 + torch.abs(fft_sr_shift))
            # Normalize to have zero mean and unit variance
            # M_HR^norm and M_SR^norm with N(·) operation
            magnitude_hr_norm = (magnitude_hr - torch.mean(magnitude_hr)) / (torch.std(magnitude_hr) + 1e-8)
            magnitude_sr_norm = (magnitude_sr - torch.mean(magnitude_sr)) / (torch.std(magnitude_sr) + 1e-8)
            # Compute MSE between normalized magnitude spectra
            batch_loss = F.mse_loss(magnitude_hr_norm, magnitude_sr_norm)
            batch_losses.append(batch_loss)
        # Average across batch
        visual_loss = torch.stack(batch_losses).mean()
        return visual_loss

class PerceptualGuidance:
    def __init__(self, vgg_layer: int = 16, device: str = "cuda"):
        """
        vgg_layer: index within torchvision.models.vgg16(pretrained=True).features to extract.
                   typical conv-layer indices: 3, 8, 15, 22, ...
        device: torch device string or torch.device
        """
        self.device = torch.device(device)
        # load pretrained VGG16 features and freeze
        vgg_full = models.vgg16(pretrained=True).features.to(self.device).eval()
        for p in vgg_full.parameters():
            p.requires_grad = False
        # clip vgg_layer to valid range
        max_idx = len(vgg_full) - 1
        layer_idx = min(max(0, vgg_layer), max_idx)
        # only keep layers up to layer_idx (inclusive)
        self.vgg_features = torch.nn.Sequential(*[vgg_full[i] for i in range(layer_idx + 1)]).to(self.device).eval()
        for p in self.vgg_features.parameters():
            p.requires_grad = False

        # SAM setup (automatic mask generator)
        checkpoint = "weights/sam_vit_h_4b8939.pth"
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        sam.to(self.device)
        sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(sam)

        # ImageNet normalization constants for VGG
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
        self.imagenet_std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)

    # ---------------- helper functions ----------------
    def _to_01(self, img: torch.Tensor) -> torch.Tensor:
        """
        输入（-1，1）
        输出（0，1）
        """
        img01 = (img + 1.0) * 0.5
        return img01

    def _normalize_for_vgg(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Bx3xHxW in [0,1] on device self.device
        returns normalized for VGG
        """
        return (x - self.imagenet_mean) / self.imagenet_std

    def _tensor_to_uint8_rgb(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        Convert a single-image tensor to HxWx3 uint8 RGB numpy for SAM automatic generator.
        Accepts shapes: (3,H,W) or (1,3,H,W) or (B,3,H,W) (will take first).
        Values allowed: [0,1] floats or [0,255].
        Returns HxWx3 uint8 (RGB).
        """
        t = img_tensor.detach().cpu()
        if t.ndim == 4:  # B,C,H,W -> take first
            t = t[0]
        if t.shape[0] == 3:
            arr = t.permute(1,2,0).numpy()  # H,W,C
        elif t.ndim == 3 and t.shape[2] == 3:
            arr = t.numpy()
        else:
            raise ValueError(f"Unsupported tensor shape for conversion: {t.shape}")

        # convert floats to uint8
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            if arr.max() <= 1.5:
                arr = (arr * 255.0).clip(0,255).astype(np.uint8)
            else:
                arr = arr.clip(0,255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        # ensure RGB channels order; user tensors expected as RGB already
        return arr

    def _masks_list_to_coverage(self, masks_list, H: int, W: int, method: str = "mean") -> np.ndarray:
        """
        Convert SamAutomaticMaskGenerator masks list to per-pixel coverage map [0,1].
        method: 'mean' or 'max'
        """
        if len(masks_list) == 0:
            return np.zeros((H, W), dtype=np.float32)
        stack = np.stack([m['segmentation'].astype(np.float32) for m in masks_list], axis=0)  # C x H x W
        if method == "mean":
            cov = stack.mean(axis=0)
        else:
            cov = stack.max(axis=0)
        return np.clip(cov, 0.0, 1.0).astype(np.float32)

    # ---------------- main API ----------------
    def compute_perceptual_loss(self, hr_image: torch.Tensor, sr_image: torch.Tensor) -> torch.Tensor:
        """
        hr_image, sr_image: torch.Tensor of shape (B, C, H, W), C==3, values in [0,1] or [0,255].
        Returns (L_perceptual, L_vgg, L_seg) as torch.Tensors on self.device.
        - L_vgg is differentiable (on device)
        - L_seg computed via SamAutomaticMaskGenerator is non-differentiable (converted to torch tensor)
        - L_perceptual = L_vgg + L_seg
        """
        assert hr_image.ndim == 4 and sr_image.ndim == 4, "Inputs must be (B,C,H,W)"
        B = hr_image.shape[0]
        assert B == sr_image.shape[0], "HR and SR batch sizes must match"

        # move to device and to [0,1] range
        hr = hr_image.to(self.device)
        sr = sr_image.to(self.device)
        hr01 = self._to_01(hr)
        sr01 = self._to_01(sr)

        # ---- compute LVGG ----
        # normalize inputs for VGG
        hr_vgg_in = self._normalize_for_vgg(hr01)
        sr_vgg_in = self._normalize_for_vgg(sr01)
        # pass through truncated VGG features
        feat_hr = self.vgg_features(hr_vgg_in)   # B x CxHxW (feature map at chosen layer)
        feat_sr = self.vgg_features(sr_vgg_in)
        # compute MSE over feature maps
        L_vgg = F.mse_loss(feat_hr, feat_sr)

        # ---- compute Lseg (non-differentiable using automatic generator) ----
        # loop over batch because mask_generator.generate expects single numpy image
        lseg_vals = []
        for i in range(B):
            hr_np = self._tensor_to_uint8_rgb(hr01[i:i+1])  # HxWx3 uint8
            sr_np = self._tensor_to_uint8_rgb(sr01[i:i+1])
            H_hr, W_hr = hr_np.shape[:2]
            # generate masks (numpy-level)
            masks_hr = self.mask_generator.generate(hr_np)
            masks_sr = self.mask_generator.generate(sr_np)
            # coverage maps
            SHR = self._masks_list_to_coverage(masks_hr, H_hr, W_hr, method="mean")
            H_sr, W_sr = sr_np.shape[:2]
            SSR_small = self._masks_list_to_coverage(masks_sr, H_sr, W_sr, method="mean")
            # resize SSR_small to HR size
            SSR = cv2.resize(SSR_small, (W_hr, H_hr), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            # per-sample mse
            mse_val = float(np.mean((SHR - SSR) ** 2))
            lseg_vals.append(mse_val)

        # average over batch and convert to torch tensor on device
        lseg_mean = float(np.mean(lseg_vals)) if len(lseg_vals) > 0 else 0.0
        L_seg = torch.tensor(lseg_mean, dtype=torch.float32, device=self.device)

        # total perceptual loss
        L_perceptual = L_vgg + L_seg

        return L_perceptual