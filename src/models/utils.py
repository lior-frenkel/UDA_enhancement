"""Utility functions for speech enhancement domain adaptation.

Core utilities: SI-SDR loss, weight initialization, spectrogram conversion, etc.
"""

import os
import glob
import math

import numpy as np
import matplotlib
import torch
import torch.nn as nn
import torchaudio

matplotlib.use("Agg")
import matplotlib.pylab as plt


def init_weights_he(m):
    """Kaiming (He) uniform initialization for Conv2d layers."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')


def si_sdr_torchaudio_calc(estimate, reference):
    """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).

    Args:
        estimate: Enhanced signal [B, T]
        reference: Clean reference signal [B, T]

    Returns:
        SI-SDR value (scalar tensor)
    """
    # Ensure same length
    min_len = min(estimate.shape[-1], reference.shape[-1])
    estimate = estimate[..., :min_len]
    reference = reference[..., :min_len]

    # Zero-mean
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    reference = reference - torch.mean(reference, dim=-1, keepdim=True)

    dot = torch.sum(estimate * reference, dim=-1, keepdim=True)
    s_ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True) + 1e-8
    proj = dot * reference / s_ref_energy

    noise = estimate - proj
    si_sdr = 10 * torch.log10(torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + 1e-8) + 1e-8)

    return si_sdr.mean()


def si_sdri_loss(cfg, enhanced_spec, clean_raw, noisy_raw, noisy_angle, device):
    """Compute SI-SDR improvement (SI-SDRi) loss.

    Converts the enhanced RI spectrogram back to waveform via iSTFT,
    then computes SI-SDR improvement over the noisy input.

    Args:
        cfg: Configuration object with fs, window_size, overlap
        enhanced_spec: Enhanced complex spectrogram [B, F, T]
        clean_raw: Clean waveform [B, T]
        noisy_raw: Noisy waveform [B, T]
        noisy_angle: Phase of noisy STFT (unused, kept for compatibility)
        device: Torch device

    Returns:
        loss: Negative SI-SDRi (for minimization)
        si_sdri: SI-SDRi value
    """
    if cfg.fs == 8000:
        window_size = cfg.window_size
    elif cfg.fs == 16000:
        window_size = 2 * cfg.window_size

    enhanced = torch.istft(
        enhanced_spec.squeeze(),
        n_fft=window_size,
        window=torch.hamming_window(window_size).to(device),
        hop_length=int(window_size * cfg.overlap),
        length=clean_raw.shape[-1]
    )

    si_sdr_enhanced = si_sdr_torchaudio_calc(enhanced, clean_raw)
    si_sdr_noisy = si_sdr_torchaudio_calc(noisy_raw, clean_raw)
    si_sdri = si_sdr_enhanced - si_sdr_noisy

    return -si_sdri, si_sdri


def RI_to_log_spec(pred):
    """Convert Real-Imaginary STFT prediction to log-magnitude spectrogram.

    Args:
        pred: Tensor [B, 2, F, T] with real and imaginary channels

    Returns:
        Log-magnitude spectrogram [B, 1, F, T]
    """
    pred_real = pred[:, 0, :, :].float()
    pred_imag = pred[:, 1, :, :].float()
    pred_spec = torch.sqrt(pred_real ** 2 + pred_imag ** 2)
    pred_log_spec = torch.log10(pred_spec + 1e-6)
    return pred_log_spec.unsqueeze(1)


def create_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_figs(img, name='spec', path='.', title='Spectrogram', squeeze_img=True):
    """Save a spectrogram figure."""
    plt.figure(figsize=(10, 4))
    if squeeze_img and img.dim() > 2:
        img = img.squeeze()
    plt.imshow(img.detach().cpu().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{path}/{name}.png')
    plt.close()


def find_existing_ckpt():
    """Find the most recent 'last.ckpt' in the current working directory tree."""
    folder_name = os.getcwd()
    ckpts = glob.glob(folder_name + "/**/**/last.ckpt", recursive=True)
    if len(ckpts) == 0:
        print("No checkpoints found.")
        return None
    times = [os.path.getctime(ckpt) for ckpt in ckpts]
    max_time = max(times)
    latest_index = times.index(max_time)
    print(f"Resuming from checkpoint: {ckpts[latest_index]}")
    return ckpts[latest_index]
