"""Audio pre-processing utilities for speech enhancement.

Core functions: STFT computation, normalization, spectrogram augmentation,
and the AudioPreProcessing class for converting wav → model inputs/targets.
"""

import warnings
import math
import numpy as np
import torch
import torchaudio
import librosa
import scipy.signal as sps

warnings.filterwarnings("ignore", category=UserWarning)


def normalize(x):
    """Normalize signal to peak amplitude."""
    if np.max(np.abs(x)) != 0:
        return x / 1.1 / np.max(np.abs(x))
    return x


def normalize_signal_slices(x, space=15, fs=8000):
    """Normalize signal in overlapping slices."""
    norm_x = x.copy()
    for inx in range(math.ceil(len(norm_x) / (space * fs))):
        end_i = len(norm_x) if inx == math.ceil(len(norm_x) / (space * fs)) - 1 else inx * (space * fs) + (space * fs)
        seg = norm_x[inx * (space * fs):end_i]
        if np.max(np.abs(seg)) != 0:
            norm_x[inx * (space * fs):end_i] = seg / 1.1 / np.max(np.abs(seg))
        else:
            norm_x[inx * (space * fs):end_i] = 0
    return norm_x


def spec_aug(args, spec):
    """Apply SpecAugment-style masking to a spectrogram."""
    spec_length = spec.size()[-1]
    I = spec_length // args.spec_aug_time_jump
    window_size = args.window_size if args.fs == 8000 else 2 * args.window_size
    K_full = int(args.overlap * window_size)

    for i in range(I):
        x_temp = torch.randint(0, args.spec_aug_time_jump - args.N_max, size=(1,))
        y_index = torch.randint(0, K_full - args.K_max, size=(1,))
        patch_len = torch.randint(args.N_min, args.N_max, size=(1,))
        patch_height = torch.randint(args.K_min, args.K_max, size=(1,))
        x_index = i * args.spec_aug_time_jump + x_temp
        spec[y_index:y_index + patch_height, x_index:x_index + patch_len] = 0
    return spec


def two_chan_spec_aug(args, first_spec, second_spec):
    """Apply synchronized SpecAugment to both RI channels."""
    spec_length = first_spec.size()[-1]
    I = spec_length // args.spec_aug_time_jump
    window_size = args.window_size if args.fs == 8000 else 2 * args.window_size
    K_full = int(args.overlap * window_size)

    for i in range(I):
        x_temp = torch.randint(0, args.spec_aug_time_jump - args.N_max, size=(1,))
        y_index = torch.randint(0, K_full - args.K_max, size=(1,))
        patch_len = torch.randint(args.N_min, args.N_max, size=(1,))
        patch_height = torch.randint(args.K_min, args.K_max, size=(1,))
        x_index = i * args.spec_aug_time_jump + x_temp
        first_spec[y_index:y_index + patch_height, x_index:x_index + patch_len] = 0
        second_spec[y_index:y_index + patch_height, x_index:x_index + patch_len] = 0
    return first_spec, second_spec


def choose_rand_frames(X, num_of_frames, start=None):
    """Randomly crop frames from a spectrogram."""
    if X.ndim < 3:
        X = X.unsqueeze(0)
    if X.shape[-1] <= num_of_frames:
        output = torch.cat((X, torch.zeros((X.shape[0], X.shape[1], num_of_frames - X.shape[-1]), dtype=X.dtype, device=X.device)), dim=-1)
        start = 0
    else:
        if start is None:
            start = torch.randint(0, X.shape[-1] - num_of_frames, (1,))
        output = X[..., start:start + num_of_frames]
    return output.squeeze(), start


class AudioPreProcessing:
    """Converts raw waveform mixtures into model input/target pairs.

    Handles STFT computation, RI channel extraction, normalization,
    and target creation for various configurations (denoising, dereverberation).
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def preprocess(self, mixed, cut_length=2, normalize_noisy=False, cut_signal=True):
        """Process a raw multi-channel waveform into STFT-domain tensors.

        Args:
            mixed: numpy array [channels, samples]
                   channels: [noisy, noise, clean] or [noisy, clean] for reverb
            cut_length: Duration in seconds to crop for training
            normalize_noisy: Whether to normalize the noisy signal
            cut_signal: Whether to randomly crop the signal

        Returns:
            input_map: Model input [C_in, F, T]
            target: Model target [C_out, F, T]
            Z_stft_complex: Noisy STFT (complex)
            S_stft_complex: Clean STFT (complex)
            V_out: Noise output (or None for reverb-only)
            mixed: Processed waveform array
        """
        cfg = self.cfg
        window_size = cfg.window_size if cfg.fs == 8000 else 2 * cfg.window_size
        duration_sec = cfg.fs * cut_length

        noisy = mixed[0, :]
        is_reverb = (cfg.dereverb or cfg.use_pra or cfg.from_reverb)

        if is_reverb:
            if cfg.with_noise and mixed.shape[0] > 2:
                noise = mixed[1, :]
                clean = mixed[2, :]
            else:
                clean = mixed[1, :]
        else:
            noise = mixed[1, :]
            clean = mixed[2, :]

        # Random crop for training
        if cfg.mode == 'train':
            if cut_signal and mixed.shape[-1] > duration_sec:
                start_point = torch.randint(0, mixed.shape[-1] - duration_sec, (1,)).item()
                mixed = mixed[..., start_point:start_point + duration_sec]
            elif mixed.shape[-1] < duration_sec and not cfg.return_full:
                mixed = np.concatenate((mixed, np.zeros((mixed.shape[-2], duration_sec - mixed.shape[-1]))), axis=1)

            noisy = mixed[0, :]
            if is_reverb:
                if cfg.with_noise and mixed.shape[0] > 2:
                    noise = mixed[1, :]
                    clean = mixed[2, :]
                    mixed = np.stack((mixed[0, :], mixed[2, :]))
                else:
                    clean = mixed[1, :]
            else:
                noise = mixed[1, :]
                clean = mixed[2, :]

            if normalize_noisy:
                noisy = normalize(noisy)
                mixed[0, :] = noisy

        # Compute STFTs
        S_stft_complex = torch.stft(torch.from_numpy(clean), window_size, int(window_size * cfg.overlap),
                                     window=torch.hamming_window(window_size), return_complex=True)
        Z_stft_complex = torch.stft(torch.from_numpy(noisy), window_size, int(window_size * cfg.overlap),
                                     window=torch.hamming_window(window_size), return_complex=True)
        if not is_reverb:
            V_stft_complex = torch.stft(torch.from_numpy(noise), window_size, int(window_size * cfg.overlap),
                                         window=torch.hamming_window(window_size), return_complex=True)

        Z_stft_complex = torch.tensor(Z_stft_complex, dtype=torch.complex128)
        S_stft_complex = torch.tensor(S_stft_complex, dtype=torch.complex128)

        V_out = None

        # Build target
        if cfg.target == 'stft_RI':
            real_target = S_stft_complex.real
            imag_target = S_stft_complex.imag
            target = torch.stack((real_target, imag_target), dim=0)

            if not is_reverb:
                real_noise = V_stft_complex.real
                imag_noise = V_stft_complex.imag
                V_out = torch.stack((real_noise, imag_noise), dim=0)

            if cfg.pred_spec or cfg.spec_mse:
                S_stft = torch.abs(S_stft_complex)
                targets_clean_log_spec = torch.log10(S_stft + cfg.eps)

        # Build input
        if cfg.input_map == 'stft_RI':
            real_input = Z_stft_complex.real
            imag_input = Z_stft_complex.imag
            if cfg.spec_augment and cfg.mode == 'train':
                real_input, imag_input = two_chan_spec_aug(cfg, real_input, imag_input)
            input_map = torch.stack((real_input, imag_input), dim=0)

        if is_reverb:
            V_out = None

        if (cfg.pred_spec or cfg.spec_mse) and cfg.mode != 'test':
            return input_map, target, targets_clean_log_spec, Z_stft_complex, S_stft_complex, V_out, mixed
        else:
            return input_map, target, Z_stft_complex, S_stft_complex, V_out, mixed


def sort_by_prob(element):
    """Sort key for samples by their pseudo-label probability."""
    return float('.'.join(element.split('_')[-1].split('.')[:-1]))


def sort_by_pesq(element):
    return float(element.split('_')[8])


def sort_by_stoi(element):
    return float(element.split('_')[10])


def sort_by_sisdr(element):
    return float('.'.join(element.split('_')[12].split('.')[:-1]))
