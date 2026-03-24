"""Inference tool for speech enhancement.

Provides a clean API for loading a trained model and enhancing audio files.

Usage:
    from inference.enhance import SpeechEnhancer
    enhancer = SpeechEnhancer("path/to/checkpoint.ckpt", device="cuda")
    enhanced = enhancer.enhance_file("noisy.wav", output_path="enhanced.wav")
"""

import os
import math
import torch
import torchaudio
import torchaudio.transforms as T

from src.models.model_parts import DoubleConv, Down, Up, OutConv


class UNetInference(torch.nn.Module):
    """Minimal UNet for inference (no config dependency).

    Hardcoded architecture matching the training UNet with default settings:
        - 2 input channels (RI)
        - 4 output channels (enhanced RI + noise RI)
        - 64 base hidden size
        - 4 encoder/decoder levels
    """

    def __init__(self):
        super().__init__()
        hidden = 64
        factor = 2

        self.inc = DoubleConv(2, hidden)
        self.down1 = Down(hidden, hidden * 2)
        self.down2 = Down(hidden * 2, hidden * 4)
        self.down3 = Down(hidden * 4, hidden * 8)
        self.down4 = Down(hidden * 8, hidden * 16 // factor)

        self.up1 = Up(hidden * 16, hidden * 8 // factor)
        self.up2 = Up(hidden * 8, hidden * 4 // factor)
        self.up3 = Up(hidden * 4, hidden * 2 // factor)
        self.up4 = Up(hidden * 2, hidden)

        self.outputs = OutConv(hidden, 4)  # clean RI + noise RI

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outputs(x)


def normalize_signal_slices(x, fs=16000, space=15):
    """Normalize signal in overlapping slices (torch version)."""
    norm_x = x.clone()
    for inx in range(math.ceil(len(norm_x) / (space * fs))):
        end_i = len(norm_x) if inx == math.ceil(len(norm_x) / (space * fs)) - 1 else inx * (space * fs) + (space * fs)
        seg = norm_x[inx * (space * fs):end_i]
        mx = torch.max(torch.abs(seg))
        if mx != 0:
            norm_x[inx * (space * fs):end_i] = seg / 1.1 / mx
    return norm_x


class SpeechEnhancer:
    """High-level speech enhancement inference tool.

    Args:
        checkpoint_path: Path to a .ckpt file (PL checkpoint with 'student' or 'model' keys)
        device: 'cuda' or 'cpu'
        fs: Sample rate (8000 or 16000)
    """

    def __init__(self, checkpoint_path, device='cuda', fs=16000):
        self.device = device
        self.fs = fs
        self.window_size = 512 if fs == 16000 else 256
        self.overlap = 0.5

        self.model = UNetInference()
        self._load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

    def _load_checkpoint(self, ckpt_path):
        """Load model weights from PL checkpoint."""
        state = torch.load(ckpt_path, map_location=self.device)['state_dict']

        # Try loading from 'student' key (Pl_UNet2) or 'model' key (Pl_UNet_enhance)
        for prefix in ('student', 'model'):
            filtered = {'.'.join(k.split('.')[1:]): v for k, v in state.items() if k.startswith(prefix + '.')}
            if filtered:
                self.model.load_state_dict(filtered)
                return

        raise ValueError(f"Could not find 'student' or 'model' keys in checkpoint {ckpt_path}")

    def preprocess(self, noisy, fs=None):
        """Convert waveform to normalized RI STFT input.

        Args:
            noisy: Tensor [1, T] or [T]
            fs: Sample rate (if different from init, will resample)

        Returns:
            input_map: Tensor [1, 2, F, T_frames] on device
        """
        if noisy.dim() == 1:
            noisy = noisy.unsqueeze(0)
        if fs is not None and fs != self.fs:
            resampler = T.Resample(fs, self.fs)
            noisy = resampler(noisy)

        # Normalize
        noisy = noisy / 1.1 / torch.max(torch.abs(noisy))
        noisy = noisy / torch.std(noisy)

        Z_stft = torch.stft(noisy, self.window_size, int(self.window_size * self.overlap),
                             window=torch.hamming_window(self.window_size),
                             return_complex=True).to(self.device)
        Z_stft = Z_stft.to(dtype=torch.complex128)

        Z_real = Z_stft.real
        Z_imag = Z_stft.imag
        input_map = torch.cat((Z_real, Z_imag), dim=0).unsqueeze(0).float()

        # Zero low frequencies
        input_map[..., 0:3, :] = input_map[..., 0:3, :] * 0.001

        # Normalize by max
        mx = torch.max(torch.abs(input_map))
        if mx > 0:
            input_map = input_map / mx

        return input_map.to(self.device)

    def postprocess(self, enhanced_spec, original_frames, original_samples):
        """Convert enhanced RI STFT back to waveform.

        Args:
            enhanced_spec: Model output [1, 4, F, T]
            original_frames: Number of frames in input STFT
            original_samples: Number of samples in original waveform

        Returns:
            enhanced: Tensor [1, T]
            noise: Tensor [1, T]
        """
        enh_real = enhanced_spec[:, 0, :, :]
        enh_imag = enhanced_spec[:, 1, :, :]
        enh_complex = enh_real + 1j * enh_imag

        noise_real = enhanced_spec[:, 2, :, :]
        noise_imag = enhanced_spec[:, 3, :, :]
        noise_complex = noise_real + 1j * noise_imag

        # Trim to original length
        enh_complex = enh_complex[..., :original_frames]

        enhanced = torch.istft(enh_complex.squeeze(), n_fft=self.window_size,
                               window=torch.hamming_window(self.window_size).to(self.device),
                               hop_length=int(self.window_size * self.overlap),
                               length=original_samples)
        noise = torch.istft(noise_complex.squeeze()[..., :original_frames], n_fft=self.window_size,
                            window=torch.hamming_window(self.window_size).to(self.device),
                            hop_length=int(self.window_size * self.overlap),
                            length=original_samples)

        return enhanced.unsqueeze(0), noise.unsqueeze(0)

    @torch.no_grad()
    def enhance(self, waveform, fs=None):
        """Enhance a waveform tensor.

        Args:
            waveform: Tensor [1, T] or [T]
            fs: Sample rate (optional, will resample if needed)

        Returns:
            enhanced: Enhanced waveform tensor [1, T]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        original_samples = waveform.shape[-1]
        input_map = self.preprocess(waveform, fs)
        original_frames = input_map.shape[-1]

        output = self.model(input_map)
        enhanced, _ = self.postprocess(output, original_frames, original_samples)

        return enhanced.cpu()

    @torch.no_grad()
    def enhance_file(self, input_path, output_path=None):
        """Enhance an audio file.

        Args:
            input_path: Path to input WAV file
            output_path: Path to save enhanced WAV (optional)

        Returns:
            enhanced: Enhanced waveform tensor [1, T]
        """
        waveform, fs = torchaudio.load(input_path)
        enhanced = self.enhance(waveform, fs)

        if output_path:
            torchaudio.save(output_path, enhanced, self.fs)

        return enhanced
