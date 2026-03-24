"""Evaluate a trained speech enhancement model.

Computes PESQ, STOI, and SI-SDR metrics on test data by SNR level.

Usage:
    # Evaluate supervised LibriSpeech model on LibriSpeech test:
    python scripts/evaluate.py \\
        data_type=librispeech data_type_test=librispeech \\
        fs=16000 version=0

    # Evaluate DANN model (LibriSpeech->DNS) on DNS test:
    python scripts/evaluate.py \\
        data_type=dns data_type_test=dns \\
        use_grl=True bce_percent=0.05 pretrained=True \\
        source_domain=librispeech fs=16000 version=0
"""

import sys
import os
import math
import json
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchaudio

from pesq import pesq
from pystoi import stoi

import hydra
from omegaconf import DictConfig

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.models.model_def import Pl_UNet_enhance, Pl_UNet2, Pl_UNet_classifier
from src.models.utils import si_sdr_torchaudio_calc, find_existing_ckpt, create_dir
from src.data.dataloader import BasicDataset
from src.data.data_utils import AudioPreProcessing, normalize_signal_slices


def load_model(cfg, ckpt_path, device):
    """Load a trained PL model for evaluation."""
    if cfg.unsupervised:
        model = Pl_UNet2.load_from_checkpoint(ckpt_path, cfg=cfg, map_location=device)
        # Use student model for inference
        model = model.student
    elif cfg.use_grl:
        model = Pl_UNet_enhance.load_from_checkpoint(ckpt_path, cfg=cfg, map_location=device)
        model = model.model
    else:
        model = Pl_UNet_enhance.load_from_checkpoint(ckpt_path, cfg=cfg, map_location=device)
        model = model.model

    model = model.to(device).eval()
    return model


def preprocess_input(cfg, input_map, device):
    """Preprocess test input: extract RI channels, normalize."""
    input_map = input_map.to(device)

    if cfg.input_map != 'stft_RI':
        input_map = input_map[:, 0, ...].unsqueeze(1)
        return input_map

    if cfg.input_map == 'stft_RI':
        # Extract real and imaginary channels
        input_map = torch.stack(
            (input_map.squeeze()[1, ...], input_map.squeeze()[2, ...]),
            dim=0
        ).unsqueeze(0)

        # Zero low frequencies
        input_map[..., 0:3, :] = input_map[..., 0:3, :] * 0.001

        # Normalize by max
        mx = torch.max(torch.max(
            torch.abs(input_map[:, 0, ...]),
            torch.max(torch.abs(input_map[:, 1, ...]))
        ))
        input_map = input_map / mx

    return input_map


def chunk_inference(model, input_map, cfg, device):
    """Run inference on long signals by chunking into num_frames segments."""
    num_frames = cfg.num_frames

    if input_map.shape[-1] <= num_frames:
        # Pad short signal
        pad_size = num_frames - input_map.shape[-1]
        input_padded = torch.cat(
            (input_map.squeeze(),
             torch.zeros(input_map.shape[-3], input_map.shape[-2], pad_size, device=device)),
            dim=-1
        )
        output = model(input_padded.unsqueeze(0))
        return output

    # Chunk long signal
    n_chunks = math.ceil(input_map.shape[-1] / num_frames)
    chunks = []

    for i in range(n_chunks):
        if i < n_chunks - 1:
            chunk = input_map[..., i * num_frames:(i + 1) * num_frames]
        else:
            # Last chunk: take from end
            chunk = input_map[..., input_map.shape[-1] - num_frames:]
        chunks.append(chunk.squeeze())

    # Batch inference
    batch = torch.stack(chunks, dim=0)
    outputs = model(batch)
    output_list = list(outputs)

    # Trim last chunk overlap
    last_chunk_size = input_map.shape[-1] % num_frames
    if last_chunk_size == 0:
        last_chunk_size = num_frames
    output_list[-1] = output_list[-1][..., num_frames - last_chunk_size:]

    return torch.cat(output_list, dim=-1).unsqueeze(0)


def reconstruct_waveform(cfg, enhanced_spec, input_map, noisy_len, device):
    """Convert enhanced RI STFT back to waveform."""
    window_size = cfg.window_size if cfg.fs == 8000 else 2 * cfg.window_size

    # Extract clean and noise from 4-channel output
    enh_spec = enhanced_spec.squeeze()[..., :input_map.shape[-1]]

    if cfg.reconst_noise and cfg.target == 'stft_RI':
        enh_real = enh_spec[0, :, :]
        enh_imag = enh_spec[1, :, :]
        enh_complex = enh_real + 1j * enh_imag

        noise_real = enh_spec[2, :, :]
        noise_imag = enh_spec[3, :, :]
        noise_complex = noise_real + 1j * noise_imag
    elif cfg.target == 'stft_RI':
        enh_real = enh_spec[0, :, :]
        enh_imag = enh_spec[1, :, :]
        enh_complex = enh_real + 1j * enh_imag
        noise_complex = None
    else:
        raise ValueError(f"Unsupported target: {cfg.target}")

    window = torch.hamming_window(window_size).to(device)
    hop = int(window_size * cfg.overlap)

    enhanced = torch.istft(enh_complex, n_fft=window_size,
                           window=window, hop_length=hop, length=noisy_len)

    enhanced_np = enhanced.detach().cpu().numpy()
    enhanced_np = normalize_signal_slices(enhanced_np, fs=cfg.fs)

    return enhanced_np


@hydra.main(config_path="../configs", config_name="prediction_cfg", version_base=None)
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Find checkpoint
    ckpt_path = find_existing_ckpt(cfg)
    if ckpt_path is None:
        print("ERROR: No checkpoint found. Set checkpoint path in config.")
        return

    print(f"Loading model from: {ckpt_path}")
    model = load_model(cfg, ckpt_path, device)

    # Create test dataset
    cfg.mode = 'test'
    test_dataset = BasicDataset(cfg)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    # Metrics accumulators per SNR
    snr_levels = ['-10', '-5', '0', '5', '10']
    metrics = {snr: {'pesq_nb': 0, 'pesq_wb': 0, 'stoi': 0, 'sisdr': 0,
                      'noisy_pesq_nb': 0, 'noisy_pesq_wb': 0, 'noisy_stoi': 0, 'noisy_sisdr': 0,
                      'count': 0}
               for snr in snr_levels}

    print(f"Evaluating on {len(test_loader)} test samples...")

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input_map, target, raw, noisy_stft_complex, S_stft_complex, snr, file_name = data
            snr_val = str(snr.item())

            # Preprocess
            input_map_proc = preprocess_input(cfg, input_map, device)

            # Inference (chunked for long signals)
            if cfg.cut_prediction:
                enhanced_spec = chunk_inference(model, input_map_proc, cfg, device)
            else:
                enhanced_spec = model(input_map_proc)

            # Get reference signals
            clean = raw[0, 2, :].numpy()
            noisy = raw[0, 0, :].numpy()

            # Reconstruct enhanced waveform
            enhanced = reconstruct_waveform(cfg, enhanced_spec, input_map_proc, len(noisy), device)

            # Trim to same length
            min_len = min(len(enhanced), len(clean))
            enhanced = enhanced[:min_len]
            clean = clean[:min_len]
            noisy = noisy[:min_len]

            # Compute metrics
            if snr_val in metrics:
                m = metrics[snr_val]
                try:
                    m['pesq_nb'] += pesq(cfg.fs, clean, enhanced, 'nb')
                    m['pesq_wb'] += pesq(cfg.fs, clean, enhanced, 'wb')
                    m['noisy_pesq_nb'] += pesq(cfg.fs, clean, noisy, 'nb')
                    m['noisy_pesq_wb'] += pesq(cfg.fs, clean, noisy, 'wb')
                except:
                    pass

                m['stoi'] += stoi(clean, enhanced, cfg.fs, extended=False)
                m['noisy_stoi'] += stoi(clean, noisy, cfg.fs, extended=False)

                m['sisdr'] += si_sdr_torchaudio_calc(
                    torch.from_numpy(enhanced).unsqueeze(0),
                    torch.from_numpy(clean).unsqueeze(0)
                ).item()
                m['noisy_sisdr'] += si_sdr_torchaudio_calc(
                    torch.from_numpy(noisy).unsqueeze(0),
                    torch.from_numpy(clean).unsqueeze(0)
                ).item()
                m['count'] += 1

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1} / {len(test_loader)} samples")

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'SNR':>5}  {'PESQ-NB':>8}  {'PESQ-WB':>8}  {'STOI':>7}  {'SI-SDR':>8}  "
          f"{'Noisy PESQ':>10}  {'Noisy STOI':>10}  {'Noisy SI-SDR':>12}")
    print("-" * 80)

    results = {}
    for snr in snr_levels:
        m = metrics[snr]
        n = max(m['count'], 1)
        results[snr] = {
            'pesq_nb': m['pesq_nb'] / n,
            'pesq_wb': m['pesq_wb'] / n,
            'stoi': m['stoi'] / n,
            'sisdr': m['sisdr'] / n,
            'noisy_pesq_nb': m['noisy_pesq_nb'] / n,
            'noisy_stoi': m['noisy_stoi'] / n,
            'noisy_sisdr': m['noisy_sisdr'] / n,
            'count': m['count'],
        }
        r = results[snr]
        print(f"{snr:>5}  {r['pesq_nb']:>8.3f}  {r['pesq_wb']:>8.3f}  {r['stoi']:>7.4f}  "
              f"{r['sisdr']:>8.2f}  {r['noisy_pesq_nb']:>10.3f}  {r['noisy_stoi']:>10.4f}  "
              f"{r['noisy_sisdr']:>12.2f}")

    # Averages
    total_count = sum(metrics[s]['count'] for s in snr_levels)
    if total_count > 0:
        avg = {k: sum(results[s][k] * results[s]['count'] for s in snr_levels) / total_count
               for k in ['pesq_nb', 'pesq_wb', 'stoi', 'sisdr']}
        print("-" * 80)
        print(f"{'AVG':>5}  {avg['pesq_nb']:>8.3f}  {avg['pesq_wb']:>8.3f}  "
              f"{avg['stoi']:>7.4f}  {avg['sisdr']:>8.2f}")

    # Save results
    results_path = os.path.join(cfg.reports_path, 'evaluation_results.json')
    create_dir(cfg.reports_path)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
