"""PyTorch Dataset and DataModule for speech enhancement domain adaptation.

Handles loading of source/target domain data with curriculum scheduling
for SRST/CLPL training paradigms.
"""

import os
import warnings
import math
import numpy as np
import librosa
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from src.data.data_utils import AudioPreProcessing, sort_by_prob, sort_by_pesq, sort_by_stoi, sort_by_sisdr
from src.models.utils import RI_to_log_spec

warnings.filterwarnings("ignore", category=UserWarning)


class BasicDataset(Dataset):
    """Dataset for loading noisy/clean speech pairs.

    Supports:
        - Standard source-only loading
        - Domain-mixed loading for DANN/GRL (source + target with domain labels)
        - Curriculum-based target selection for SRST/CLPL
        - Multiple data configurations: denoising, dereverberation, with_noise

    Args:
        cfg: Configuration object
        data_path: Root data directory
        batch_size: Batch size (used for curriculum calculations)
        current_epoch: Current training epoch
        mode: 'train', 'val', or 'test'
        domain_ratio: Bernoulli parameter for source/target sampling
        source_percent: Fraction of source samples
        target_percent: Fraction of target samples
    """

    def __init__(self, cfg, data_path, batch_size=1, current_epoch=0, mode='train',
                 domain_ratio=1, merge_domains=False, source_percent=0.3,
                 target_percent=0.1, use_raw=False, start_inx=None, end_inx=None):
        self.cfg = cfg
        self.mode = mode
        self.use_raw = use_raw
        self.data_path = data_path
        self.batch_size = batch_size
        self.audio_pre_process = AudioPreProcessing(cfg)
        self.current_epoch = current_epoch
        self.source_percent = source_percent
        self.target_percent = target_percent
        self.cut_signal = not cfg.make_hist_source

        # Resolve data path based on configuration
        self._resolve_data_path(cfg, data_path, mode)

        # Load file list
        self.waves_list = sorted([f for f in os.listdir(self.data_path) if f.endswith(".wav")])

        if start_inx is not None and end_inx is not None:
            self.waves_list = self.waves_list[start_inx:end_inx]

        # Domain adaptation: load source samples
        self.use_source = False
        if (cfg.classify_domains or cfg.use_grl or cfg.unsupervised or cfg.supervised) and mode == 'train':
            self._setup_source_data(cfg, mode)
        elif (cfg.classify_domains or cfg.use_grl) and mode == 'val':
            self._setup_source_data(cfg, mode)
        elif cfg.unsupervised and mode == 'val':
            self._setup_source_data(cfg, mode)

        # Labeled source data for SRST
        if cfg.use_labeled and mode == 'train':
            self._setup_labeled_source(cfg)

        # Normalization
        self.mx_mix = None

    def _resolve_data_path(self, cfg, data_path, mode):
        """Resolve the actual data directory based on config flags."""
        if cfg.dereverb:
            self.data_path = os.path.join(data_path, 'reverb')
        elif cfg.use_pra:
            self.data_path = os.path.join(data_path, 'reverb', 'pra')
            if cfg.use_corners:
                self.data_path = os.path.join(self.data_path, 'corners')
        elif cfg.with_noise:
            self.data_path = os.path.join(data_path, 'reverb')
        else:
            self.data_path = data_path

        if cfg.fs == 16000:
            self.data_path = os.path.join(self.data_path, '16k')

        self.data_path = os.path.join(self.data_path, cfg.data_type)

        if cfg.with_noise and not cfg.dereverb and not cfg.use_pra:
            self.data_path = os.path.join(self.data_path, 'with_noise')

        if cfg.return_full or not cfg.use_h5:
            printed_mode = mode + '_full'
        else:
            printed_mode = mode
        self.data_path = os.path.join(self.data_path, printed_mode)

    def _setup_source_data(self, cfg, mode):
        """Setup source domain data path for domain adaptation."""
        self.use_source = True
        source_key = f'source_path_{cfg.source_domain}_16k' if cfg.fs == 16000 else f'source_path_{cfg.source_domain}'

        if cfg.dereverb:
            source_key += '_dereverb'
        elif cfg.use_pra:
            source_key += '_pra'
        elif cfg.with_noise:
            source_key += '_dereverb_noise'

        if hasattr(cfg, source_key):
            source_base = getattr(cfg, source_key)
        else:
            source_base = cfg.data_path

        if cfg.return_full or not cfg.use_h5:
            self.source_path = os.path.join(source_base, f'{mode}_full')
        else:
            self.source_path = os.path.join(source_base, mode)

        if os.path.exists(self.source_path):
            self.source_waves_list = sorted([f for f in os.listdir(self.source_path) if f.endswith(".wav")])
        else:
            self.source_waves_list = []
            print(f"Warning: source path {self.source_path} not found")

    def _setup_labeled_source(self, cfg):
        """Setup labeled (high-confidence) source data for SRST."""
        labeled_key = f'source_path_{cfg.source_domain}_labeled_16k' if cfg.fs == 16000 else f'source_path_{cfg.source_domain}_labeled'
        if hasattr(cfg, labeled_key):
            self.labeled_path = getattr(cfg, labeled_key)
            if os.path.exists(self.labeled_path):
                self.labeled_list = sorted([f for f in os.listdir(self.labeled_path) if f.endswith(".wav")],
                                           key=sort_by_prob, reverse=True)
            else:
                self.labeled_list = []
        else:
            self.labeled_list = []

    def __len__(self):
        return len(self.waves_list)

    def __getitem__(self, idx):
        cfg = self.cfg
        file_name = self.waves_list[idx]
        file_path = os.path.join(self.data_path, file_name)

        # Load waveform
        mixed, fs = librosa.load(file_path, sr=cfg.fs, mono=False)
        if mixed.ndim == 1:
            mixed = mixed[np.newaxis, :]

        # Normalize
        if np.max(np.abs(mixed[0])) != 0:
            mx = np.max(np.abs(mixed[0]))
            mixed = mixed / 1.1 / mx
            mixed = mixed / np.std(mixed[0])

        # Preprocess to STFT domain
        result = self.audio_pre_process.preprocess(mixed, cut_signal=self.cut_signal)

        if (cfg.pred_spec or cfg.spec_mse) and cfg.mode != 'test':
            input_map, target, targets_clean_log_spec, Z_stft_complex, S_stft_complex, V_out, mixed = result
        else:
            input_map, target, Z_stft_complex, S_stft_complex, V_out, mixed = result

        # Concatenate noise RI to target when reconstructing noise
        if cfg.reconst_noise and V_out is not None and cfg.target == 'stft_RI':
            target = torch.cat((target, V_out), dim=0)  # [4, F, T]: clean_real, clean_imag, noise_real, noise_imag

        # Zero low frequencies (before normalization, matching original pipeline)
        if cfg.input_map == 'stft_RI':
            input_map[..., 0:3, :] = input_map[..., 0:3, :] * 0.001

        # Normalize RI input (target is NOT normalized, matching original)
        mx_mix = torch.max(torch.max(torch.abs(input_map[0, ...]),
                                      torch.max(torch.abs(input_map[1, ...]))))
        if mx_mix > 0:
            input_map = input_map / mx_mix

        # Remove low frequencies (additional flag)
        if cfg.remove_small_freq:
            input_map[..., 0:3, :] = input_map[..., 0:3, :] * 0.001

        # Build raw waveforms tensor
        mixed_tensor = torch.from_numpy(mixed).float()

        if cfg.use_sisdri:
            if self.use_source and self.mode in ('train', 'val'):
                # Return with domain label
                true_domain = torch.tensor(0.0)  # target=0
                return input_map.float(), target.float(), mixed_tensor, Z_stft_complex, file_name, true_domain
            return input_map.float(), target.float(), mixed_tensor, Z_stft_complex, file_name
        else:
            return input_map.float(), target.float(), mixed_tensor, file_name


class EnhancementDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for speech enhancement.

    Handles creation of train/val datasets with proper domain mixing
    and curriculum scheduling for SRST/CLPL.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        cfg = self.cfg
        data_path = cfg.data_path

        self.train_set = BasicDataset(cfg, data_path, batch_size=cfg.batch_size, mode='train')
        self.val_set = BasicDataset(cfg, data_path, batch_size=cfg.batch_size, mode='val')

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
