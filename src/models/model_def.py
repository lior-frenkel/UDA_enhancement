"""Model definitions for domain adaptation in speech enhancement.

Paper-relevant models:
    - UNet: Core speech enhancement U-Net (STFT RI → RI)
    - Domain_UNet: UNet with integrated GRL domain classifier branch
    - Domain_classifier: Standalone domain classifier (encoder + classifier head)
    - Pl_UNet_enhance: PL module for supervised training & DANN/GRL adaptation
    - Pl_UNet_classifier: PL module for pre-training the domain classifier
    - Pl_UNet2: PL module for RemixIT / SRST / supervised fine-tuning

References:
    - DANN: Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016
    - RemixIT: Tzinis et al., "RemixIT: Continual self-training of speech enhancement
      models via bootstrapped remixing", IEEE JSTSP 2022
"""

import os
import warnings
import random
import glob
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchaudio

from .model_parts import DoubleConv, Down, Up, OutConv
from .utils import init_weights_he, si_sdri_loss, RI_to_log_spec, create_dir, si_sdr_torchaudio_calc
from .gradient_reversal import GradientReversal

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def nll_criterion_gaussian(cfg, mu, logvar, target, reduction='mean'):
    """Negative log-likelihood loss for Gaussian distribution."""
    loss = torch.exp(-logvar) * torch.pow(target - mu, 2) + logvar
    return loss.mean() if reduction == 'mean' else loss.sum()


def loss_function(cfg, criterion, pred, target, input_map, mixed_raw=None,
                  noisy_angle=None, current_epoch=0, mode='train',
                  true_domain=None, tgt_alpha=0.5):
    """Compute the training loss based on configuration.

    Supports:
        - SI-SDRi loss (main loss for RI STFT predictions)
        - MSE loss
        - Noise reconstruction loss
        - GRL domain-gated loss (only source samples contribute to SI-SDRi)
        - SRST/CLPL per-domain weighted loss
    """
    window_size = cfg.window_size if cfg.fs == 8000 else 2 * cfg.window_size
    sisdri_enhanced = 0.0

    if cfg.target == 'stft_RI':
        pred_real = pred[:, 0, :, :].float()
        pred_imag = pred[:, 1, :, :].float()
        target_real = target[:, 0, :, :]
        target_imag = target[:, 1, :, :]

        if cfg.reconst_noise and not cfg.dereverb and not cfg.use_pra and not cfg.with_noise:
            pred_noise_real = pred[:, 2, :, :].float()
            pred_noise_imag = pred[:, 3, :, :].float()
            target_noise_real = target[:, 2, :, :]
            target_noise_imag = target[:, 3, :, :]

        if cfg.use_sisdri or cfg.combined_loss:
            if cfg.dereverb or cfg.use_pra or cfg.from_reverb:
                clean_raw = mixed_raw[:, 1, :]
            else:
                clean_raw = mixed_raw[:, 2, :]
            noisy_raw = mixed_raw[:, 0, :]
            enhanced_spec = pred_real + 1j * pred_imag

            if cfg.use_grl:
                # Only compute enhancement loss on source-domain samples
                loss = 0.0
                sisdri_enhanced = 0.0
                loss_count = 0
                for es, cr, nr, td in zip(enhanced_spec, clean_raw, noisy_raw, true_domain):
                    if td == 1:
                        l, s = si_sdri_loss(cfg, es.unsqueeze(0), cr.unsqueeze(0), nr.unsqueeze(0), noisy_angle, enhanced_spec.device)
                        loss += l
                        sisdri_enhanced += s
                        loss_count += 1
                loss /= max(loss_count, 1)
                sisdri_enhanced /= max(loss_count, 1)

            elif (cfg.use_clpl or cfg.use_tgt_labels) and mode != 'val':
                # Weighted source + target loss for SRST/CLPL
                loss_source = 0.0
                loss_target = 0.0
                sisdri_enhanced = 0.0
                loss_source_count = 0.0
                loss_target_count = 0.0
                for es, cr, nr, td in zip(enhanced_spec, clean_raw, noisy_raw, true_domain):
                    l, s = si_sdri_loss(cfg, es.unsqueeze(0), cr.unsqueeze(0), nr.unsqueeze(0), noisy_angle, enhanced_spec.device)
                    if td == 1:
                        loss_source += l
                        loss_source_count += 1
                    else:
                        loss_target += l
                        loss_target_count += 1
                    sisdri_enhanced += s
                if loss_source_count > 0:
                    loss_source /= loss_source_count
                if loss_target_count > 0:
                    loss_target /= loss_target_count
                if cfg.use_tgt_labels:
                    loss = loss_target + loss_source
                else:
                    loss = tgt_alpha * loss_target + (1 - tgt_alpha) * loss_source
                total_count = loss_source_count + loss_target_count
                if total_count > 0:
                    sisdri_enhanced /= total_count
            else:
                loss, sisdri_enhanced = si_sdri_loss(cfg, enhanced_spec, clean_raw, noisy_raw, noisy_angle, enhanced_spec.device)

            # Noise reconstruction branch
            if cfg.reconst_noise and not cfg.dereverb and not cfg.use_pra and not cfg.with_noise:
                noise_raw = mixed_raw[:, 1, :]
                enhanced_noise_spec = pred_noise_real + 1j * pred_noise_imag
                noise_loss, _ = si_sdri_loss(cfg, enhanced_noise_spec, noise_raw, noisy_raw, noisy_angle, enhanced_spec.device)
                loss = loss + noise_loss
        else:
            if cfg.reconst_noise:
                loss = criterion(pred_real, target_real) + criterion(pred_imag, target_imag) + \
                       criterion(pred_noise_real, target_noise_real) + criterion(pred_noise_imag, target_noise_imag)
            else:
                loss = criterion(pred_real, target_real) + criterion(pred_imag, target_imag)
    else:
        loss = criterion(pred, target)

    if cfg.use_sisdri or cfg.combined_loss:
        return loss, sisdri_enhanced
    else:
        return loss


# ---------------------------------------------------------------------------
# Neural network architectures
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """U-Net for speech enhancement operating on STFT Real-Imaginary input.

    Architecture: 4-level encoder-decoder with skip connections.
    Input: [B, 2, F, T] (real + imaginary STFT channels)
    Output: [B, C, F, T] where C depends on configuration
        - C=2: enhanced RI (no noise reconstruction)
        - C=4: enhanced RI + noise RI (with noise reconstruction)

    Optional GRL branch for domain-adversarial training.
    """

    def __init__(self, cfg, bilinear=True):
        super(UNet, self).__init__()
        self.cfg = cfg

        in_channels = 2  # RI input
        hidden = cfg.hidden_size

        # Determine output channels
        if cfg.target == 'stft_RI' and cfg.reconst_noise and not (cfg.with_noise or cfg.dereverb or cfg.use_pra):
            out_channels = 4  # clean RI + noise RI
        elif cfg.target == 'stft_RI' and not cfg.reconst_noise:
            out_channels = 2  # clean RI only
        elif cfg.reconst_noise and (cfg.with_noise or cfg.dereverb or cfg.use_pra):
            out_channels = 2  # clean RI only (reverb scenario)
        else:
            out_channels = 2

        factor = 2 if bilinear else 1

        # Encoder
        self.inc = DoubleConv(in_channels, hidden)
        self.down1 = Down(hidden, hidden * 2)
        self.down2 = Down(hidden * 2, hidden * 4)
        self.down3 = Down(hidden * 4, hidden * 8)
        self.down4 = Down(hidden * 8, hidden * 16 // factor)

        # Optional GRL branch
        if cfg.use_grl:
            conv1d_dim = 16 if cfg.fs == 16000 else 8
            self.grl_module = nn.Sequential(
                GradientReversal(alpha=1.0),
                nn.Conv2d(hidden * 16 // factor, 1, kernel_size=1),
                nn.PReLU(),
                nn.Flatten(start_dim=1, end_dim=2),
                nn.Conv1d(conv1d_dim, 1, kernel_size=1),
                nn.PReLU(),
                nn.Flatten(start_dim=1, end_dim=2),
            )

        # Decoder
        self.up1 = Up(hidden * 16, hidden * 8 // factor, bilinear)
        self.up2 = Up(hidden * 8, hidden * 4 // factor, bilinear)
        self.up3 = Up(hidden * 4, hidden * 2 // factor, bilinear)
        self.up4 = Up(hidden * 2, hidden, bilinear)
        self.outputs = OutConv(hidden, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.cfg.use_grl:
            x_features = self.grl_module(x5)
            grl_out = torch.mean(x_features, dim=-1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outputs(x)

        if self.cfg.use_grl:
            return out, grl_out
        return out


class Domain_classifier(nn.Module):
    """Domain classifier: encoder (from pretrained UNet) + classifier head.

    Used to pre-train the binary source/target classifier before GRL training.
    The encoder weights are initialized from a supervised UNet and frozen.
    """

    def __init__(self, cfg, bilinear=True):
        super(Domain_classifier, self).__init__()
        self.cfg = cfg

        in_channels = 2
        hidden = cfg.hidden_size
        factor = 2 if bilinear else 1

        # Encoder (same architecture as UNet encoder)
        self.inc = DoubleConv(in_channels, hidden)
        self.down1 = Down(hidden, hidden * 2)
        self.down2 = Down(hidden * 2, hidden * 4)
        self.down3 = Down(hidden * 4, hidden * 8)
        self.down4 = Down(hidden * 8, hidden * 16 // factor)

        # Classifier head
        conv1d_dim = 16 if cfg.fs == 16000 else 8
        self.classifier = nn.Sequential(
            nn.Conv2d(hidden * 16 // factor, 1, kernel_size=1),
            nn.PReLU(),
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Conv1d(conv1d_dim, 1, kernel_size=1),
            nn.PReLU(),
            nn.Flatten(start_dim=1, end_dim=2),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_features = self.classifier(x5)
        out = torch.mean(x_features, dim=-1)
        return out


class Domain_UNet(nn.Module):
    """UNet with integrated domain classifier branch (used by DANN/GRL).

    Same encoder-decoder as UNet, but adds a GRL + classifier branch
    at the bottleneck for domain-adversarial training.
    """

    def __init__(self, cfg, bilinear=True):
        super(Domain_UNet, self).__init__()
        self.cfg = cfg

        in_channels = 2
        hidden = cfg.hidden_size

        if cfg.target == 'stft_RI' and cfg.reconst_noise and not (cfg.with_noise or cfg.dereverb or cfg.use_pra):
            out_channels = 4
        elif cfg.target == 'stft_RI' and not cfg.reconst_noise:
            out_channels = 2
        elif cfg.reconst_noise and (cfg.with_noise or cfg.dereverb or cfg.use_pra):
            out_channels = 2
        else:
            out_channels = 2

        factor = 2 if bilinear else 1

        # Encoder
        self.inc = DoubleConv(in_channels, hidden)
        self.down1 = Down(hidden, hidden * 2)
        self.down2 = Down(hidden * 2, hidden * 4)
        self.down3 = Down(hidden * 4, hidden * 8)
        self.down4 = Down(hidden * 8, hidden * 16 // factor)

        # Domain classifier with GRL
        conv1d_dim = 16 if cfg.fs == 16000 else 8
        self.classifier = nn.Sequential(
            nn.Conv2d(hidden * 16 // factor, 1, kernel_size=1),
            nn.PReLU(),
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Conv1d(conv1d_dim, 1, kernel_size=1),
            nn.PReLU(),
            nn.Flatten(start_dim=1, end_dim=2),
        )
        self.grl = GradientReversal(alpha=1.0)

        # Decoder
        self.up1 = Up(hidden * 16, hidden * 8 // factor, bilinear)
        self.up2 = Up(hidden * 8, hidden * 4 // factor, bilinear)
        self.up3 = Up(hidden * 4, hidden * 2 // factor, bilinear)
        self.up4 = Up(hidden * 2, hidden, bilinear)
        self.outputs = OutConv(hidden, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # GRL → classifier branch
        x_grl = self.grl(x5)
        x_features = self.classifier(x_grl)
        grl_out = torch.mean(x_features, dim=-1)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outputs(x)

        return out, grl_out


# ---------------------------------------------------------------------------
# PyTorch Lightning modules
# ---------------------------------------------------------------------------

class Pl_UNet_enhance(pl.LightningModule):
    """Supervised speech enhancement + DANN/GRL domain adaptation.

    Training modes (determined by config):
        1. Source-only supervised: default (no use_grl)
        2. DANN/GRL: use_grl=True, pretrained=True
           Loads pretrained classifier weights, trains with combined
           SI-SDRi + BCE(domain) loss with gradient reversal.

    Args:
        cfg: Hydra/OmegaConf configuration object.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.use_grl and cfg.pretrained:
            self.model = Domain_UNet(cfg)
            classifier_model = Domain_classifier(cfg)
        else:
            self.model = UNet(cfg)

        # Loss
        self.criterion = nn.MSELoss() if cfg.criterion == "MSELoss" else nn.L1Loss()
        self.model.apply(init_weights_he)

        # Load pretrained weights
        if cfg.pretrained and not cfg.use_grl:
            self._load_pretrained_unet(cfg)

        if cfg.use_grl:
            self.bce_criterion = nn.BCEWithLogitsLoss()
            if cfg.pretrained:
                self._load_pretrained_grl(cfg, classifier_model)

    def _load_pretrained_unet(self, cfg):
        """Load pretrained UNet weights for supervised baseline."""
        ckpt_path = cfg.pretrained_ckpt
        if not ckpt_path or ckpt_path == 'None':
            print("Warning: pretrained_ckpt not set, skipping pretrained weight loading")
            return
        pretrained_dict = {
            '.'.join(k.split('.')[1:]): v
            for k, v in torch.load(ckpt_path, map_location='cpu')["state_dict"].items()
            if k.split('.')[0] == 'model'
        }
        model_dict = self.model.state_dict()
        for k, v in pretrained_dict.items():
            if k in model_dict:
                model_dict[k] = v
        self.model.load_state_dict(model_dict)

    def _load_pretrained_grl(self, cfg, classifier_model):
        """Load pretrained domain classifier weights for DANN/GRL."""
        ckpt_path = cfg.pretrained_classifier_ckpt
        if not ckpt_path or ckpt_path == 'None':
            print("Warning: pretrained_classifier_ckpt not set, skipping")
            return
        classifier_model.load_state_dict({
            '.'.join(k.split('.')[1:]): v
            for k, v in torch.load(ckpt_path, map_location='cpu')["state_dict"].items()
            if k.split('.')[0] == 'domain_classifier'
        })
        pretrained_dict = classifier_model.state_dict()
        model_dict = self.model.state_dict()
        for k, v in pretrained_dict.items():
            prefix = k.split('.')[0].split('_')[0]
            if prefix.rstrip('0123456789') in ('down', 'inc', 'classifier', ''):
                if k in model_dict:
                    model_dict[k] = v
        self.model.load_state_dict(model_dict)

    def forward(self, x):
        if self.cfg.use_grl:
            y_hat, _ = self.model(x)
        else:
            y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        if self.cfg.use_sisdri:
            if self.cfg.use_grl:
                input_map, target, mixed, noisy_stft, file_name, true_domain = batch
            else:
                input_map, target, mixed, noisy_stft, file_name = batch
            noisy_angle = torch.angle(noisy_stft)
        else:
            input_map, target, mixed, file_name = batch

        if self.cfg.use_grl:
            pred, grl_out = self.model(input_map)
        else:
            pred = self(input_map)

        if self.cfg.use_sisdri:
            if self.cfg.use_grl:
                loss_sisdri, sisdri_enhanced = loss_function(
                    self.cfg, self.criterion, pred, target, input_map,
                    mixed, noisy_angle, true_domain=true_domain
                )
                bce_loss = self.bce_criterion(grl_out, true_domain)
                loss = loss_sisdri + self.cfg.bce_percent * bce_loss
            else:
                loss, sisdri_enhanced = loss_function(
                    self.cfg, self.criterion, pred, target, input_map, mixed, noisy_angle
                )
                loss_sisdri = loss.clone()
        else:
            loss = loss_function(self.cfg, self.criterion, pred, target, input_map)

        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.cfg.use_sisdri:
            self.log('SI-SDR_enhanced_epoch', sisdri_enhanced, on_step=False, on_epoch=True, prog_bar=True)
            self.log('SISDR_loss_epoch', loss_sisdri, on_step=False, on_epoch=True, prog_bar=True)
        if self.cfg.use_grl:
            self.log('BCE_epoch', bce_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.cfg.use_sisdri:
            if self.cfg.use_grl:
                input_map, target, mixed, noisy_stft, file_name, true_domain = batch
            else:
                input_map, target, mixed, noisy_stft, file_name = batch
            noisy_angle = torch.angle(noisy_stft)
        else:
            input_map, target, mixed, file_name = batch

        if self.cfg.use_grl:
            pred, grl_out = self.model(input_map)
        else:
            pred = self(input_map)

        if self.cfg.use_sisdri:
            if self.cfg.use_grl:
                loss_sisdri, sisdri_enhanced = loss_function(
                    self.cfg, self.criterion, pred, target, input_map,
                    mixed, noisy_angle, true_domain=true_domain
                )
                bce_loss = self.bce_criterion(grl_out, true_domain)
                loss = loss_sisdri + self.cfg.bce_percent * bce_loss
            else:
                loss, sisdri_enhanced = loss_function(
                    self.cfg, self.criterion, pred, target, input_map, mixed, noisy_angle
                )
                loss_sisdri = loss.clone()
        else:
            loss = loss_function(self.cfg, self.criterion, pred, target, input_map)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        if self.cfg.use_sisdri:
            self.log('val_SI-SDR_enhanced', sisdri_enhanced, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_SISDR_loss_epoch', loss_sisdri, on_step=False, on_epoch=True, prog_bar=True)
        if self.cfg.use_grl:
            self.log('val_BCE_epoch', bce_loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2)
        )
        if self.cfg.use_sched:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.8, patience=5, cooldown=3
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        return optimizer


class Pl_UNet_classifier(pl.LightningModule):
    """Domain classifier pre-training module.

    Freezes the UNet encoder (initialized from a supervised model) and trains
    only the classifier head with BCE loss on source/target domain labels.
    This is a prerequisite step before DANN/GRL training.

    Args:
        cfg: Hydra/OmegaConf configuration object.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        model = UNet(cfg)
        self.domain_classifier = Domain_classifier(cfg)

        # Load pretrained encoder from supervised model
        ckpt_path = cfg.pretrained_ckpt
        if not ckpt_path or ckpt_path == 'None':
            print("Warning: pretrained_ckpt not set, skipping pretrained encoder loading for classifier")
        else:
            self._load_pretrained_encoder(cfg, model, ckpt_path)
        for i in range(1, 5):
            for param in self.domain_classifier._modules['down' + str(i)].parameters():
                param.requires_grad = False
        for param in self.domain_classifier._modules['inc'].parameters():
            param.requires_grad = False

        self.criterion = nn.BCEWithLogitsLoss()

    def _load_pretrained_encoder(self, cfg, model, ckpt_path):
        """Load pretrained encoder weights from a supervised checkpoint."""
        model.load_state_dict({
            '.'.join(k.split('.')[1:]): v
            for k, v in torch.load(ckpt_path, map_location='cpu')["state_dict"].items()
            if k.split('.')[0] == 'model'
        })
        pretrained_dict = model.state_dict()
        classifier_dict = self.domain_classifier.state_dict()

        for k, v in pretrained_dict.items():
            prefix = k.split('.')[0].split('_')[0]
            if prefix.rstrip('0123456789') in ('down', '') or prefix == 'inc':
                if k in classifier_dict:
                    classifier_dict[k] = v
        self.domain_classifier.load_state_dict(classifier_dict)

    def forward(self, x):
        return self.domain_classifier(x)

    def training_step(self, batch, batch_idx):
        input_map, target, mixed, noisy_stft, file_name, true_domain = batch
        pred = self(input_map)
        loss = self.criterion(pred, true_domain)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_map, target, mixed, noisy_stft, file_name, true_domain = batch
        pred = self(input_map)
        loss = self.criterion(pred, true_domain)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2)
        )
        return optimizer


class Pl_UNet2(pl.LightningModule):
    """Teacher-student model for RemixIT, SRST, and supervised fine-tuning.

    Training modes:
        1. RemixIT (unsupervised=True, teacher_update=True):
           Teacher generates pseudo-labels, student trains on them.
           Teacher is periodically updated from student's weights.
        2. SRST (use_tgt_labels=True):
           Uses domain classifier confidence to select target samples.
           Mixes source + selected target samples with curriculum scheduling.
        3. Supervised fine-tuning (supervised=True):
           Fine-tunes on limited labeled target data.

    Args:
        cfg: Hydra/OmegaConf configuration object.
        dm: PyTorch Lightning DataModule (optional, for curriculum updates).
    """

    def __init__(self, cfg, dm=None):
        super().__init__()
        self.automatic_optimization = False
        self.dm = dm
        self.cfg = cfg

        # Build teacher and student UNets
        if cfg.unsupervised:
            self.teacher = UNet(cfg)
        self.student = UNet(cfg)

        # Load pretrained weights
        if cfg.pretrained:
            domain_model = Domain_UNet(cfg)
            self._load_from_dann(cfg, domain_model)
        else:
            self._load_from_supervised(cfg)

        if cfg.unsupervised:
            for param in self.teacher.parameters():
                param.requires_grad = False

        self.use_existing_mix = False
        self.epoch_for_loss = 0
        self.current_teacher = 'first'
        self.manual_epoch = 0
        self.max_alter = cfg.max_alter
        self.alter_every = cfg.alter_every
        self.alter_inx = cfg.alter_inx
        self.tgt_alpha = 0.5

        # Mixtures save path
        self.save_path = os.path.join(cfg.reports_path, 'training_files', os.path.basename(os.getcwd()))

        self.criterion = nn.MSELoss() if cfg.criterion == "MSELoss" else nn.L1Loss()

    def _load_from_dann(self, cfg, domain_model):
        """Load weights from a pretrained DANN/GRL model."""
        ckpt_path = cfg.pretrained_ckpt
        if not ckpt_path or ckpt_path == 'None':
            print("Warning: pretrained_ckpt not set, skipping DANN weight loading")
            return
        domain_model.load_state_dict({
            '.'.join(k.split('.')[1:]): v
            for k, v in torch.load(ckpt_path, map_location='cpu')["state_dict"].items()
            if k.split('.')[0] == 'model'
        })
        self.pretrained_dict = domain_model.state_dict()

        def _copy_weights(src_dict, dst_dict):
            for k, v in src_dict.items():
                prefix = k.split('.')[0].split('_')[0]
                if prefix.rstrip('0123456789') in ('down', 'up', '') or prefix in ('inc', 'outputs'):
                    if k in dst_dict:
                        dst_dict[k] = v
            return dst_dict

        if cfg.unsupervised:
            teacher_dict = _copy_weights(self.pretrained_dict, self.teacher.state_dict())
            self.teacher.load_state_dict(teacher_dict)

        if cfg.use_other_weights and hasattr(cfg, 'pretrained_ckpt_2') and cfg.pretrained_ckpt_2:
            domain_model2 = Domain_UNet(cfg)
            domain_model2.load_state_dict({
                '.'.join(k.split('.')[1:]): v
                for k, v in torch.load(cfg.pretrained_ckpt_2, map_location='cpu')["state_dict"].items()
                if k.split('.')[0] == 'model'
            })
            student_dict = _copy_weights(domain_model2.state_dict(), self.student.state_dict())
        else:
            student_dict = _copy_weights(self.pretrained_dict, self.student.state_dict())
        self.student.load_state_dict(student_dict)

    def _load_from_supervised(self, cfg):
        """Load weights from a supervised source-only model."""
        ckpt_path = cfg.pretrained_ckpt
        if not ckpt_path or ckpt_path == 'None':
            print("Warning: pretrained_ckpt not set, skipping supervised weight loading")
            return
        state = {
            '.'.join(k.split('.')[1:]): v
            for k, v in torch.load(ckpt_path, map_location='cpu')["state_dict"].items()
            if k.split('.')[0] == 'model'
        }
        if cfg.unsupervised:
            self.teacher.load_state_dict(state)
        if cfg.use_other_weights and hasattr(cfg, 'pretrained_ckpt_2') and cfg.pretrained_ckpt_2:
            state2 = {
                '.'.join(k.split('.')[1:]): v
                for k, v in torch.load(cfg.pretrained_ckpt_2, map_location='cpu')["state_dict"].items()
                if k.split('.')[0] == 'model'
            }
            self.student.load_state_dict(state2)
        else:
            self.student.load_state_dict(state)

    def forward(self, x):
        if self.training and self.cfg.unsupervised and (
            self.manual_epoch % self.alter_every == 0 and self.cfg.teacher_update
            or not self.use_existing_mix and self.manual_epoch == 0
        ):
            return self.teacher(x)
        return self.student(x)

    def save_batch_data(self, enhanced_data, files_names, noisy_data, mode='train',
                        mixed=None, domains=None):
        """Save teacher's enhanced outputs for pseudo-label generation."""
        separate_path = f'{self.save_path}/{mode}/separate'
        create_dir(separate_path)

        window_size = 2 * self.cfg.window_size if self.cfg.fs == 16000 else self.cfg.window_size

        enhanced_files = []
        noise_files = []
        snrs = []
        samples_ids = []

        for inx, (enhanced_sample, name, noisy_sample) in enumerate(zip(enhanced_data, files_names, noisy_data)):
            name_no_ext = '.'.join(name.split('/')[-1].split(".")[:-1])
            sample = name_no_ext.split("_")[0]

            if self.cfg.use_tgt_labels or self.cfg.use_clpl:
                snr = float(name_no_ext.split("_")[-2]) if (self.cfg.dereverb or self.cfg.use_pra) else int(name_no_ext.split("_")[-2])
            else:
                snr = float(name_no_ext.split("_")[-1]) if (self.cfg.dereverb or self.cfg.use_pra) else int(name_no_ext.split("_")[-1])

            should_process = domains is None or domains[inx] == 0
            if should_process:
                enh_real = enhanced_sample[0, :, :]
                enh_imag = enhanced_sample[1, :, :]
                enh_spec = enh_real + 1j * enh_imag
                enh_wav = torch.istft(enh_spec, n_fft=window_size,
                                      window=torch.hamming_window(window_size).to(self.device),
                                      hop_length=int(window_size * self.cfg.overlap),
                                      length=noisy_data.shape[-1])

                if self.cfg.reconst_noise:
                    noise_real = enhanced_sample[2, :, :]
                    noise_imag = enhanced_sample[3, :, :]
                    noise_spec = noise_real + 1j * noise_imag
                    noise_wav = torch.istft(noise_spec, n_fft=window_size,
                                            window=torch.hamming_window(window_size).to(self.device),
                                            hop_length=int(window_size * self.cfg.overlap),
                                            length=noisy_data.shape[-1])
            else:
                enh_wav = mixed[inx, -1]
                if self.cfg.reconst_noise:
                    noise_wav = mixed[inx, 1]

            enhanced_files.append(enh_wav)
            if self.cfg.reconst_noise:
                noise_files.append(noise_wav)
            snrs.append(snr)
            samples_ids.append(sample)

            # Save mixture wavs for next epoch
            if self.cfg.save_mix_batch and self.manual_epoch % self.alter_every == 0:
                mix_path = f'{self.save_path}/{mode}/mixtures'
                create_dir(mix_path)
                if self.cfg.reconst_noise:
                    save_data = torch.stack((noisy_sample, noise_wav, enh_wav))
                else:
                    save_data = torch.stack((noisy_sample, noisy_sample, enh_wav))
                torchaudio.save(f'{mix_path}/{name_no_ext}.wav', save_data.detach().cpu().float(), sample_rate=self.cfg.fs)

        if self.cfg.reconst_noise:
            return enhanced_files, noise_files, snrs, samples_ids, None
        else:
            return enhanced_files, snrs, samples_ids, None

    def lists2tensor(self, clean_files, noisy_files, noise_files=None):
        """Convert lists of waveforms to a batched tensor."""
        mixed = []
        if noise_files:
            for clean, noise, noisy in zip(clean_files, noise_files, noisy_files):
                mixed.append(torch.stack((noisy, noise, clean)))
        else:
            for clean, noisy in zip(clean_files, noisy_files):
                if self.cfg.use_pra or self.cfg.dereverb:
                    mixed.append(torch.stack((noisy, clean)))
                else:
                    mixed.append(torch.stack((noisy, noisy, clean)))
        return torch.stack(mixed)

    def update_teacher(self):
        """Update teacher weights from the latest student checkpoint."""
        versions = [int(v.split('_')[-1]) for v in os.listdir(f'{os.getcwd()}/lightning_logs')]
        version = max(versions)
        last_ckpt = f'{os.getcwd()}/lightning_logs/version_{version}/checkpoints/last.ckpt'
        loaded_dict = {
            '.'.join(k.split('.')[1:]): v
            for k, v in torch.load(last_ckpt)['state_dict'].items()
            if k.split('.')[0] == 'student'
        }
        self.teacher.load_state_dict(loaded_dict)

    def training_step(self, batch, batch_idx):
        if self.cfg.unsupervised:
            teacher_opt, student_opt = self.optimizers()
        else:
            student_opt = self.optimizers()

        if self.cfg.use_sisdri:
            if self.cfg.use_clpl or self.cfg.use_tgt_labels:
                input_map, target, mixed, noisy_stft, file_name, domain = batch
            else:
                input_map, target, mixed, noisy_stft, file_name = batch
            noisy_angle = torch.angle(noisy_stft)
        else:
            input_map, target = batch

        # Teacher generates pseudo-labels
        if self.cfg.unsupervised:
            self.eval()
            pred_teacher = self.teacher(input_map)

            if (self.cfg.teacher_update and self.manual_epoch % self.alter_every == 0) or self.manual_epoch == 0:
                if self.cfg.reconst_noise:
                    enh_files, noise_files, snrs, ids, _ = self.save_batch_data(pred_teacher, file_name, mixed[:, 0, :])
                else:
                    enh_files, snrs, ids, _ = self.save_batch_data(pred_teacher, file_name, mixed[:, 0, :])
                    noise_files = None
                mixed = self.lists2tensor(enh_files, mixed[:, 0, :], noise_files=noise_files)

            if self.cfg.use_clpl or self.cfg.use_tgt_labels:
                loss_teacher, _ = loss_function(self.cfg, self.criterion, pred_teacher, target, input_map, mixed, noisy_angle, true_domain=domain, tgt_alpha=self.tgt_alpha)
            else:
                loss_teacher, _ = loss_function(self.cfg, self.criterion, pred_teacher, target, input_map, mixed, noisy_angle)
            self.train()

        # Student trains on pseudo-labels / target data
        pred = self.student(input_map)

        if self.cfg.use_sisdri:
            if self.cfg.use_clpl or self.cfg.use_tgt_labels:
                loss, sisdri_enhanced = loss_function(self.cfg, self.criterion, pred, target, input_map, mixed, noisy_angle, true_domain=domain, tgt_alpha=self.tgt_alpha)
            else:
                loss, sisdri_enhanced = loss_function(self.cfg, self.criterion, pred, target, input_map, mixed, noisy_angle)
        else:
            loss = loss_function(self.cfg, self.criterion, pred, target, input_map)

        student_opt.zero_grad()
        self.manual_backward(loss)
        student_opt.step()

        log_dict = {"train_loss_epoch/student": loss, "train_loss_epoch": loss}
        if self.cfg.unsupervised:
            log_dict["train_loss_epoch/teacher"] = loss_teacher
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True)

        if self.cfg.use_sisdri:
            self.log('SI-SDR_enhanced_epoch/student', sisdri_enhanced, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        if self.cfg.use_sisdri:
            input_map, target, mixed, noisy_stft, file_name, input_map_source, mixed_source = batch
            noisy_angle = torch.angle(noisy_stft)
        else:
            input_map, target = batch

        pred = self.student(input_map)

        if self.cfg.use_sisdri:
            loss, sisdri_enhanced = loss_function(self.cfg, self.criterion, pred, target, input_map, mixed, noisy_angle, mode='val')
        else:
            loss = loss_function(self.cfg, self.criterion, pred, target, input_map, mode='val')

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        if self.cfg.use_sisdri:
            self.log('val_SI-SDR_enhanced/student', sisdri_enhanced, on_step=False, on_epoch=True, prog_bar=True)

        if self.cfg.unsupervised:
            pred_teacher = self.teacher(input_map)
            loss_teacher, si_teacher = loss_function(self.cfg, self.criterion, pred_teacher, target, input_map, mixed, noisy_angle, mode='val')
            self.log('val_loss/teacher', loss_teacher, on_step=False, on_epoch=True)
            self.log('val_SI-SDR_enhanced/teacher', si_teacher, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        if (self.manual_epoch + 1) % self.alter_every == 0 and self.cfg.teacher_update:
            self.update_teacher()

        if self.cfg.use_clpl and self.manual_epoch % self.cfg.increase_every == 0 and self.manual_epoch != 0:
            self.tgt_alpha = min(0.95, self.tgt_alpha + self.cfg.alpha_amount)

        self.manual_epoch += 1

    def configure_optimizers(self):
        if self.cfg.unsupervised:
            teacher_opt = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2))
            student_opt = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2))
            return teacher_opt, student_opt
        else:
            return torch.optim.Adam(self.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2))
