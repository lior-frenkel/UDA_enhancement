"""Training script for domain adaptation in speech enhancement.

Usage:
    python scripts/train.py [hydra overrides]

Examples:
    # Source-only supervised baseline (LibriSpeech)
    python scripts/train.py batch_size=64 lr=1e-4 fs=16000 data_type=librispeech

    # DANN/GRL adaptation (LibriSpeech → DNS)
    python scripts/train.py batch_size=64 lr=1e-4 fs=16000 data_type=dns \\
        use_grl=True pretrained=True bce_percent=0.05

    # Domain classifier pre-training
    python scripts/train.py batch_size=64 lr=1e-4 fs=16000 data_type=dns \\
        classify_domains=True

    # RemixIT unsupervised adaptation
    python scripts/train.py batch_size=64 lr=5e-5 fs=16000 data_type=dns \\
        unsupervised=True teacher_update=True pretrained=True

    # SRST self-retraining
    python scripts/train.py batch_size=64 lr=5e-5 fs=16000 data_type=dns \\
        unsupervised=True use_labeled=True pretrained=True \\
        use_tgt_labels=True target_percent=0.5 increase_every=1
"""

import os
import sys
import shutil
import warnings
import glob

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from omegaconf import DictConfig
import hydra

warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(threshold=sys.maxsize)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.model_def import Pl_UNet_enhance, Pl_UNet_classifier, Pl_UNet2
from src.data.dataloader import EnhancementDataModule
from src.models.utils import find_existing_ckpt

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"


@hydra.main(config_path="../configs", config_name="train_cfg.yaml", version_base=None)
def main(cfg):
    """Main training entry point."""

    pl_checkpoints_path = os.getcwd() + '/'

    # Initialize data
    if cfg.unsupervised and not cfg.use_clpl and not cfg.use_tgt_labels:
        init_data(cfg)

    dm = EnhancementDataModule(cfg)

    # Select model based on training mode
    if cfg.unsupervised or cfg.supervised:
        model = Pl_UNet2(cfg, dm)
    elif cfg.classify_domains:
        model = Pl_UNet_classifier(cfg)
    else:
        model = Pl_UNet_enhance(cfg)

    # Resume from checkpoint if available
    if cfg.resume_from_checkpoint == 'None':
        existing_ckpt = find_existing_ckpt()
        if existing_ckpt:
            model = model.load_from_checkpoint(
                model=model, checkpoint_path=existing_ckpt, cfg=cfg
            )
    else:
        model = model.load_from_checkpoint(
            model=model, checkpoint_path=cfg.resume_from_checkpoint, cfg=cfg
        )

    # Checkpoint callback
    if cfg.use_grl:
        ckpt_monitor = 'val_SISDR_loss_epoch'
        filename = 'epoch-{epoch:02d}-val-loss-{val_loss:.3f}-sisdr-loss-{val_SISDR_loss_epoch:.3f}'
    else:
        ckpt_monitor = cfg.ckpt_monitor
        filename = 'epoch-{epoch:02d}-val-loss-{val_loss:.3f}'

    checkpoint_callback = ModelCheckpoint(
        monitor=ckpt_monitor,
        filename=filename,
        save_last=cfg.save_last,
        save_top_k=cfg.save_top_k,
        mode='min',
        verbose=cfg.verbose
    )

    stop_callback = EarlyStopping(
        monitor=ckpt_monitor,
        patience=cfg.patience,
        check_finite=False
    )

    trainer = Trainer(
        accelerator='cuda',
        fast_dev_run=False,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        default_root_dir=pl_checkpoints_path,
        callbacks=[stop_callback, checkpoint_callback],
        precision=cfg.precision,
        strategy=DDPStrategy(find_unused_parameters=True),
        num_sanity_val_steps=0,
        max_epochs=-1,
        reload_dataloaders_every_n_epochs=1,
    )

    trainer.fit(model, dm)
    print(f"Best model: {checkpoint_callback.best_model_path}")


def init_data(cfg, mode='train'):
    """Copy source data for unsupervised training (RemixIT/SRST)."""
    dir_model = os.path.basename(os.getcwd())
    parent_dir = cfg.reports_path
    dir_mix_data = os.path.join(parent_dir, 'training_files', dir_model, mode, 'mixtures')

    data_path = cfg.data_path

    if cfg.dereverb or cfg.with_noise:
        data_path = os.path.join(data_path, 'reverb')
    elif cfg.use_pra:
        data_path = os.path.join(data_path, 'reverb', 'pra')
        if cfg.use_corners:
            data_path = os.path.join(data_path, 'corners')

    if cfg.fs == 16000:
        data_path = os.path.join(data_path, '16k')

    data_path = os.path.join(data_path, cfg.data_type)

    if cfg.with_noise:
        data_path = os.path.join(data_path, 'with_noise')

    printed_mode = mode + '_full' if (cfg.return_full or not cfg.use_h5) else mode
    data_path = os.path.join(data_path, printed_mode)

    print(f"Source data: {data_path}")
    print(f"Target mixtures: {dir_mix_data}")

    if not os.path.exists(dir_mix_data):
        shutil.copytree(data_path, dir_mix_data)


if __name__ == "__main__":
    main()
