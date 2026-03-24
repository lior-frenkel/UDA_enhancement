# UDA Enhancement

Unsupervised Domain Adaptation (UDA) for deep-learning-based speech enhancement.

This repository contains the code for reproducing the experiments in our paper on domain adaptation methods for speech enhancement.

## Overview

We investigate several unsupervised domain adaptation (UDA) strategies for adapting a speech enhancement model trained on a **source domain** (with paired clean/noisy data) to a **target domain** (with only noisy data):

| Method | Description |
|--------|-------------|
| **Supervised** | Baseline UNet trained on source domain with SI-SDRi loss |
| **DANN** | Domain-Adversarial Neural Network with Gradient Reversal Layer |
| **RemixIT** | Teacher-student self-training with noise remixing |
| **SSST** | Source-Supervised Self-Training (RemixIT + supervised source samples). Called `SRST` internally in the codebase. |
| **SSST+CL** | SSST with Curriculum Pseudo-Labeling (gradually increasing pseudo-labeled target samples). Called `CLPL` internally. |

## Repository Structure

```
├── configs/
│   ├── train_cfg.yaml          # Training configuration
│   ├── prediction_cfg.yaml     # Evaluation configuration
│   ├── generate_data_cfg.yaml  # Data generation configuration
│   └── experiments/            # Per-method experiment configs
│       ├── supervised.yaml
│       ├── dann.yaml
│       ├── remixit.yaml
│       ├── ssst.yaml
│       └── ssst_cl.yaml
├── scripts/
│   ├── train.py                # Training entry point
│   └── evaluate.py             # Evaluation script
├── src/
│   ├── models/
│   │   ├── model_def.py        # All PL model definitions
│   │   ├── model_parts.py      # UNet building blocks
│   │   ├── utils.py            # SI-SDRi loss, utilities
│   │   └── gradient_reversal/  # GRL module for DANN
│   └── data/
│       ├── dataloader.py       # PyTorch dataset and datamodule
│       └── data_utils.py       # Audio preprocessing
├── inference/
│   ├── __init__.py
│   └── enhance.py              # Standalone inference tool
├── data/                       # Data directory (gitignored)
│   ├── raw/                    # Raw downloaded datasets
│   └── processed/              # Generated training/test data
├── models/                     # Saved checkpoints (gitignored)
└── reports/                    # Evaluation results (gitignored)
```

## Setup

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/UDA_enhancement.git
cd UDA_enhancement
pip install -e ".[data_gen]"

# Or just install requirements
pip install -r requirements.txt
```

**Python**: ≥ 3.8  
**GPU**: NVIDIA GPU with CUDA support recommended  
**PyTorch**: ≥ 1.10

---

## Pretrained Checkpoints

Download the pretrained model checkpoints and place them under `models/`:

| Model | Description | Download |
|-------|-------------|----------|
| `supervised.ckpt` | Supervised baseline (source domain) | [Dropbox](DROPBOX_LINK_HERE) |
| `dann.ckpt` | DANN with gradient reversal | [Dropbox](DROPBOX_LINK_HERE) |
| `remixit.ckpt` | RemixIT teacher-student | [Dropbox](DROPBOX_LINK_HERE) |
| `ssst.ckpt` | SSST (our proposed method) | [Dropbox](DROPBOX_LINK_HERE) |
| `ssst_cl.ckpt` | SSST + Curriculum Learning | [Dropbox](DROPBOX_LINK_HERE) |

```bash
# Download all checkpoints:
mkdir -p models
cd models
# Replace with actual Dropbox links:
# wget -O supervised.ckpt "DROPBOX_LINK_HERE"
# wget -O ssst.ckpt "DROPBOX_LINK_HERE"
# ... etc.
```

> **Note**: Replace `DROPBOX_LINK_HERE` with the actual shared Dropbox links (use `?dl=1` suffix for direct download).

---

## Data Preparation

### 1. Download Raw Datasets

You need the following datasets. Download them and place under `data/raw/`:

| Dataset | Purpose | Download |
|---------|---------|----------|
| **LibriSpeech** train-clean-360 | Clean speech (source/target) | [openslr.org/12](https://www.openslr.org/12) |
| **DNS Challenge** | Multi-language clean speech + noise | [github.com/microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge) |
| **WHAM!** | Environmental noise | [wham.whisper.ai](http://wham.whisper.ai/) |
| **MUSAN** | Music + noise corpus | [openslr.org/17](https://www.openslr.org/17) |
| **DEMAND** | Environmental noise (optional) | [zenodo.org/record/1227121](https://zenodo.org/record/1227121) |

After downloading, update the paths in `configs/generate_data_cfg.yaml`:

```yaml
s_path: /path/to/LibriSpeech/train-clean-360
n_path: /path/to/wham_noise
dns_s_path: /path/to/dns_dataset/datasets/clean
dns_n_path: /path/to/dns_dataset/datasets/noise
musan_path: /path/to/musan
demand_path: /path/to/DEMAND
```

### 2. Generate Training Data

Generate paired noisy/clean data at 16kHz for each domain:

```bash
# LibriSpeech + WHAM! noise (train/val/test)
python src/data/create_dataset.py \
    mode=train data_type=librispeech fs=16000 \
    max_sentences=50000 save_data_path=./data/processed/16k/librispeech

python src/data/create_dataset.py \
    mode=val data_type=librispeech fs=16000 \
    save_data_path=./data/processed/16k/librispeech

python src/data/create_dataset.py \
    mode=test data_type=librispeech fs=16000 \
    test_max_sentences=250 samples_per_snr=50 return_full=True \
    save_data_path=./data/processed/16k/librispeech

# DNS (multi-language speech + DNS noise)
python src/data/create_dataset.py \
    mode=train data_type=dns fs=16000 \
    max_sentences=50000 save_data_path=./data/processed/16k/dns

python src/data/create_dataset.py \
    mode=val data_type=dns fs=16000 \
    save_data_path=./data/processed/16k/dns

python src/data/create_dataset.py \
    mode=test data_type=dns fs=16000 \
    test_max_sentences=250 samples_per_snr=50 return_full=True \
    save_data_path=./data/processed/16k/dns
```

Each generated sample is a WAV file with channels: `[noisy, noise, clean]`.

### 3. Data Directory Layout

After generation, your data directory should look like:

```
data/processed/16k/
├── librispeech/
│   ├── train_full/      # 50k training samples
│   ├── val_full/        # 3k validation samples
│   └── test_full/       # 250 test samples (50 per SNR)
└── dns/
    ├── train_full/
    ├── val_full/
    └── test_full/
```

---

## Training

### Supervised Baseline

```bash
python scripts/train.py \
    batch_size=64 lr=1e-4 fs=16000 \
    data_type=librispeech
```

### DANN (Domain-Adversarial)

**Step 1**: Pre-train domain classifier with frozen encoder:
```bash
python scripts/train.py \
    batch_size=64 lr=1e-4 fs=16000 \
    data_type=dns source_domain=librispeech \
    classify_domains=True
```

**Step 2**: Fine-tune with gradient reversal:
```bash
python scripts/train.py \
    batch_size=64 lr=1e-4 fs=16000 \
    data_type=dns source_domain=librispeech \
    use_grl=True bce_percent=0.05 pretrained=True \
    pretrained_ckpt=models/supervised.ckpt \
    pretrained_classifier_ckpt=models/classifier.ckpt
```

### RemixIT

```bash
python scripts/train.py \
    batch_size=64 lr=5e-5 fs=16000 \
    data_type=dns \
    unsupervised=True teacher_update=True \
    pretrained=True use_other_weights=True \
    pretrained_ckpt=models/supervised.ckpt \
    pretrained_ckpt_2=models/supervised_2.ckpt
```

### SSST (Source-Supervised Self-Training)

```bash
python scripts/train.py \
    batch_size=64 lr=5e-5 fs=16000 \
    data_type=dns \
    unsupervised=True teacher_update=True \
    pretrained=True use_other_weights=True \
    use_labeled=True source_percent=0.3 \
    reduce_from=2 reduce_amount=0.03 min_source=2 \
    pretrained_ckpt=models/supervised.ckpt \
    pretrained_ckpt_2=models/supervised_2.ckpt
```

### SSST+CL (SSST with Curriculum Pseudo-Labeling)

```bash
python scripts/train.py \
    batch_size=64 lr=5e-5 fs=16000 \
    data_type=dns \
    unsupervised=True pretrained=True use_other_weights=True \
    use_labeled=True use_tgt_labels=True use_clpl=True \
    target_percent=0.5 increase_every=1 \
    pretrained_ckpt=models/supervised.ckpt \
    pretrained_ckpt_2=models/supervised_2.ckpt
```

---

## Evaluation

```bash
# Evaluate on target domain test set
python scripts/evaluate.py \
    data_type=librispeech data_type_test=dns \
    fs=16000 version=0

# Results are printed per-SNR and saved to reports/evaluation_results.json
```

Metrics computed: **PESQ** (NB & WB), **STOI**, **SI-SDR**.

---

## Inference on Custom Audio

```python
from inference.enhance import SpeechEnhancer

enhancer = SpeechEnhancer("models/ssst.ckpt", device="cuda", fs=16000)

# Enhance a file
enhanced = enhancer.enhance_file("noisy_input.wav", output_path="enhanced_output.wav")

# Or enhance a tensor
import torchaudio
waveform, sr = torchaudio.load("noisy.wav")
enhanced = enhancer.enhance(waveform, fs=sr)
```

---

## Model Architecture

The core model is a **UNet** operating on the Real-Imaginary (RI) STFT representation:

- **Input**: 2-channel RI STFT of noisy signal `[B, 2, F, T]`
- **Output**: 4-channel RI STFT `[B, 4, F, T]` — enhanced speech RI + estimated noise RI
- **Encoder**: 4 down-sampling blocks (DoubleConv + MaxPool)
- **Decoder**: 4 up-sampling blocks with skip connections
- **Hidden size**: 64 base, doubling at each level
- **Loss**: Scale-Invariant SDR improvement (SI-SDRi)

Audio parameters:
- Sample rate: 16 kHz
- STFT window: 512 samples (Hamming), 50% overlap
- Training frames: 129 (~2 seconds)

---

## Experiment → Paper Mapping

| Config file | Paper Section / Table |
|------------|----------------------|
| `experiments/supervised.yaml` | Tables 1-3, "Supervised" baseline |
| `experiments/dann.yaml` | Tables 1-3, "DANN" |
| `experiments/remixit.yaml` | Tables 1-3, "RemixIT" |
| `experiments/ssst.yaml` | Tables 1-3, "SSST" (our proposed method) |
| `experiments/ssst_cl.yaml` | Tables 1-3, "SSST+CL" (with curriculum learning) |

Domain scenarios tested:
- **LibriSpeech → DNS**: Different speech languages, different noise types
- **DNS → LibriSpeech**: Reverse adaptation direction
- **Clean → Reverberant**: Adaptation to reverberant conditions
- **LibriSpeech+WHAM → LibriSpeech+MUSAN Music**: Different noise types

---

## Citation

```bibtex
@article{uda_enhancement,
    title={Unsupervised Domain Adaptation for Speech Enhancement},
    author={...},
    year={2025}
}
```

## License

See [LICENSE](LICENSE) for details.
