# FID / KID Computation Tools

This directory contains utility scripts for computing FID (Fréchet Inception Distance) and KID (Kernel Inception Distance). Two main scripts are provided:

| Script | Feature Extractor | Use Case |
|---|---|---|
| `compute_cem_fid.py` | CEM pretrained ResNet50 (CEM500K / CEM1.5M) | FID/KID for EM images |
| `compute_normal_fid.py` | ImageNet pretrained Inception v3 | General FID/KID computation |

---

## Prerequisites (Dependencies)

Both scripts require the following Python packages:

- `torch`
- `torchvision`
- `numpy`
- `scipy`
- `tqdm`
- `Pillow`

If not installed, activate your virtual environment and run:

```bash
pip install torch torchvision numpy scipy tqdm Pillow
```

---

## compute_cem_fid.py (Using CEM ResNet50)

### Overview

`compute_cem_fid.py` uses CEM500K (MoCo v2) or CEM1.5M (SwAV) pretrained ResNet50 as a feature extractor to compute FID between two EM image folders. KID can also be optionally computed.

#### Processing Flow

1. Convert grayscale EM images to 3 channels (replicated values)
2. Resize to specified resolution (default 224×224)
3. Apply normalization matching CEM pretraining (`cem500k`: mean=0.573, std=0.127 / `cem1.5m`: mean=0.576, std=0.128)
4. Extract 2048-dimensional feature vectors from ResNet50's global average pooling layer
5. Compute mean and covariance matrices from features and calculate FID

### Basic Usage

```bash
python fid/compute_cem_fid.py REAL_DIR GEN_DIR [options]
```

- `REAL_DIR`: Directory containing real images
- `GEN_DIR`: Directory containing generated images

### Full Options List

| Option | Default | Description |
|---|---|---|
| `--backbone {cem500k, cem1.5m}` | `cem500k` | CEM pretrained model to use (MoCo v2 or SwAV) |
| `--batch-size INT` | `32` | Batch size for feature extraction |
| `--num-workers INT` | `4` | Number of DataLoader worker processes |
| `--device` | Auto (`cuda` / `cpu`) | Inference device |
| `--image-size INT` | `224` | Size to resize input images |
| `--weights-path PATH` | None | Path to manually downloaded weights file |
| `--download-dir PATH` | None | Directory to cache downloaded weights |
| `--output-json PATH` | `cem_fid.json` | Output file (timestamp auto-appended) |
| `--data-volume STR` | None | Environment memo (Docker mount info, etc.) |
| `--compute-kid` | Disabled | Also compute KID if specified |
| `--kid-subset-size INT` | `1000` | Number of samples per KID subset |
| `--kid-subset-count INT` | `100` | Number of KID subset iterations |
| `--seed INT` | `42` | Random seed for KID |

### Examples

```bash
# Basic usage
python fid/compute_cem_fid.py /data/real /data/generated --backbone cem500k

# With KID computation
python fid/compute_cem_fid.py /data/real /data/generated \
    --backbone cem1.5m \
    --compute-kid \
    --batch-size 64 \
    --output-json results/cem_fid_result.json

# Using local weights file
python fid/compute_cem_fid.py /data/real /data/generated \
    --weights-path ./fid/weights/cem500k_mocov2_resnet50_200ep.pth.tar
```

### Output

- **Console output**: Displays FID value, backbone used, image counts, weight source, etc.
- **JSON file**: Saves the following information:
  - `fid`: FID value
  - `backbone`: Backbone name used
  - `weights`: Weight file source (URL or local path)
  - `num_real`, `num_generated`: Number of evaluated images
  - `image_size`: Input resolution
  - `normalization_mean`, `normalization_std`: Normalization parameters
  - `timestamp_utc`: Execution time (UTC)
  - `real_dir`, `gen_dir`: Absolute paths of input directories
  - `kid`, `kid_std`: KID mean and standard error (only with `--compute-kid`)

### Downloading Weights

Weight files are automatically downloaded from Zenodo on first run. For offline environments, manually download from:

| Backbone | Download URL |
|---|---|
| CEM500K (MoCo v2) | https://zenodo.org/record/6453140/files/cem500k_mocov2_resnet50_200ep.pth.tar |
| CEM1.5M (SwAV) | https://zenodo.org/record/6453160/files/cem15m_swav_resnet50_200ep.pth.tar |

After downloading, specify the file path with the `--weights-path` option.

---

## compute_normal_fid.py (Using ImageNet Inception v3)

### Overview

`compute_normal_fid.py` uses torchvision's ImageNet pretrained Inception v3 (`IMAGENET1K_V1`) to compute FID (and optionally KID) between real and generated image sets. This performs standard Inception-based FID evaluation.

#### Processing Flow

1. Convert images to RGB (grayscale images auto-converted)
2. Resize to 299×299
3. Apply ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
4. Extract 2048-dimensional feature vectors from Inception v3's pre-fc layer
5. Compute mean and covariance matrices from features and calculate FID

### Basic Usage

```bash
python fid/compute_normal_fid.py REAL_DIR GEN_DIR [options]
```

### Full Options List

| Option | Default | Description |
|---|---|---|
| `--batch-size INT` | `32` | Batch size for feature extraction |
| `--num-workers INT` | `4` | Number of DataLoader worker processes |
| `--device` | Auto (`cuda` / `cpu`) | Inference device |
| `--image-size INT` | `299` | Input resolution expected by Inception v3 |
| `--output-json PATH` | `inception_fid.json` | Output file (timestamp auto-appended) |
| `--data-volume STR` | None | Environment memo (Docker mount info, etc.) |
| `--compute-kid` | Disabled | Also compute KID if specified |
| `--kid-subset-size INT` | `1000` | Number of samples per KID subset |
| `--kid-subset-count INT` | `100` | Number of KID subset iterations |
| `--seed INT` | `42` | Random seed for KID |

### Examples

```bash
# Basic usage
python fid/compute_normal_fid.py /data/real /data/generated

# With KID computation
python fid/compute_normal_fid.py /data/real /data/generated \
    --compute-kid \
    --batch-size 64 \
    --output-json results/inception_fid_result.json
```

---

## Preprocessing Differences (Script Comparison)

| Item | compute_cem_fid.py | compute_normal_fid.py |
|---|---|---|
| **Backbone** | CEM ResNet50 | ImageNet Inception v3 |
| **Input Channels** | Grayscale → 3ch replicated | RGB (grayscale auto-converted) |
| **Input Resolution** | 224×224 | 299×299 |
| **Normalization** | CEM pretraining values | ImageNet standard values |
| **Feature Dimension** | 2048 | 2048 |
| **Recommended Use** | EM / Electron microscopy images | General natural images |

---

## Automatic FID Evaluation During Training

The `train.py` script can automatically compute CEM FID at the end of each epoch by specifying the `--enable_val_fid` option.

### Related Options

| Option | Default | Description |
|---|---|---|
| `--enable_val_fid` | `False` | Compute CEM FID during validation |
| `--fid_batch_size` | `2` | Batch size for FID computation |
| `--fid_num_workers` | `0` | Number of workers for FID computation |
| `--fid_ddim_steps` | `50` | DDIM steps for validation image generation |
| `--fid_guidance_scale` | `9.0` | Guidance scale for validation image generation |
| `--fid_eta` | `0.0` | DDIM eta value for validation |
| `--fid_control_strength` | `1.0` | ControlNet strength for validation |
| `--fid_backbone` | `cem500k` | CEM backbone (`cem500k` / `cem1.5m`) |
| `--fid_image_size` | `512` | Input size for CEM backbone |
| `--fid_device` | `cuda` | Device for FID computation |
| `--fid_weights_path` | None | Local path to CEM weights |
| `--fid_download_dir` | None | Cache directory for CEM weights |
| `--fid_seed` | `1234` | Random seed for validation |

### Usage Example

```bash
python train.py \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --image_path /data/images/train \
    --mask_path /data/masks/train \
    --val_image_path /data/images/val \
    --val_mask_path /data/masks/val \
    --enable_val_fid \
    --fid_backbone cem500k \
    --resume_path ./models/control_sd15_ini.ckpt \
    --gpus 1
```

---

## Best Practices

1. **Image Directory Preparation**: Specify directories containing only the images to evaluate (scripts recursively search for images)
2. **Supported Formats**: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp` are supported
3. **GPU Memory**: Increasing `--batch-size` speeds up computation but watch for out-of-memory errors
4. **KID Computation**: Enabling `--compute-kid` stores all features in memory, so be careful with large datasets
5. **Reproducibility**: Use `--seed` option to fix KID sampling

---

## Docker Usage Example

```bash
# CEM FID computation
docker run --gpus all --rm \
  -v /path/to/data:/data \
  -v /path/to/weights:/weights \
  -v /path/to/results:/results \
  hannahkniesel/natural2nanoscale:latest \
  python fid/compute_cem_fid.py \
    /data/real /data/generated \
    --backbone cem500k \
    --weights-path /weights/cem500k_mocov2_resnet50_200ep.pth.tar \
    --output-json /results/cem_fid.json \
    --data-volume /path/to/data:/data
```

---

## Pretraining Directory (pretraining/)

The `pretraining/` directory contains code for CEM backbone pretraining.

| Subdirectory | Contents |
|---|---|
| `mocov2/` | MoCo v2 ResNet50 pretraining (for CEM500K) |
| `swav/` | SwAV ResNet50 pretraining (for CEM1.5M) |

These provide the ResNet50 architecture definitions required for FID computation.

---

## Troubleshooting

### Weight Download Fails

For offline environments or when Zenodo access is restricted:

1. Manually download weight files from the URLs above
2. Place in `fid/weights/` directory (auto-detected by scripts)
3. Or explicitly specify with `--weights-path` option

### CUDA Out of Memory

- Reduce `--batch-size` (e.g., 16 → 8)
- Use `--device cpu` for CPU mode

### Images Not Found

- Check supported extensions (`.png`, `.jpg`, etc.)
- Verify directory path is correct
- Subdirectories are searched recursively

---

## References

- CEM Pretraining: [Zenodo - CEM500K](https://zenodo.org/record/6453140), [Zenodo - CEM1.5M](https://zenodo.org/record/6453160)
- FID Paper: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", NeurIPS 2017
- KID Paper: Bińkowski et al., "Demystifying MMD GANs", ICLR 2018
