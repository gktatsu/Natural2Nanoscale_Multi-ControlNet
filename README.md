# From Natural to Nanoscale: Training ControlNet on Scarce FIB-SEM Data for Augmenting Semantic Segmentation Data
[![Project](https://img.shields.io/badge/Project-Webpage-blue.svg)](https://viscom.uni-ulm.de/publications/from-natural-to-nanoscale-training-controlnet-on-scarce-fib-sem-data-for-augmenting-semantic-segmentation-data/)
[![ICCVW](https://img.shields.io/badge/ICCVW-2025-green.svg)]()
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](source/paper.pdf)

This repository contains the code for 

```
From Natural to Nanoscale: Training ControlNet on Scarce FIB-SEM Data for Augmenting Semantic Segmentation Data
Hannah Kniesel*, Pascal Rapp*, Pedro Hermosilla, Timo Ropinski
ICCVW BIC
```
, which is heavily based on and extends the official ControlNet [1] repository by Lvmin Zhang and Maneesh Agrawala. We are grateful for their foundational work.

Original ControlNet Repository: [https://github.com/lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)


![Teaser](source/image.png)
*Representative samples from the dataset, shown alongside their corresponding color-coded segmentation masks used to condition
ControlNet, and the resulting synthetic images. The visual consistency across samples (columns) highlights the low variability within the
dataset. Despite visible differences between real (row 1) and synthetic (row 3) data, our quantitative experiments demonstrate that the
U-Net can still extract valuable image features from the synthetic samples.*

---
[1] Zhang, Lvmin, Anyi Rao, and Maneesh Agrawala. "Adding conditional control to text-to-image diffusion models." Proceedings of the IEEE/CVF international conference on computer vision. 2023.

## Setup

### ✅ Requirements Summary

- Python: `3.10.13`
- CUDA: `12.1`
- cuDNN: `8.9`
- PyTorch: `2.1.1`
- torchvision: `0.16.1`

---

### 🔧 Option 1: Docker

You can download the docker image directly:
```bash
docker run --gpus all -v $PWD:/workspace -it hannahkniesel/natural2nanoscale:latest bash
```

Or build your own: 
```bash
docker build -t natural2nanoscale .
docker run --gpus all -v $PWD:/workspace -it natural2nanoscale bash
```

Make sure you have Docker ≥ 20.10 and the NVIDIA Container Toolkit installed.

---

### ⚙️ Option 2: Virtualenv + pip

```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r req.txt
```

---

## Model Weights and Data

You can download the all model weights [here](https://viscom.datasets.uni-ulm.de/Natural2Nanoscale/Weights.zip).

If you wish to only download our pretrained controlnet, you can do this [here](https://viscom.datasets.uni-ulm.de/Natural2Nanoscale/ControlNet-Weights.zip)

You can download the generated images and corresponding masks [here](https://viscom.datasets.uni-ulm.de/Natural2Nanoscale/Generated.zip).

The real images originate from Dataset 1 of Devan et al [2]. The data can be found [here](https://data.mendeley.com/datasets/9rdmnn2x4x/1).

---
*[2] Shaga Devan, Kavitha, et al. "Weighted average ensemble-based semantic segmentation in biological electron microscopy images." Histochemistry and Cell Biology 158.5 (2022): 447-462.*

## Train ControlNet

You will need to download the initial sd1.5 checkpoint. There are two options to do this:
1.  You can download `v1-5-pruned.ckpt` from [here](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main) and move it to the `models` directory. Then run `python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt` to prepare the checkpoint for the controlnet architecture.

2. Otherwise, when you download our controlnet model weights (see above), you can use the checkpoint at `ControlNet-Weights/control_sd15_ini.ckpt`.

Next, if you wish to train your own ControlNet, similar as to the one in the paper, run: 
```bash
python train.py \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --image_path /path/to/my/images \
    --mask_path /path/to/my/masks \
    --resume_path ./models/control_sd15_ini.ckpt \
    --gpus 1 \
    --precision 32 \
    --wandb_api_key YOUR_WANDB_KEY_HERE
```

*Please note that the code currently only supports encoding of three classes.*
Model weights and log images will be saved under `./models/<timestamp>_...` by default. Pass `--output_root /custom/path` to store runs in a different base directory.

To train the unified RGBA ControlNet (mask + Canny edge in one tensor), first precompute the conditioning tensors (see next section) and then switch the `condition_type`:

```bash
python train.py \
    --condition_type rgba \
    --image_path /path/to/my/images \
    --rgba_path /path/to/my_rgba/train \
    --val_rgba_path /path/to/my_rgba/val \
    --resume_path ./models/control_sd15_ini.ckpt \
    --gpus 1
```

When `condition_type=rgba` the trainer automatically instantiates a ControlNet whose hint branch expects 4 channels.

---

## Precompute RGBA Conditioning

Use `utils/build_rgba_dataset.py` to fuse masks and freshly computed Canny edges into `(H, W, 4)` tensors before launching cluster jobs:

```bash
python utils/build_rgba_dataset.py \
    --img_dir data/images/train \
    --mask_dir data/masks/train \
    --dest_dir data_rgba/train \
    --fmt npz \
    --canny-low 100 --canny-high 200 --beta-edge 1.0 \
    --preview-max 32
```

### RGBA Build Options

| Option | Default | Description |
|---|---|---|
| `--img_dir` | (required) | Directory containing source images |
| `--mask_dir` | (required) | Directory containing mask images (grayscale labels) |
| `--dest_dir` | (required) | Base directory for output RGBA tensors |
| `--fmt` | `npz` | Output format(s): `npz`, `png`, or both |
| `--preview_dir` | `<dest_dir>/preview` | Directory for preview panels |
| `--preview-max` | `32` | Maximum number of preview panels to save |
| `--num-mask-classes` | `3` | Number of mask classes to one-hot encode |
| `--canny-low` | `100` | Lower hysteresis threshold for Canny |
| `--canny-high` | `200` | Upper hysteresis threshold for Canny |
| `--beta-edge` | `1.0` | Multiplier for edge channel before normalization |
| `--overwrite` | `False` | Overwrite existing outputs |

The script stores compressed `.npz` tensors (mask RGB in channels R/G/B, Canny edge in channel A) and optional `preview/*.png` triptychs (original/mask/edge) to visually inspect alignment before training. The `--dest_dir` acts as the base folder; outputs are written to `<dest_dir>/<canny_low>_<canny_high>` so each threshold pair stays isolated automatically.

---

## Generate Images

To generate images with a pretrained ControlNet do: 
```bash
python generate.py \
    --config_yaml_path ./models/cldm_v15.yaml \
    --model_weights_path ./models/EM_best_results.ckpt \
    --mask_dir /path/to/my/segmentation/masks \
    --output_base_dir ./my_synth_data \
    --n_augmentations_per_mask 1 \
    --batch_size_per_inference 1 
```

For the single-branch RGBA model, point the generator to the precomputed RGBA tensors and enable the new mode:

```bash
python generate.py \
    --generation_mode rgba \
    --rgba_dir data_rgba/val \
    --config_yaml_path ./models/cldm_v15.yaml \
    --rgba_model_path ./models/rgba_controlnet.ckpt \
    --output_base_dir ./my_synth_data_rgba
```

The same RGBA tensors used for training can be re-used during inference, ensuring parity between train/test pipelines.

---

## Training Options Reference

The `train.py` script supports many command-line options:

### Core Training Parameters

| Option | Default | Description |
|---|---|---|
| `--batch_size` | `2` | Batch size for training |
| `--learning_rate` | `1e-5` | Learning rate for the optimizer |
| `--logger_freq` | `300` | Frequency (in batches) to log images to WandB |
| `--sd_locked` | `True` | Lock the Stable Diffusion backbone during training |
| `--only_mid_control` | `False` | Only use mid-block control for ControlNet |
| `--gpus` | `1` | Number of GPUs (0=CPU, -1=all available) |
| `--precision` | `32` | Floating point precision (16 or 32) |
| `--num_workers` | `0` | Number of data loading workers |
| `--output_root` | `./models` | Parent directory for checkpoints and logs |

### Data Path Options

| Option | Default | Description |
|---|---|---|
| `--image_path` | `data/EM-Dataset/train_images` | Path to training images |
| `--mask_path` | `data/EM-Dataset/train_masks` | Path to training masks |
| `--edge_path` | `None` | Path to training edge maps |
| `--rgba_path` | `None` | Path to training RGBA tensors |
| `--val_image_path` | `None` | Path to validation images |
| `--val_mask_path` | `None` | Path to validation masks |
| `--val_edge_path` | `None` | Path to validation edge maps |
| `--val_rgba_path` | `None` | Path to validation RGBA tensors |
| `--rgba_alpha_scale` | `1.0` | Scaling factor for alpha (edge) channel |

### Model Configuration

| Option | Default | Description |
|---|---|---|
| `--resume_path` | `./models/control_sd15_ini.ckpt` | Checkpoint to resume from |
| `--cldm_config_path` | `./models/cldm_v15.yaml` | ControlNet model configuration YAML |
| `--condition_type` | `segmentation` | Condition modality: `segmentation`, `edge`, or `rgba` |

### WandB Configuration

| Option | Default | Description |
|---|---|---|
| `--wandb_project` | `EM-ControlNet` | WandB project name |
| `--wandb_api_key` | `INSERT KEY` | WandB API key (or use `WANDB_API_KEY` env var) |

### Validation FID Options

Enable CEM FID computation during training with `--enable_val_fid`:

| Option | Default | Description |
|---|---|---|
| `--enable_val_fid` | `False` | Compute CEM FID on validation splits after each epoch |
| `--fid_batch_size` | `2` | Batch size for validation generation and FID feature extraction |
| `--fid_num_workers` | `0` | Number of workers for validation dataloaders |
| `--fid_ddim_steps` | `50` | Number of DDIM steps for validation image generation |
| `--fid_guidance_scale` | `9.0` | Classifier-free guidance scale for validation |
| `--fid_eta` | `0.0` | DDIM eta value for validation generation |
| `--fid_control_strength` | `1.0` | Control strength multiplier for validation generation |
| `--fid_backbone` | `cem500k` | Backbone for CEM FID (`cem500k` or `cem1.5m`) |
| `--fid_image_size` | `512` | Input size expected by the CEM backbone |
| `--fid_device` | `cuda` | Device for CEM FID feature extraction |
| `--fid_weights_path` | `None` | Optional local path to pre-downloaded CEM weights |
| `--fid_download_dir` | `None` | Optional directory to cache downloaded CEM weights |
| `--fid_seed` | `1234` | Random seed for validation prompt sampling |

---

## Generation Options Reference

The `generate.py` script provides extensive options for fine-grained control over image generation:

### Model Path Options

| Option | Default | Description |
|---|---|---|
| `--config_yaml_path` | `./models/cldm_v15.yaml` | Path to ControlNet model configuration YAML |
| `--model_weights_path` | `./models/EM_best_results.ckpt` | Path to ControlNet model weights checkpoint |
| `--mask_model_path` | `None` | Path to mask-conditioned ControlNet (defaults to `--model_weights_path`) |
| `--edge_model_path` | `None` | Single edge ControlNet checkpoint (deprecated, use `--edge_model_paths`) |
| `--edge_model_paths` | `[]` | Edge ControlNet checkpoints as `<name>=/path/to/model.ckpt` |
| `--rgba_model_path` | `None` | Checkpoint for RGBA ControlNet (defaults to `--model_weights_path`) |

### Condition Directory Options

| Option | Default | Description |
|---|---|---|
| `--mask_dir` | (required for mask modes) | Path to input mask images |
| `--edge_dir` | `None` | Path to precomputed edge images |
| `--edge_dirs` | `[]` | Edge directories as `<name>=/path/to/edges` |
| `--rgba_dir` | `None` | Directory containing RGBA npz/png control tensors |

### Sampling Parameters

| Option | Default | Description |
|---|---|---|
| `--ddim_steps` | `70` | Number of DDIM sampling steps |
| `--strength` | `2.0` | ControlNet conditioning strength |
| `--scale` | `9.0` | Classifier-free guidance scale |
| `--seed` | `-1` | Random seed (`-1` for random each time) |
| `--eta` | `1.0` | DDIM eta parameter for stochasticity |
| `--guess_mode` | `False` | Enable guess mode (less strict conditioning) |

### Multi-Condition Strength Control

| Option | Default | Description |
|---|---|---|
| `--mask_strength` | `1.0` | Relative strength for mask ControlNet branch |
| `--edge_strengths` | `[]` | Relative strengths for edge branches as `<name>=<float>` |
| `--skip_missing_edges` | `False` | Skip samples with missing edge files instead of erroring |

### Generation Modes

Use `--generation_mode` to select which condition branches to use:

| Mode | Description |
|---|---|
| `mask_only` | Use only mask conditioning |
| `edge_only` | Use only edge conditioning |
| `mask_and_edge` | Use both mask and edge conditioning (Multi-Condition ControlNet) |
| `rgba` | Use unified RGBA conditioning (single 4-channel branch) |

### Output Options

| Option | Default | Description |
|---|---|---|
| `--output_base_dir` | `my_synth_data` | Base directory for generated images and masks |
| `--n_augmentations_per_mask` | `1` | Number of synthetic images per input mask |
| `--batch_size_per_inference` | `1` | Number of samples per inference call |

---

## Multi-Condition ControlNet

The `generate.py` script includes a `MultiConditionControlNet` class that enables simultaneous conditioning on multiple modalities (e.g., segmentation masks and Canny edges). This follows the official ControlNet multi-conditioning design.

### Example: Mask + Edge Generation

```bash
python generate.py \
    --generation_mode mask_and_edge \
    --config_yaml_path ./models/cldm_v15.yaml \
    --mask_model_path ./models/mask_controlnet.ckpt \
    --edge_model_paths canny=./models/edge_controlnet.ckpt \
    --mask_dir /path/to/masks \
    --edge_dirs canny=/path/to/canny_edges \
    --mask_strength 1.0 \
    --edge_strengths canny=1.0 \
    --output_base_dir ./my_synth_data_multi \
    --ddim_steps 50 \
    --scale 9.0
```

This loads separate ControlNet branches for masks and edges, then aggregates their control signals during generation.

---

## FID / KID Computation

See [fid/README.md](fid/README.md) for detailed documentation on computing FID and KID metrics.

### Quick Start

```bash
# CEM FID for EM images
python fid/compute_cem_fid.py /path/to/real /path/to/generated --backbone cem500k

# Standard Inception FID
python fid/compute_normal_fid.py /path/to/real /path/to/generated
```

---

## Project Structure

```
Natural2Nanoscale/
├── train.py              # Main training script for ControlNet
├── generate.py           # Image generation script with multi-condition support
├── tool_add_control.py   # Utility to prepare SD checkpoint for ControlNet
├── config.py             # Global configuration (e.g., save_memory flag)
├── dataset.py            # PyTorch Dataset implementation for training
├── share.py              # Shared imports and setup
│
├── annotator/            # Condition extractors
│   ├── canny/            # Canny edge detector
│   ├── hed/              # HED edge detector
│   ├── midas/            # MiDaS depth estimator
│   ├── mlsd/             # M-LSD line detector
│   ├── openpose/         # OpenPose body keypoint detector
│   └── uniformer/        # Uniformer semantic segmentation
│
├── cldm/                 # ControlNet model implementation
│   ├── cldm.py           # ControlNet architecture
│   ├── ddim_hacked.py    # Modified DDIM sampler for ControlNet
│   ├── hack.py           # Model hacking utilities
│   ├── logger.py         # Image logging for training
│   └── model.py          # Model creation and loading utilities
│
├── ldm/                  # Latent Diffusion Model base code
│   ├── data/             # Data utilities
│   ├── models/           # Diffusion model architectures
│   └── modules/          # Neural network modules (attention, encoders, etc.)
│
├── fid/                  # FID/KID computation tools
│   ├── compute_cem_fid.py    # CEM ResNet50-based FID for EM images
│   ├── compute_normal_fid.py # ImageNet Inception v3-based FID
│   ├── pretraining/      # CEM pretraining utilities (MoCo v2, SwAV)
│   └── README.md         # Detailed FID tool documentation
│
├── utils/                # Utility scripts
│   └── build_rgba_dataset.py  # Precompute RGBA conditioning tensors
│
├── models/               # Model checkpoints and configs
│   ├── cldm_v15.yaml     # ControlNet model configuration
│   ├── control_sd15_ini.ckpt  # Initial SD1.5 checkpoint for ControlNet
│   └── saved/            # Saved training runs
│
├── demo/                 # Demo images (edges, masks)
└── my_synth_data/        # Default output directory for generated images
```

### Key Files

| File | Description |
|---|---|
| `config.py` | Global settings including `save_memory` flag for low-VRAM environments |
| `dataset.py` | Custom PyTorch Dataset for loading image/mask/edge/RGBA pairs with augmentation |
| `share.py` | Common imports and initialization shared across scripts |
| `cldm/model.py` | Model creation with config overrides for RGBA (4-channel) ControlNet |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{kniesel2025natural,
  title={From Natural to Nanoscale: Training ControlNet on Scarce FIB-SEM Data for Augmenting Semantic Segmentation Data},
  author={Kniesel, Hannah and Rapp, Pascal and Hermosilla, Pedro and Ropinski, Timo},
  booktitle={ICCVW BIC},
  year={2025}
}
```

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
