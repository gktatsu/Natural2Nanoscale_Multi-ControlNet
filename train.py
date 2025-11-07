from share import *

import datetime
import argparse # Import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from einops import rearrange
from torch.utils.data import DataLoader

from dataset import MyDataset
from fid import compute_cem_fid as cem_fid
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import wandb


class ValidationFIDCallback(pl.Callback):
    def __init__(
        self,
        val_dataset,
        real_image_dir: str,
        output_dir: str,
        generation_batch_size: int,
        generation_num_workers: int,
        ddim_steps: int,
        guidance_scale: float,
        eta: float,
        control_strength: float,
        metric_batch_size: int,
        metric_num_workers: int,
        metric_device: str,
        backbone: str,
        image_size: int,
        weights_path: Optional[str],
        download_dir: Optional[str],
        random_seed: int,
    ) -> None:
        super().__init__()
        self.val_dataset = val_dataset
        self.real_image_dir = Path(real_image_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.generation_batch_size = generation_batch_size
        self.generation_num_workers = generation_num_workers
        self.ddim_steps = ddim_steps
        self.guidance_scale = guidance_scale
        self.eta = eta
        self.control_strength = control_strength

        self.metric_batch_size = metric_batch_size
        self.metric_num_workers = metric_num_workers
        if metric_device.startswith("cuda") and not torch.cuda.is_available():
            print("ValidationFIDCallback: CUDA unavailable, falling back to CPU for FID computation.")
            metric_device = "cpu"
        self.metric_device = torch.device(metric_device)

        self.backbone = backbone
        self.image_size = image_size
        self.weights_path = Path(weights_path).resolve() if weights_path else None
        self.download_dir = Path(download_dir).resolve() if download_dir else None
        self.random_seed = random_seed

        fid_module_path = Path(cem_fid.__file__).resolve()
        self.repo_root = cem_fid.find_repo_root(fid_module_path)
        self._fid_model: Optional[torch.nn.Module] = None
        self._fid_transform = None
        self._real_stats: Optional[cem_fid.FeatureStats] = None

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        if len(self.val_dataset) == 0:
            return
        if not self.real_image_dir.exists():
            print(f"ValidationFIDCallback: real image directory not found: {self.real_image_dir}")
            return

        generated_dir = self._generate_images(pl_module, trainer.current_epoch)
        fid_value = self._compute_cem_fid(generated_dir, trainer.current_epoch)
        if fid_value is None:
            return

        pl_module.log(
            "val/cem_fid",
            fid_value,
            prog_bar=False,
            logger=True,
            sync_dist=False,
            batch_size=self.metric_batch_size,
        )

    def _generate_images(self, pl_module, epoch: int) -> Path:
        device = pl_module.device
        sampler = DDIMSampler(pl_module)
        epoch_dir = self.output_dir / f"epoch_{epoch:04d}"
        if epoch_dir.exists():
            shutil.rmtree(epoch_dir)
        epoch_dir.mkdir(parents=True, exist_ok=True)

        was_training = pl_module.training
        pl_module.eval()

        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        np.random.seed(self.random_seed + epoch)
        torch.manual_seed(self.random_seed + epoch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed + epoch)

        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.generation_batch_size,
            shuffle=False,
            num_workers=self.generation_num_workers,
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if "hint" not in batch or "txt" not in batch:
                    continue
                control = batch["hint"].to(device=device, dtype=torch.float32)
                control = rearrange(control, "b h w c -> b c h w").contiguous()

                prompts = batch["txt"]
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                cond_txt = pl_module.get_learned_conditioning(prompts).to(device)

                num_samples = control.shape[0]
                shape = (pl_module.channels, control.shape[2] // 8, control.shape[3] // 8)

                pl_module.control_scales = [self.control_strength] * 13

                if self.guidance_scale > 1.0:
                    uc_cross = pl_module.get_unconditional_conditioning(num_samples).to(device)
                    un_cond = {"c_concat": [control], "c_crossattn": [uc_cross]}
                else:
                    un_cond = {"c_concat": [control], "c_crossattn": [cond_txt]}

                cond = {"c_concat": [control], "c_crossattn": [cond_txt]}

                samples, _ = sampler.sample(
                    S=self.ddim_steps,
                    batch_size=num_samples,
                    shape=shape,
                    conditioning=cond,
                    eta=self.eta,
                    unconditional_guidance_scale=self.guidance_scale,
                    unconditional_conditioning=un_cond,
                )

                decoded = pl_module.decode_first_stage(samples)
                decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
                decoded = decoded.permute(0, 2, 3, 1).cpu().numpy()

                for local_idx in range(decoded.shape[0]):
                    array = np.clip(decoded[local_idx] * 255.0, 0, 255).astype(np.uint8)
                    Image.fromarray(array).save(epoch_dir / f"{batch_idx * self.generation_batch_size + local_idx:06d}.png")

        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

        if was_training:
            pl_module.train()

        return epoch_dir

    def _ensure_fid_components(self) -> None:
        if self._fid_model is not None and self._fid_transform is not None:
            return

        model, mean, std, _ = cem_fid.instantiate_backbone(
            variant=self.backbone,
            repo_root=self.repo_root,
            weights_path=self.weights_path,
            download_dir=self.download_dir,
        )
        model = model.to(self.metric_device)
        model.eval()

        self._fid_model = model
        self._fid_transform = cem_fid.build_transform(mean, std, self.image_size)

    def _compute_real_stats(self) -> None:
        if self._real_stats is not None:
            return
        self._ensure_fid_components()
        try:
            real_paths = cem_fid.collect_image_paths(self.real_image_dir)
        except Exception as exc:
            raise RuntimeError(
                f"ValidationFIDCallback: unable to collect validation images from {self.real_image_dir}"
            ) from exc
        dataset = cem_fid.ImageFolderDataset(real_paths, self._fid_transform)
        loader = DataLoader(
            dataset,
            batch_size=self.metric_batch_size,
            num_workers=self.metric_num_workers,
            pin_memory=self.metric_device.type == "cuda",
        )
        self._real_stats = cem_fid.compute_feature_stats(
            model=self._fid_model,
            dataloader=loader,
            device=self.metric_device,
            store=False,
            desc="FID Real",
        )

    def _compute_cem_fid(self, generated_dir: Path, epoch: int) -> Optional[float]:
        self._compute_real_stats()
        if self._real_stats is None:
            return None

        self._ensure_fid_components()
        try:
            gen_paths = cem_fid.collect_image_paths(generated_dir)
        except Exception:
            print(f"ValidationFIDCallback: no generated images found in {generated_dir}")
            return None

        dataset = cem_fid.ImageFolderDataset(gen_paths, self._fid_transform)
        loader = DataLoader(
            dataset,
            batch_size=self.metric_batch_size,
            num_workers=self.metric_num_workers,
            pin_memory=self.metric_device.type == "cuda",
        )
        gen_stats = cem_fid.compute_feature_stats(
            model=self._fid_model,
            dataloader=loader,
            device=self.metric_device,
            store=False,
            desc=f"FID Generated (epoch {epoch})",
        )

        fid_value = cem_fid.compute_fid(
            self._real_stats.mean,
            self._real_stats.cov,
            gen_stats.mean,
            gen_stats.cov,
        )

        return float(fid_value)

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train ControlNet model.")

    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training."
    )
    parser.add_argument(
        "--logger_freq",
        type=int,
        default=300,
        help="Frequency (in batches) to log images to WandB."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--sd_locked",
        type=bool, # Note: argparse handles bools from 'true'/'false' strings if you use action='store_true'/'store_false'
        default=True,
        help="Whether to lock the SD (Stable Diffusion) backbone during training."
    )
    parser.add_argument(
        "--only_mid_control",
        type=bool, # Same note as above for boolean args
        default=False,
        help="Whether to only use mid-block control for ControlNet."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="data/EM-Dataset/train_images",
        help="Path to the directory containing training images."
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default="data/EM-Dataset/train_masks",
        help="Path to the directory containing training masks."
    )
    parser.add_argument(
        "--edge_path",
        type=str,
        default=None,
        help="Path to the directory containing training edge maps."
    )
    parser.add_argument(
        "--val_image_path",
        type=str,
        default=None,
        help="Path to the directory containing validation images."
    )
    parser.add_argument(
        "--val_mask_path",
        type=str,
        default=None,
        help="Path to the directory containing validation masks."
    )
    parser.add_argument(
        "--val_edge_path",
        type=str,
        default=None,
        help="Path to the directory containing validation edge maps."
    )
    parser.add_argument(
        "--condition_type",
        type=str,
        default="segmentation",
        choices=["segmentation", "edge"],
        help="Condition modality to use for ControlNet conditioning."
    )
    parser.add_argument(
        "--enable_val_fid",
        action="store_true",
        help="If set, compute CEM FID on the validation splits after each epoch."
    )
    parser.add_argument(
        "--fid_batch_size",
        type=int,
        default=2,
        help="Batch size for validation generation and FID feature extraction."
    )
    parser.add_argument(
        "--fid_num_workers",
        type=int,
        default=0,
        help="Number of workers for validation generation and FID dataloaders."
    )
    parser.add_argument(
        "--fid_ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM steps to use during validation image generation."
    )
    parser.add_argument(
        "--fid_guidance_scale",
        type=float,
        default=9.0,
        help="Classifier-free guidance scale for validation generation."
    )
    parser.add_argument(
        "--fid_eta",
        type=float,
        default=0.0,
        help="DDIM eta value for validation generation."
    )
    parser.add_argument(
        "--fid_control_strength",
        type=float,
        default=1.0,
        help="Control strength multiplier for ControlNet during validation generation."
    )
    parser.add_argument(
        "--fid_backbone",
        type=str,
        default="cem500k",
        choices=list(cem_fid.MODEL_CONFIGS.keys()),
        help="Backbone configuration to use for CEM FID computation."
    )
    parser.add_argument(
        "--fid_image_size",
        type=int,
        default=512,
        help="Input size expected by the CEM backbone."
    )
    parser.add_argument(
        "--fid_device",
        type=str,
        default="cuda",
        help="Device to run CEM FID feature extraction on (cuda or cpu)."
    )
    parser.add_argument(
        "--fid_download_dir",
        type=str,
        default=None,
        help="Optional directory to cache downloaded CEM weights."
    )
    parser.add_argument(
        "--fid_weights_path",
        type=str,
        default=None,
        help="Optional local path to pre-downloaded CEM weights."
    )
    parser.add_argument(
        "--fid_seed",
        type=int,
        default=1234,
        help="Random seed used for validation prompt sampling."
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default='./models/control_sd15_ini.ckpt',
        help="Path to the checkpoint to resume training from or initialize the model."
    )
    parser.add_argument(
        "--cldm_config_path",
        type=str,
        default='./models/cldm_v15.yaml',
        help="Path to the ControlNet model configuration YAML file."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="EM-ControlNet",
        help="WandB project name."
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default="INSERT KEY", # Consider loading this from an environment variable for security
        help="WandB API key. For production, consider using WANDB_API_KEY environment variable."
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for training. Set to 0 for CPU, or -1 for all available GPUs."
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        choices=[16, 32],
        help="Floating point precision for training (16 or 32)."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0, # Keeping original 0, but often >0 is better for data loading
        help="Number of data loading workers."
    )


    args = parser.parse_args()
    return args

# --- Main Script ---
def main():
    args = parse_args()

    # --- WandB Login (handle API key more securely if in production) ---
    if args.wandb_api_key and args.wandb_api_key != "INSERT KEY":
        wandb.login(key=args.wandb_api_key)
    elif os.environ.get("WANDB_API_KEY"):
        wandb.login() # Logs in using the environment variable
    else:
        print("WandB API key not provided as argument or environment variable. WandB logging might fail.")

    # --- Model Loading ---
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.cldm_config_path).cpu()
    # Check strict=False to allow loading partial models if necessary
    model.load_state_dict(load_state_dict(args.resume_path, location='cpu'), strict=False)
    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control
    
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    output_dir = f"./models/{timestamp_str}/"
    os.makedirs(output_dir, exist_ok=True)

    # --- WandB Initialization ---
    wandb.init(
        project=args.wandb_project,
        config=vars(args), # Log all argparse arguments to WandB config
        name=f"training_run_{timestamp_str}" # Optional: add a run name
    )
    wandb_logger = pl.loggers.WandbLogger(save_dir=output_dir, project=args.wandb_project, log_model=False)

    # --- Data Loading ---
    if args.condition_type == "segmentation":
        train_condition_dir = args.mask_path
        if train_condition_dir is None:
            raise ValueError("--mask_path must be provided when condition_type is 'segmentation'.")
    else:
        train_condition_dir = args.edge_path
        if train_condition_dir is None:
            raise ValueError("--edge_path must be provided when condition_type is 'edge'.")

    train_dataset = MyDataset(
        args.image_path,
        train_condition_dir,
        augment=True,
        condition_type=args.condition_type,
    )
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    checkpoint_monitor = 'train/loss_epoch'
    if args.val_image_path:
        if args.condition_type == "segmentation":
            val_condition_dir = args.val_mask_path
            if val_condition_dir is None:
                raise ValueError("--val_mask_path must be provided when condition_type is 'segmentation'.")
        else:
            val_condition_dir = args.val_edge_path
            if val_condition_dir is None:
                raise ValueError("--val_edge_path must be provided when condition_type is 'edge'.")

        val_dataset = MyDataset(
            args.val_image_path,
            val_condition_dir,
            augment=False,
            condition_type=args.condition_type,
        )
        val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
        checkpoint_monitor = 'val/loss'

    # --- Checkpointing ---
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename=f'EM_best_results',
        save_weights_only=True,
        monitor=checkpoint_monitor,
        mode='min',
        save_top_k=1,
        verbose=True
    )

    logger = ImageLogger(batch_frequency=args.logger_freq)

    fid_callback = None
    if val_loader is not None and args.enable_val_fid:
        fid_output_dir = os.path.join(output_dir, "val_generations")
        fid_callback = ValidationFIDCallback(
            val_dataset=val_dataset,
            real_image_dir=args.val_image_path,
            output_dir=fid_output_dir,
            generation_batch_size=args.fid_batch_size,
            generation_num_workers=args.fid_num_workers,
            ddim_steps=args.fid_ddim_steps,
            guidance_scale=args.fid_guidance_scale,
            eta=args.fid_eta,
            control_strength=args.fid_control_strength,
            metric_batch_size=args.fid_batch_size,
            metric_num_workers=args.fid_num_workers,
            metric_device=args.fid_device,
            backbone=args.fid_backbone,
            image_size=args.fid_image_size,
            weights_path=args.fid_weights_path,
            download_dir=args.fid_download_dir,
            random_seed=args.fid_seed,
        )

    # --- Trainer Setup ---
    callbacks = [logger, checkpoint_callback]
    if fid_callback is not None:
        callbacks.append(fid_callback)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        precision=args.precision,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_checkpointing=True,
    )

    # --- Train! ---
    if val_loader is not None:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_dataloaders=train_loader)

if __name__ == "__main__":
    main()