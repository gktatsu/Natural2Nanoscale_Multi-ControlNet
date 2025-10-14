from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import wandb
import datetime
import argparse # Import argparse
import os 

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
    wandb_logger = pl.loggers.WandbLogger(save_dir=output_dir, project=args.wandb_project, log_model='all')

    # --- Checkpointing ---
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename=f'EM_best_results',
        save_weights_only=True,
        monitor= 'train/loss_epoch', #'val_loss', # Often good to monitor a metric, e.g., validation loss
        mode='min',         # 'min' for loss, 'max' for accuracy
        save_top_k=1,       # Save only the best model
        verbose=True
    )

    # --- Data Loading ---
    dataset = MyDataset(args.image_path, args.mask_path)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=args.logger_freq)

    # --- Trainer Setup ---
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        precision=args.precision,
        callbacks=[logger, checkpoint_callback],
        logger=wandb_logger,
        enable_checkpointing=True,
    )

    # --- Train! ---
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()