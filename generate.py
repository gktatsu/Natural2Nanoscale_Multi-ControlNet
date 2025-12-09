from share import *
import config # Assuming this config module still has necessary global settings

import os
import json
import cv2
import einops
# import gradio as gr
import numpy as np
import torch
import random
import datetime
import argparse # Import argparse
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

DEFAULT_EDGE_KEY = "edge"


def build_hint_override(hint_channels: int):
    return {
        "model": {
            "params": {
                "control_stage_config": {
                    "params": {
                        "hint_channels": hint_channels
                    }
                }
            }
        }
    }


@dataclass(frozen=True)
class ConditionDefinition:
    key: str
    modality: str
    display_name: str
    channels: int = 3


class MultiConditionControlNet(torch.nn.Module):
    """Aggregates multiple ControlNet branches following the official ControlNet multi-conditioning design."""

    def __init__(
        self,
        modules: Sequence[torch.nn.Module],
        channel_splits: Sequence[int],
        condition_keys: Sequence[str],
        per_condition_scales: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if not (len(modules) == len(channel_splits) == len(condition_keys)):
            raise ValueError("modules, channel_splits, and condition_keys must have the same length")

        self.control_modules = torch.nn.ModuleList(modules)
        self.channel_splits = list(channel_splits)
        self.condition_keys = list(condition_keys)

        if per_condition_scales is None:
            self.per_condition_scales = [1.0] * len(self.control_modules)
        else:
            if len(per_condition_scales) != len(self.control_modules):
                raise ValueError("per_condition_scales must have the same length as modules")
            self.per_condition_scales = list(per_condition_scales)

    def forward(self, x, hint, timesteps, context, **kwargs):
        if hint is None:
            raise ValueError("hint tensor must not be None when using MultiConditionControlNet")

        hint_splits = torch.split(hint, self.channel_splits, dim=1)
        if len(hint_splits) != len(self.control_modules):
            raise ValueError(
                f"Expected {len(self.control_modules)} hint splits but received {len(hint_splits)}"
            )

        aggregated_control = None
        for module, hint_chunk, strength in zip(self.control_modules, hint_splits, self.per_condition_scales):
            control_outputs = module(x=x, hint=hint_chunk, timesteps=timesteps, context=context, **kwargs)
            if strength != 1.0:
                control_outputs = [c * strength for c in control_outputs]

            if aggregated_control is None:
                aggregated_control = [c.clone() for c in control_outputs]
            else:
                aggregated_control = [acc + cur for acc, cur in zip(aggregated_control, control_outputs)]

        return aggregated_control

    def set_condition_scale(self, key: str, value: float) -> None:
        try:
            idx = self.condition_keys.index(key)
        except ValueError as exc:
            raise KeyError(f"Condition key '{key}' not registered in MultiConditionControlNet") from exc
        self.per_condition_scales[idx] = float(value)

    def get_condition_scales(self) -> Dict[str, float]:
        return {key: scale for key, scale in zip(self.condition_keys, self.per_condition_scales)}


def freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def parse_key_value_pairs(items: Sequence[str], flag_name: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Entries for {flag_name} must be in the format '<name>=<path>', got: {raw}")
        name, value = raw.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError(f"Invalid entry for {flag_name}: '{raw}'")
        if name in result:
            raise ValueError(f"Duplicate key '{name}' encountered in {flag_name}")
        result[name] = value
    return result


def parse_strength_pairs(items: Sequence[str], flag_name: str) -> Dict[str, float]:
    strength_map: Dict[str, float] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Entries for {flag_name} must be in the format '<name>=<float>', got: {raw}")
        name, value = raw.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError(f"Invalid entry for {flag_name}: '{raw}'")
        try:
            strength_map[name] = float(value)
        except ValueError as exc:
            raise ValueError(f"Strength value for '{name}' in {flag_name} must be a float, got '{value}'") from exc
    return strength_map


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rgba_condition(path: str) -> np.ndarray:
    if path.lower().endswith(".npz"):
        data = np.load(path)
        if "rgba" not in data:
            raise KeyError(f"NPZ file {path} missing 'rgba' array")
        rgba = data["rgba"].astype(np.float32)
        value_range = None
        if "meta" in data:
            try:
                meta_raw = data["meta"]
                if isinstance(meta_raw, np.ndarray):
                    meta_raw = meta_raw.tolist()
                if isinstance(meta_raw, bytes):
                    meta_raw = meta_raw.decode("utf-8")
                meta = json.loads(str(meta_raw))
                value_range = meta.get("value_range")
            except Exception:
                value_range = None
        if value_range == "neg_one_to_one":
            rgba = (rgba + 1.0) / 2.0
        elif value_range == "zero_one":
            pass
        elif rgba.max() > 1.0:
            rgba = rgba / 255.0
        return np.clip(rgba, 0.0, 1.0)

    rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise FileNotFoundError(f"Unable to load RGBA condition at {path}")
    if rgba.ndim == 2:
        rgba = cv2.cvtColor(rgba, cv2.COLOR_GRAY2RGBA)
    if rgba.shape[2] == 3:
        alpha = np.zeros_like(rgba[:, :, :1])
        rgba = np.concatenate([rgba, alpha], axis=-1)
    elif rgba.shape[2] > 4:
        rgba = rgba[:, :, :4]
    rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
    return np.clip(rgba.astype(np.float32) / 255.0, 0.0, 1.0)

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic images using ControlNet with configurable conditioning inputs.")

    parser.add_argument(
        "--config_yaml_path",
        type=str,
        default='./models/cldm_v15.yaml',
        help="Path to the ControlNet model configuration YAML file (e.g., cldm_v15.yaml)."
    )
    parser.add_argument(
        "--model_weights_path",
        type=str,
        default='./models/EM_best_results.ckpt', # Assuming this is your best-trained model
        help="Path to the ControlNet model weights checkpoint (.ckpt) file."
    )
    parser.add_argument(
        "--mask_model_path",
        type=str,
        default=None,
        help="Path to the ControlNet checkpoint fine-tuned for mask conditioning. Defaults to --model_weights_path if omitted."
    )
    parser.add_argument(
        "--edge_model_path",
        type=str,
        default=None,
        help="Optional single edge ControlNet checkpoint (deprecated, use --edge_model_paths)."
    )
    parser.add_argument(
        "--edge_model_paths",
        type=str,
        nargs='+',
        default=[],
        help="Edge ControlNet checkpoints defined as '<name>=/path/to/model.ckpt'."
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="/mnt/hdd/pascalr/EM-ControlNet/data/EM-Dataset/train_masks",
        help="Path to the directory containing input mask images (e.g., train_masks)."
    )
    parser.add_argument(
        "--edge_dir",
        type=str,
        default=None,
        help="Path to the directory containing precomputed edge images."
    )
    parser.add_argument(
        "--edge_dirs",
        type=str,
        nargs='+',
        default=[],
        help="Edge image directories defined as '<name>=/path/to/edges'. Names must match --edge_model_paths entries."
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="my_synth_data",
        help="Base directory to save generated images and masks. A timestamped folder will be created inside this for each run."
    )
    parser.add_argument(
        "--n_augmentations_per_mask",
        type=int,
        default=1,
        help="Number of synthetic images to generate for each input mask."
    )
    parser.add_argument(
        "--batch_size_per_inference",
        type=int,
        default=1,
        help="Number of samples to generate in a single inference call to the model. This should match num_samples in process func."
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=70,
        help="Number of DDIM sampling steps."
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=2.0,
        help="ControlNet conditioning strength."
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="Classifier-free guidance scale."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed for reproducibility. Use -1 for a random seed each time."
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="DDIM eta parameter (for stochasticity)."
    )
    parser.add_argument(
        "--guess_mode",
        action='store_true', # This will make guess_mode True if the flag is present
        help="Enable guess mode (less strict conditioning)."
    )
    parser.add_argument(
        "--generation_mode",
        type=str,
        default="mask_and_edge",
        choices=["mask_only", "edge_only", "mask_and_edge", "rgba"],
        help="Select which condition branches to use during generation."
    )
    parser.add_argument(
        "--mask_strength",
        type=float,
        default=None,
        help="Optional relative strength applied to the mask ControlNet branch (defaults to 1.0)."
    )
    parser.add_argument(
        "--edge_strengths",
        type=str,
        nargs='+',
        default=[],
        help="Optional relative strengths for edge ControlNet branches formatted as '<name>=<float>'."
    )
    parser.add_argument(
        "--rgba_dir",
        type=str,
        default=None,
        help="Directory containing RGBA npz/png control tensors (required when generation_mode=rgba)."
    )
    parser.add_argument(
        "--rgba_model_path",
        type=str,
        default=None,
        help="Optional checkpoint to use for RGBA ControlNet (defaults to --model_weights_path)."
    )
    parser.add_argument(
        "--skip_missing_edges",
        action='store_true',
        help="If set, skip samples whose edge condition files are missing instead of raising an error."
    )

    args = parser.parse_args()
    return args

# --- Process Function ---
def process(
    batch_idx: int,
    img_idx: int,
    condition_sample: Dict[str, str],
    condition_definitions: Sequence[ConditionDefinition],
    prompt: str,
    a_prompt: str,
    n_prompt: str,
    num_samples_per_inference: int,
    ddim_sampler,
    ddim_steps: int,
    guess_mode: bool,
    strength: float,
    scale: float,
    seed: int,
    eta: float,
    img_save_path: str,
    condition_save_path: str,
    model,
    num_segmentation_classes: int = 3,
) -> List[np.ndarray]:
    """Generate samples conditioned on multiple modalities (mask + edges)."""

    with torch.no_grad():
        prepared_conditions: List[Tuple[ConditionDefinition, np.ndarray, np.ndarray]] = []
        target_hw: Optional[Tuple[int, int]] = None

        for definition in condition_definitions:
            condition_path = condition_sample.get(definition.key)
            if condition_path is None:
                raise KeyError(f"Missing condition path for key '{definition.key}' in condition_sample")

            if definition.modality in ("segmentation", "edge"):
                condition_gray = cv2.imread(condition_path, cv2.IMREAD_GRAYSCALE)
                if condition_gray is None:
                    print(f"Error: Could not read condition image at {condition_path}. Skipping.")
                    return []

                if target_hw is None:
                    target_hw = condition_gray.shape[:2]
                elif condition_gray.shape[:2] != target_hw:
                    interpolation = cv2.INTER_NEAREST if definition.modality == "segmentation" else cv2.INTER_LINEAR
                    condition_gray = cv2.resize(condition_gray, (target_hw[1], target_hw[0]), interpolation=interpolation)

                if definition.modality == "segmentation":
                    H, W = condition_gray.shape
                    rgb_condition = np.zeros((H, W, definition.channels), dtype=np.uint8)
                    max_channel = min(definition.channels, num_segmentation_classes)
                    for class_idx in range(max_channel):
                        rgb_condition[:, :, class_idx] = (condition_gray == class_idx).astype(np.uint8) * 255
                    if definition.channels > max_channel:
                        for channel_idx in range(max_channel, definition.channels):
                            rgb_condition[:, :, channel_idx] = 0
                    preview_img = condition_gray
                else:  # edge
                    rgb_condition = cv2.cvtColor(condition_gray, cv2.COLOR_GRAY2RGB)
                    preview_img = condition_gray

            elif definition.modality == "rgba":
                try:
                    rgba = load_rgba_condition(condition_path)
                except Exception as exc:
                    print(f"Error: Could not read RGBA condition at {condition_path} ({exc}). Skipping.")
                    return []

                if target_hw is None:
                    target_hw = rgba.shape[:2]
                elif rgba.shape[:2] != target_hw:
                    rgba = cv2.resize(rgba, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)

                rgb_condition = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
                if rgb_condition.shape[2] != definition.channels:
                    raise ValueError(
                        f"RGBA condition {condition_path} has {rgb_condition.shape[2]} channels but expected {definition.channels}"
                    )
                
                # Generate edge visualization (white background + colored edges by class)
                # RGB channels contain mask class info, Alpha channel contains edge intensity
                H, W = rgb_condition.shape[:2]
                rgb_only = rgb_condition[:, :, :3]  # R=class0, G=class1, B=class2
                alpha_channel = rgb_condition[:, :, 3]  # Edge intensity
                
                # Create white background
                edge_vis = np.ones((H, W, 3), dtype=np.uint8) * 255
                
                # For each pixel with edge (alpha > threshold), color it based on max RGB channel
                edge_threshold = 10  # Threshold to detect edge pixels
                edge_mask = alpha_channel > edge_threshold
                
                # Get class assignment for edge pixels (argmax of RGB)
                class_assignment = np.argmax(rgb_only, axis=2)
                
                # Color mapping: class0=red, class1=green, class2=blue
                colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
                
                # Apply colors only where there's an edge
                for class_idx in range(3):
                    class_edge_mask = edge_mask & (class_assignment == class_idx)
                    edge_vis[class_edge_mask] = colors[class_idx]
                
                edge_vis_bgr = cv2.cvtColor(edge_vis, cv2.COLOR_RGB2BGR)
                
                # Generate segmentation mask (argmax over RGB channels -> 0, 1, 2)
                segmentation_mask = np.argmax(rgb_only, axis=2).astype(np.uint8)
                
                # Store as tuple: (edge_vis_bgr, segmentation_mask)
                preview_img = (edge_vis_bgr, segmentation_mask)
            else:
                raise ValueError(f"Unsupported condition modality: {definition.modality}")

            prepared_conditions.append((definition, rgb_condition, preview_img))

        if not prepared_conditions:
            print("Warning: No conditioning information prepared. Skipping batch.")
            return []

        control_tensors: List[torch.Tensor] = []
        for _, rgb_condition, _ in prepared_conditions:
            control = torch.from_numpy(rgb_condition.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples_per_inference)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            control_tensors.append(control)

        combined_control = torch.cat(control_tensors, dim=1)
        H_control, W_control = combined_control.shape[-2:]

        current_seed = seed
        if current_seed == -1:
            current_seed = random.randint(0, 65535)
        seed_everything(current_seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [combined_control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples_per_inference)],
        }
        un_cond = {
            "c_concat": None if guess_mode else [combined_control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples_per_inference)],
        }
        shape = (4, H_control // 8, W_control // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples_per_inference,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5
        ).cpu().numpy().clip(0, 255).astype(np.uint8)

        os.makedirs(img_save_path, exist_ok=True)
        os.makedirs(condition_save_path, exist_ok=True)

        for i in range(num_samples_per_inference):
            output_img_filename = f'image_cond{img_idx}_batch{batch_idx}_sample{i}.png'
            img_result_path = os.path.join(img_save_path, output_img_filename)
            cv2.imwrite(img_result_path, cv2.cvtColor(x_samples[i], cv2.COLOR_RGB2BGR))
            print(f'{output_img_filename} has the prompt "{prompt}"')

            for definition, _, condition_preview in prepared_conditions:
                if definition.modality == "rgba" and isinstance(condition_preview, tuple):
                    # RGBA mode: save two types of images
                    edge_vis_bgr, segmentation_mask = condition_preview
                    
                    # Save edge visualization (white background + RGB colors)
                    edge_vis_dir = os.path.join(condition_save_path, "rgba_edge_vis")
                    os.makedirs(edge_vis_dir, exist_ok=True)
                    edge_vis_filename = f'rgba_edge_vis_cond{img_idx}_batch{batch_idx}_sample{i}.png'
                    cv2.imwrite(os.path.join(edge_vis_dir, edge_vis_filename), edge_vis_bgr)
                    
                    # Save segmentation mask (0, 1, 2 grayscale)
                    mask_dir = os.path.join(condition_save_path, "rgba_mask")
                    os.makedirs(mask_dir, exist_ok=True)
                    mask_filename = f'rgba_mask_cond{img_idx}_batch{batch_idx}_sample{i}.png'
                    cv2.imwrite(os.path.join(mask_dir, mask_filename), segmentation_mask)
                else:
                    # Other modalities: save as before
                    condition_dir = os.path.join(condition_save_path, definition.key)
                    os.makedirs(condition_dir, exist_ok=True)
                    condition_filename = f'{definition.key}_cond{img_idx}_batch{batch_idx}_sample{i}.png'
                    condition_result_path = os.path.join(condition_dir, condition_filename)
                    cv2.imwrite(condition_result_path, condition_preview)

        return [x_samples[i] for i in range(num_samples_per_inference)]


def save_generation_state(
    args,
    condition_paths,
    run_output_dir,
    filename='generate_state.json',
    condition_records: Optional[Sequence[Dict[str, str]]] = None,
    primary_condition_key: Optional[str] = None,
):
    """
    Save the generation 'situation' to a JSON file. This includes:
    - the parsed `args` (as a dict)
    - the list of condition image paths that will be processed
    - a small set of simple values from the `config` module

    The file is written to `<run_output_dir>/<filename>` and the
    full path is returned.
    """
    state = {}
    # args -> dict
    try:
        state['args'] = vars(args)
    except Exception:
        # Fallback: try to coerce to dict
        try:
            state['args'] = dict(args)
        except Exception:
            state['args'] = str(args)

    # condition paths (retain legacy key for backward compatibility)
    condition_list = list(condition_paths)
    state['condition_paths'] = condition_list
    if primary_condition_key is not None:
        state['primary_condition_key'] = primary_condition_key

    mask_paths: List[str] = []
    if condition_records:
        mask_paths = [record['mask'] for record in condition_records if 'mask' in record]
    elif primary_condition_key == 'mask':
        mask_paths = condition_list
    state['mask_paths'] = mask_paths
    if condition_records is not None:
        state['condition_records'] = [dict(record) for record in condition_records]

    # capture simple config values (str/int/float/bool/list/dict/None)
    cfg = {}
    for name in dir(config):
        if name.startswith('_'):
            continue
        try:
            value = getattr(config, name)
        except Exception:
            continue
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            # Ensure JSON serializable
            try:
                json.dumps({name: value})
                cfg[name] = value
            except Exception:
                # skip non-serializable
                continue
    state['config'] = cfg

    # timestamp for when this state was saved
    state['saved_at'] = datetime.datetime.now().isoformat()

    os.makedirs(run_output_dir, exist_ok=True)
    filepath = os.path.join(run_output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"Saved generation state to {filepath}")
    return filepath


def load_generation_state(filepath):
    """Load a generation state JSON file saved by `save_generation_state`.
    Returns the parsed dict.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


# --- Main Script Logic ---
def main():
    args = parse_args()

    generation_mode = args.generation_mode
    use_mask = generation_mode in ("mask_only", "mask_and_edge")
    use_edges = generation_mode in ("edge_only", "mask_and_edge")
    use_rgba = generation_mode == "rgba"

    if use_rgba:
        use_mask = False
        use_edges = False

    if not use_mask and args.mask_strength is not None:
        print("Warning: --mask_strength specified but generation mode does not use mask conditioning; ignoring.")
    if use_rgba and args.mask_strength is not None:
        print("Warning: --mask_strength is ignored in rgba generation mode.")

    edge_model_specs = parse_key_value_pairs(args.edge_model_paths, "--edge_model_paths") if args.edge_model_paths else {}
    if args.edge_model_path:
        edge_model_specs.setdefault(DEFAULT_EDGE_KEY, args.edge_model_path)
    edge_dir_specs = parse_key_value_pairs(args.edge_dirs, "--edge_dirs") if args.edge_dirs else {}
    if args.edge_dir:
        edge_dir_specs.setdefault(DEFAULT_EDGE_KEY, args.edge_dir)

    if not use_edges and (edge_model_specs or edge_dir_specs):
        print("Warning: Edge-related arguments provided but generation mode does not use them; ignoring edge branches.")
        edge_model_specs = {}
        edge_dir_specs = {}

    if use_edges and not edge_model_specs:
        raise ValueError("At least one edge ControlNet checkpoint must be provided via --edge_model_paths/--edge_model_path when generation_mode requires edges.")
    if use_edges and not edge_dir_specs:
        raise ValueError("Edge image directories must be provided via --edge_dirs/--edge_dir when generation_mode requires edges.")
    if use_edges and set(edge_model_specs.keys()) != set(edge_dir_specs.keys()):
        raise ValueError("--edge_model_paths and --edge_dirs must refer to the same condition names.")

    edge_dir_specs = {name: os.path.abspath(path) for name, path in edge_dir_specs.items()} if edge_dir_specs else {}
    if use_edges:
        for name, path in edge_dir_specs.items():
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Edge directory for '{name}' not found: {path}")

    mask_dir: Optional[str] = None
    if use_mask:
        if args.mask_dir is None:
            raise ValueError("--mask_dir must be specified when mask conditioning is enabled.")
        mask_dir = os.path.abspath(args.mask_dir)
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    elif args.mask_dir is not None:
        print("Warning: --mask_dir was provided but generation mode does not use masks; ignoring.")

    rgba_dir: Optional[str] = None
    if use_rgba:
        if args.rgba_dir is None:
            raise ValueError("--rgba_dir must be specified when generation_mode=rgba.")
        rgba_dir = os.path.abspath(args.rgba_dir)
        if not os.path.isdir(rgba_dir):
            raise FileNotFoundError(f"RGBA directory not found: {rgba_dir}")
    elif args.rgba_dir is not None:
        print("Warning: --rgba_dir provided but generation mode is not 'rgba'; ignoring.")

    base_model_path = args.model_weights_path or args.mask_model_path
    if use_rgba and args.rgba_model_path:
        base_model_path = args.rgba_model_path
    elif not use_rgba and args.rgba_model_path is not None:
        print("Warning: --rgba_model_path provided but generation mode is not 'rgba'; ignoring.")
    if base_model_path is None:
        raise ValueError("A ControlNet checkpoint path must be provided via --model_weights_path (or --rgba_model_path).")
    base_model_path = os.path.abspath(base_model_path)
    if not os.path.isfile(base_model_path):
        raise FileNotFoundError(f"Base ControlNet checkpoint not found: {base_model_path}")

    edge_strength_map = parse_strength_pairs(args.edge_strengths, "--edge_strengths") if args.edge_strengths else {}
    if use_edges:
        unused_strength_keys = set(edge_strength_map.keys()) - set(edge_model_specs.keys())
        if unused_strength_keys:
            print(
                "Warning: --edge_strengths provided for unknown edge keys: "
                + ", ".join(sorted(unused_strength_keys))
            )
    elif edge_strength_map:
        print("Warning: --edge_strengths specified but generation mode does not use edges; ignoring.")
        edge_strength_map = {}

    model_config_overrides = build_hint_override(4) if use_rgba else None
    model = create_model(args.config_yaml_path, config_overrides=model_config_overrides).cpu()
    model.load_state_dict(load_state_dict(base_model_path, location='cpu'), strict=False)

    condition_definitions: List[ConditionDefinition] = []
    control_modules: List[torch.nn.Module] = []
    condition_strengths: List[float] = []
    channel_splits: List[int] = []
    module_condition_keys: List[str] = []

    if use_mask:
        mask_model_path = args.mask_model_path or base_model_path
        mask_model_path = os.path.abspath(mask_model_path)
        if not os.path.isfile(mask_model_path):
            raise FileNotFoundError(f"Mask ControlNet checkpoint not found: {mask_model_path}")
        mask_model = create_model(args.config_yaml_path).cpu()
        mask_model.load_state_dict(load_state_dict(mask_model_path, location='cpu'), strict=False)
        mask_control = mask_model.control_model
        freeze_module(mask_control)
        mask_control.eval()
        mask_definition = ConditionDefinition(key="mask", modality="segmentation", display_name="mask", channels=3)
        condition_definitions.append(mask_definition)
        control_modules.append(mask_control)
        condition_strengths.append(args.mask_strength if args.mask_strength is not None else 1.0)
        channel_splits.append(mask_definition.channels)
        module_condition_keys.append(mask_definition.key)
        del mask_model

    edge_order: List[str] = []
    if use_edges:
        edge_order = list(edge_model_specs.keys())
        for edge_key in edge_order:
            edge_model_path = os.path.abspath(edge_model_specs[edge_key])
            if not os.path.isfile(edge_model_path):
                raise FileNotFoundError(f"Edge ControlNet checkpoint for '{edge_key}' not found: {edge_model_path}")
            edge_model = create_model(args.config_yaml_path).cpu()
            edge_model.load_state_dict(load_state_dict(edge_model_path, location='cpu'), strict=False)
            edge_control = edge_model.control_model
            freeze_module(edge_control)
            edge_control.eval()
            control_modules.append(edge_control)
            edge_definition = ConditionDefinition(key=edge_key, modality="edge", display_name=edge_key, channels=3)
            condition_definitions.append(edge_definition)
            condition_strengths.append(edge_strength_map.get(edge_key, 1.0))
            channel_splits.append(edge_definition.channels)
            module_condition_keys.append(edge_definition.key)
            del edge_model

    if use_rgba:
        rgba_definition = ConditionDefinition(key="rgba", modality="rgba", display_name="rgba", channels=4)
        condition_definitions.append(rgba_definition)

    if not condition_definitions:
        raise RuntimeError("No conditioning branches configured. Check --generation_mode and related arguments.")

    if control_modules:
        multi_control_model = MultiConditionControlNet(
            modules=control_modules,
            channel_splits=channel_splits,
            condition_keys=module_condition_keys,
            per_condition_scales=condition_strengths,
        )
        model.control_model = multi_control_model
        condition_scale_map = multi_control_model.get_condition_scales()
    else:
        multi_control_model = None
        condition_scale_map = {definition.key: args.strength for definition in condition_definitions}

    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    condition_branches = ", ".join(
        f"{definition.key}(strength={condition_scale_map[definition.key]:.2f})"
        for definition in condition_definitions
    )
    print(f"Configured condition branches: {condition_branches}")

    condition_file_maps: Dict[str, Dict[str, str]] = {}
    if use_mask and mask_dir is not None:
        mask_files = {
            file: os.path.join(mask_dir, file)
            for file in os.listdir(mask_dir)
            if file.lower().endswith('.png')
        }
        if not mask_files:
            raise ValueError(f"No mask images found in {mask_dir}")
        condition_file_maps['mask'] = mask_files

    if use_edges:
        for edge_key in edge_order:
            edge_dir = edge_dir_specs[edge_key]
            edge_files = {
                file: os.path.join(edge_dir, file)
                for file in os.listdir(edge_dir)
                if file.lower().endswith('.png')
            }
            if not edge_files:
                raise ValueError(f"No edge images found for '{edge_key}' in {edge_dir}")
            condition_file_maps[edge_key] = edge_files

    if use_rgba and rgba_dir is not None:
        rgba_files = {
            file: os.path.join(rgba_dir, file)
            for file in os.listdir(rgba_dir)
            if file.lower().endswith(('.npz', '.png'))
        }
        if not rgba_files:
            raise ValueError(f"No RGBA tensors found in {rgba_dir}")
        condition_file_maps['rgba'] = rgba_files

    if use_rgba:
        primary_condition_key = 'rgba'
    else:
        primary_condition_key = 'mask' if use_mask else (edge_order[0] if edge_order else None)
    if primary_condition_key is None:
        raise RuntimeError("Unable to determine primary condition key for dataset enumeration.")

    anchor_map = condition_file_maps.get(primary_condition_key, {})
    if not anchor_map:
        raise ValueError(f"No condition images found for primary key '{primary_condition_key}'.")

    anchor_names = sorted(anchor_map.keys())
    condition_samples: List[Dict[str, str]] = []
    skipped_due_to_missing = 0
    for name in anchor_names:
        record: Dict[str, str] = {}
        skip_sample = False
        for definition in condition_definitions:
            key = definition.key
            file_map = condition_file_maps.get(key, {})
            candidate_path = file_map.get(name)
            if candidate_path is None:
                if key == 'mask':
                    expected_root = mask_dir
                elif key in edge_dir_specs:
                    expected_root = edge_dir_specs.get(key, '')
                elif key == 'rgba':
                    expected_root = rgba_dir
                else:
                    expected_root = ''
                expected_path = os.path.join(expected_root, name) if expected_root else name
                message = f"Missing condition file for '{key}' with base name {name}: expected {expected_path}"
                if definition.modality == "edge" and args.skip_missing_edges:
                    print(f"Warning: {message}. Skipping this sample.")
                    skip_sample = True
                    break
                raise FileNotFoundError(message)
            record[key] = candidate_path
        if skip_sample:
            skipped_due_to_missing += 1
            continue
        condition_samples.append(record)

    if not condition_samples:
        print("No valid condition samples found after applying the configured requirements. Nothing to generate.")
        return

    if skipped_due_to_missing:
        print(f"Skipped {skipped_due_to_missing} samples due to missing edge conditions.")

    primary_condition_paths = [record[primary_condition_key] for record in condition_samples]

    prompts = [
        "A realistic scientific image of a cell taken from an electron microscope, photorealistic style, very detailed, black, white, detail",
        "A realistic scientific image of a cell taken from an electron microscope, photorealistic style, very detailed, black, white, detail, rotated by 90 or 180 degrees",
        "A realistic scientific image of a cell taken from an electron microscope, photorealistic style, very detailed, black, white, detail, Retake",
        "A realistic scientific image of a cell taken from an electron microscope, photorealistic style, very detailed, black, white, detail, scaled brightness",
        "A realistic scientific image of a cell taken from an electron microscope, photorealistic style, very detailed, black, white, detail, noise",
        "A realistic scientific image of a fat cell taken from an electron microscope, photorealistic style, very detailed, black, white, detail, Retake",
        "A realistic scientific cell image by an electron microscope, photorealistic style, very detailed, black, white, detail, Retake",
        "Electron microscopy, Fat call, Realistic, Science, Black and white, very detailed",
        "A realistic scientific image of a cell taken from an electron microscope, photorealistic style, very detailed, black, white, detail, Fine structure",
        "Scientific image of a cell taken from an electron microscope, photorealistic style, very detailed, black, white, detail",
        "A realistic scientific image of a cell taken from an electron microscope, photorealistic style, very detailed, black, white, model",
        "A realistic scientific image of a cell taken from an electron microscope, photorealistic style, very detailed, black, white, detail, Cross section",
        "Fat cell, Realistic, Black and white, detailed, Photorealistic",
        "Scientific image of a fat cell taken from an electron microscope, photorealistic style, very detailed, black, white, detail",
        "A realistic scientific image of a cell taken from an electron microscope, photorealistic style, very detailed",
        "An impressive black-and-white image featuring a detailed, photorealistic depiction of a fat cell with subtle contours and textures",
        "Black and white image showing an isolated fat cell in captivating photorealism, with every detail from the lipids to the cell membranes artfully captured",
        "A masterful black-and-white composition with a solitary fat cell, sharp contrasts, and meticulous representation of cell structure.",
        "A poignant black-and-white image displays a fat cell in unparalleled intricacy, with every detail from intracellular lipids to cell nuclei portrayed with impressive accuracy.",
        "Impressive black-and-white representation of a fat cell, captivating with its realistic portrayal of subtle structures.",
        "A fat cell with remarkable precision, shaded areas, and precise lines resembling a microscopic capture, black and white, photorealistic",
        "An outstanding black-and-white image captures the beauty of a fat cell with perfect detail, masterfully depicted from fine fat droplets to delicate cell structures",
        "High-contrast black-and-white image portraying a fat cell with remarkable accuracy, accentuating subtle details from cell membranes to lipids",
        "A fascinating black-and-white image showcasing a single fat cell with astonishing detail precision, creating the impression of delving into the microcosm of cells",
        "An impressive black-and-white image presents a fat cell in unparalleled photorealism, with fine shades and precise depiction of cell components adding remarkable depth",
        "In this mesmerizing black-and-white composition, a meticulously rendered fat cell captures attention with its intricate details and compelling realism",
        "A captivating black-and-white depiction of a fat cell showcases an extraordinary level of detail, from the intricate lipid droplets to the delicate cellular structures",
        "This striking black-and-white image offers a detailed and photorealistic portrayal of a fat cell, with its subtle nuances and textures meticulously captured",
        "An evocative black-and-white composition highlights the beauty of a fat cell, revealing intricate details from lipid droplets to the nuanced cellular membranes",
        "With stunning precision and a black-and-white palette, this image captures the essence of a fat cell, emphasizing its complexity and beauty through meticulous representation",
        "A captivating microscopic image showcasing the intricate details of cellular organelles and structures, in black and white, with exceptional clarity",
        "Mesmerizing black-and-white composition featuring a high-resolution image of a cell, highlighting the delicate interplay of shadows and light",
        "A stunning black-and-white portrayal of cellular diversity, showcasing unique patterns and shapes in high resolution",
        "A finely detailed black-and-white composition offering a glimpse into the microscopic world of cells, revealing the beauty of intricate structures",
        "Captivating black-and-white image portraying cellular membranes and structures with exceptional clarity and artistic finesse",
        "An exceptional black-and-white visualization of cellular intricacies, highlighting the beauty of fine structures and detailed textures",
        "Intricate black-and-white composition showcasing the elegance of cellular architecture, with a focus on fine details and contours",
    ]

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_suffix = f"p{os.getpid()}_{uuid.uuid4().hex[:8]}"
    run_id = f"{timestamp_str}_{unique_suffix}"
    run_output_dir = os.path.join(args.output_base_dir, f"run_{run_id}")
    os.makedirs(run_output_dir, exist_ok=True)

    try:
        save_generation_state(
            args,
            primary_condition_paths,
            run_output_dir,
            condition_records=condition_samples,
            primary_condition_key=primary_condition_key,
        )
    except Exception as exc:
        print(f"Warning: failed to save generation state: {exc}")

    img_save_path_base = os.path.join(run_output_dir, "images")
    condition_save_path_base = os.path.join(run_output_dir, "conditions")

    print(f"Saving generated images to: {img_save_path_base}")
    print(f"Saving corresponding condition maps to: {condition_save_path_base}")
    print(f"Total condition sets to process: {len(condition_samples)}")

    for img_idx, condition_record in enumerate(condition_samples):
        condition_details = ", ".join(
            f"{definition.key}:{os.path.basename(condition_record.get(definition.key, ''))}"
            for definition in condition_definitions
        )
        print(f"\n--- Processing condition {img_idx + 1}/{len(condition_samples)}: {condition_details} ---")

        generated_so_far = 0
        batch_count_for_condition = 0

        while generated_so_far < args.n_augmentations_per_mask:
            current_num_samples_to_generate = min(
                args.batch_size_per_inference,
                args.n_augmentations_per_mask - generated_so_far,
            )
            if current_num_samples_to_generate <= 0:
                break

            generated_so_far += current_num_samples_to_generate
            batch_count_for_condition += 1

            prompt = np.random.choice(prompts)
            a_prompt = ""
            n_prompt = ""

            process(
                batch_idx=batch_count_for_condition,
                img_idx=img_idx,
                condition_sample=condition_record,
                condition_definitions=condition_definitions,
                prompt=prompt,
                a_prompt=a_prompt,
                n_prompt=n_prompt,
                num_samples_per_inference=current_num_samples_to_generate,
                ddim_sampler=ddim_sampler,
                ddim_steps=args.ddim_steps,
                guess_mode=args.guess_mode,
                strength=args.strength,
                scale=args.scale,
                seed=args.seed,
                eta=args.eta,
                img_save_path=img_save_path_base,
                condition_save_path=condition_save_path_base,
                model=model,
            )

            print(
                f"Generated {current_num_samples_to_generate} images for set {img_idx + 1} (Batch {batch_count_for_condition})"
            )

    print("\n--- All synthetic images generated! ---")


if __name__ == "__main__":
    main()