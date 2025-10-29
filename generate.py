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

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic images using ControlNet based on masks.")

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
        "--mask_dir",
        type=str,
        default="/mnt/hdd/pascalr/EM-ControlNet/data/EM-Dataset/train_masks",
        help="Path to the directory containing input mask images (e.g., train_masks)."
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="generated_images",
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

    args = parser.parse_args()
    return args

# --- Process Function ---
def process(
    batch_idx, img_idx, mask_path, prompt, a_prompt, n_prompt,
    num_samples_per_inference, ddim_sampler, ddim_steps, guess_mode, strength, scale, seed, eta,
    img_save_path, mask_save_path, model # Pass model to process
):
    """
    Processes a single mask to generate synthetic images using ControlNet.
    """
    with torch.no_grad():
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Read as grayscale
        if raw_mask is None:
            print(f"Error: Could not read mask image at {mask_path}. Skipping.")
            return []

        # Ensure raw_mask is 2D (H, W) for class mapping, then create RGB
        if raw_mask.ndim == 3: # If somehow read as 3 channel, take first one
            raw_mask = raw_mask[:, :, 0]

        H, W = raw_mask.shape
        rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)

        # Assuming raw_mask contains pixel values representing class_idx (0, 1, 2 for 3 classes)
        # Class 0 -> Red, Class 1 -> Green, Class 2 -> Blue
        # This encoding maps distinct pixel values to distinct RGB channels.
        # If your classes are 1, 2, 3, adjust `raw_mask == class_idx` accordingly.
        for class_idx in range(3): # Iterate for 3 classes
            rgb_mask[:, :, class_idx] = (raw_mask == class_idx).astype(np.uint8) * 255

        control = torch.from_numpy(rgb_mask.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples_per_inference)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        H_control, W_control = control.shape[-2:]

        current_seed = seed
        if current_seed == -1:
            current_seed = random.randint(0, 65535)
        seed_everything(current_seed)

        if config.save_memory: # Assuming 'config' module is still available and has save_memory
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples_per_inference)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples_per_inference)]}
        shape = (4, H_control // 8, W_control // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        # Apply control scales
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples_per_inference,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        # Create output directories if they don't exist
        os.makedirs(img_save_path, exist_ok=True)
        os.makedirs(mask_save_path, exist_ok=True)


        for i in range(num_samples_per_inference):
            output_img_filename = f'image_mask{img_idx}_batch{batch_idx}_sample{i}.png'
            # Save the *original* single-channel mask used for this generation
            output_mask_filename = f'mask_mask{img_idx}_batch{batch_idx}_sample{i}.png'

            img_result_path = os.path.join(img_save_path, output_img_filename)
            mask_result_path = os.path.join(mask_save_path, output_mask_filename)

            cv2.imwrite(img_result_path, cv2.cvtColor(x_samples[i], cv2.COLOR_RGB2BGR))
            print(f'{output_img_filename} has the prompt "{prompt}"')

            # Save the raw_mask (single channel) that corresponds to the generated image
            cv2.imwrite(mask_result_path, raw_mask) # raw_mask is (H, W) for class indices

        # results is a list of generated images (numpy arrays)
        # The original code returned [rgb_mask] + results, but rgb_mask is the input, not output
        # Usually, `results` for generative tasks refers to the generated samples.
        return [x_samples[i] for i in range(num_samples_per_inference)]


def save_generation_state(args, mask_paths, run_output_dir, filename='generate_state.json'):
    """
    Save the generation 'situation' to a JSON file. This includes:
    - the parsed `args` (as a dict)
    - the list of mask paths that will be processed
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

    # mask paths
    state['mask_paths'] = list(mask_paths)

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

    # --- Initialize Model (using args) ---
    model = create_model(args.config_yaml_path).cpu()
    model.load_state_dict(load_state_dict(args.model_weights_path, location='cuda'), strict=False) # Load to CUDA
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    # --- Prepare Mask Paths ---
    mask_paths = [os.path.join(args.mask_dir, file) for file in os.listdir(args.mask_dir) if file.endswith(".png")]
    mask_paths.sort() # Ensure consistent order

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

    # Get current timestamp for the entire run's output folder
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_base_dir, f"run_{timestamp_str}")
    os.makedirs(run_output_dir, exist_ok=True)
    # Save the generation state (args, mask list, small config snapshot)
    try:
        save_generation_state(args, mask_paths, run_output_dir)
    except Exception as e:
        print(f"Warning: failed to save generation state: {e}")

    img_save_path_base = os.path.join(run_output_dir, "images")
    mask_save_path_base = os.path.join(run_output_dir, "masks")

    print(f"Saving generated images to: {img_save_path_base}")
    print(f"Saving corresponding masks to: {mask_save_path_base}")
    print(f"Total masks to process: {len(mask_paths)}")

    for img_idx, mask_path in enumerate(mask_paths):
        print(f"\n--- Processing mask {img_idx + 1}/{len(mask_paths)}: {os.path.basename(mask_path)} ---")

        i = 0 # Counter for augmentations generated for the current mask
        batch_count_for_mask = 0 # Counter for inference batches for the current mask

        while i < args.n_augmentations_per_mask:
            # Determine how many samples to generate in the current inference call
            current_num_samples_to_generate = min(args.batch_size_per_inference, args.n_augmentations_per_mask - i)

            if current_num_samples_to_generate <= 0:
                break # No more augmentations needed for this mask

            i += current_num_samples_to_generate
            batch_count_for_mask += 1

            prompt = np.random.choice(prompts) # Random prompt for each batch
            a_prompt = ""
            n_prompt = ""

            results = process(
                batch_idx=batch_count_for_mask, # Pass current batch index for this mask
                img_idx=img_idx,
                mask_path=mask_path,
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
                mask_save_path=mask_save_path_base,
                model=model # Pass the model object
            )

            print(f"Generated {current_num_samples_to_generate} images for {os.path.basename(mask_path)} (Batch {batch_count_for_mask})")

    print("\n--- All synthetic images generated! ---")

if __name__ == "__main__":
    main()