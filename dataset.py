import os
import json
from pathlib import Path
import cv2
import numpy as np
import glob

from torch.utils.data import Dataset
# from torchvision import transforms
import torchvision.transforms.v2 as transforms
import re

def is_image_not_augmented(filename):
    # pattern = re.compile(r'^image_\d+\.png$')
    pattern = re.compile(r'^image_(\d+|\d+_\d{1,2})\.png$')
    return bool(pattern.match(filename))

class MyDataset(Dataset):
    def __init__(self, image_dir, condition_dir, augment=True, condition_type="segmentation", rgba_alpha_scale: float = 1.0):
        if image_dir is None:
            raise ValueError("image_dir must be provided.")
        if condition_dir is None:
            raise ValueError("condition_dir must be provided.")

        self.imagePaths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        if condition_type == "rgba":
            rgba_npz = sorted(glob.glob(os.path.join(condition_dir, "*.npz")))
            rgba_png = sorted(glob.glob(os.path.join(condition_dir, "*.png")))
            stem_to_path = {}
            for path in rgba_png:
                stem_to_path.setdefault(Path(path).stem, path)
            for path in rgba_npz:
                stem_to_path[Path(path).stem] = path
            self.conditionPaths = [stem_to_path[stem] for stem in sorted(stem_to_path.keys())]
        else:
            self.conditionPaths = sorted(glob.glob(os.path.join(condition_dir, "*.png")))
        self.condition_type = condition_type
        self.rgba_alpha_scale = float(rgba_alpha_scale)

        if len(self.imagePaths) == 0:
            raise ValueError(f"No training images found in {image_dir}.")
        if len(self.conditionPaths) == 0:
            raise ValueError(f"No condition images found in {condition_dir}.")
        if len(self.imagePaths) != len(self.conditionPaths):
            raise ValueError(
                f"Image count ({len(self.imagePaths)}) and condition count ({len(self.conditionPaths)}) differ."
            )

        print(f"Using {len(self.imagePaths)} samples with condition type '{self.condition_type}'.")

        if augment:
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=(0, 360)),
                transforms.RandomResizedCrop(size=(512, 512), scale=(0.08, 1.0)),
                # transforms.ColorJitter(brightness=0.4),
            ])
        else:
            self.transforms = None

        self.prompts = [
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

    def __len__(self):
        return len(self.imagePaths)

    def resizePrepare(self, image, size):
        h, w = image.shape[:2]
        aspect_ratio = float(size[1]) / size[0]

        h, w = image.shape[:2]
        h_start = (h - size[0]) // 2
        w_start = (w - size[1]) // 2

        cropped_image = image[h_start:h_start + size[0], w_start:w_start + size[1]]

        return cropped_image

    def __getitem__(self, idx):
        # item = self.data[idx]
        imagePath = self.imagePaths[idx]
        conditionPath = self.conditionPaths[idx]

        if self.condition_type == "segmentation":
            mask = cv2.imread(conditionPath, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Failed to read segmentation mask: {conditionPath}")
            rgb_source = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            for class_idx in range(3):
                rgb_source[:, :, class_idx] = (mask == class_idx).astype(np.uint8) * 255
            source_img = rgb_source
        elif self.condition_type == "edge":
            edge = cv2.imread(conditionPath, cv2.IMREAD_GRAYSCALE)
            if edge is None:
                raise FileNotFoundError(f"Failed to read edge image: {conditionPath}")
            source_img = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
        elif self.condition_type == "rgba":
            source_img = self._load_rgba_uint8(conditionPath)
        else:
            raise ValueError(f"Unsupported condition_type: {self.condition_type}")

        target = cv2.imread(imagePath) # 512,512,3 ; max = 244 min = 0

        source = self.resizePrepare(source_img, (512, 512))
        target = self.resizePrepare(target, (512, 512))

        if self.transforms is not None:
            # if random.random() > 0.5:
            target, source = self.transforms(target, source)
            target = np.array(target)
            source = np.array(source)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        if self.condition_type == "rgba" and self.rgba_alpha_scale != 1.0 and source.shape[-1] > 3:
            source_alpha = source[:, :, 3]
            source[:, :, 3] = np.clip(source_alpha * self.rgba_alpha_scale, 0.0, 1.0)

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        prompt = np.random.choice(self.prompts)

        return dict(jpg=target, txt=prompt, hint=source)

    def _load_rgba_uint8(self, condition_path):
        if condition_path.lower().endswith(".npz"):
            data = np.load(condition_path)
            if "rgba" not in data:
                raise KeyError(f"NPZ file {condition_path} missing 'rgba' key")
            rgba = data["rgba"].astype(np.float32)
            value_range = None
            if "meta" in data:
                try:
                    meta_raw = data["meta"]
                    if isinstance(meta_raw, np.ndarray):
                        meta_raw = meta_raw.tolist()
                    if isinstance(meta_raw, bytes):
                        meta_raw = meta_raw.decode("utf-8")
                    meta_dict = json.loads(str(meta_raw))
                    value_range = meta_dict.get("value_range")
                except Exception:
                    value_range = None
            if value_range == "neg_one_to_one":
                rgba = (rgba + 1.0) * 127.5
            elif value_range == "zero_one":
                rgba = rgba * 255.0
            elif rgba.max() <= 1.0:
                rgba = rgba * 255.0
            rgba_uint8 = np.clip(rgba, 0, 255).astype(np.uint8)
            return rgba_uint8
        rgba = cv2.imread(condition_path, cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise FileNotFoundError(f"Failed to read RGBA condition: {condition_path}")
        if rgba.ndim == 2:
            rgba = cv2.cvtColor(rgba, cv2.COLOR_GRAY2RGBA)
        if rgba.shape[2] == 3:
            alpha = np.zeros_like(rgba[:, :, :1])
            rgba = np.concatenate([rgba, alpha], axis=-1)
        elif rgba.shape[2] > 4:
            rgba = rgba[:, :, :4]
        return rgba

