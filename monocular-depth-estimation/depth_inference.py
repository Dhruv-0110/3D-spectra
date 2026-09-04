"""
depth_inference.py
-------------------
Spectra3D - Stage 2: Monocular Depth Estimation

Loads a pre-trained Depth Anything V2 (Small) model and generates a
per-pixel depth map for an input RGB image.

Usage:
    python depth_inference.py --image test.jpg

Outputs (next to the input image, or in --output-dir if given):
    <basename>_depth.npy       -> raw float32 depth array, normalized 0.0-1.0
    <basename>_depth_viz.png   -> grayscale visualization of the depth map

Dependencies:
    pip install torch torchvision transformers opencv-python numpy matplotlib
"""

import argparse
import os

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib.pyplot as plt


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def get_device() -> str:
    """Prefer CUDA as specified, fall back to CPU if no GPU is available."""
    if torch.cuda.is_available():
        return "cuda"
    print("[WARN] CUDA GPU not found — falling back to CPU. "
          "Inference will be slower.")
    return "cpu"


def load_model(device: str):
    """Load Depth Anything V2 Small + its matching image processor."""
    print(f"[INFO] Loading model '{MODEL_ID}' on device: {device}")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return processor, model


def read_image(image_path: str) -> np.ndarray:
    """Read an image from disk as RGB (OpenCV loads BGR by default)."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise ValueError(f"Could not decode image: {image_path}")

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image


def run_inference(processor, model, device: str, rgb_image: np.ndarray) -> np.ndarray:
    """
    Preprocess the image, run it through the model, and return a raw
    depth map resized back to the original image's height/width.
    """
    original_h, original_w = rgb_image.shape[:2]

    inputs = processor(images=rgb_image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth  # shape: (1, H', W')

    # Resize the low-res model output back to the original image size
    depth_map = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(original_h, original_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    return depth_map.cpu().numpy()


def normalize_depth(raw_depth: np.ndarray) -> np.ndarray:
    """Scale a raw depth array to the 0.0-1.0 range."""
    depth_min = raw_depth.min()
    depth_max = raw_depth.max()

    if depth_max - depth_min < 1e-8:
        # Avoid divide-by-zero on a degenerate (flat) depth map
        return np.zeros_like(raw_depth, dtype=np.float32)

    normalized = (raw_depth - depth_min) / (depth_max - depth_min)
    return normalized.astype(np.float32)


def save_outputs(normalized_depth: np.ndarray, image_path: str, output_dir: str) -> tuple[str, str]:
    """Save the normalized depth as a .npy array and a grayscale .png."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    npy_path = os.path.join(output_dir, f"{base_name}_depth.npy")
    png_path = os.path.join(output_dir, f"{base_name}_depth_viz.png")

    # Raw float array — used later for 3D math (point cloud generation)
    np.save(npy_path, normalized_depth)

    # Grayscale image — used for quick visual inspection
    plt.imsave(png_path, normalized_depth, cmap="gray")

    return npy_path, png_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a per-pixel depth map from an RGB image using Depth Anything V2 (Small)."
    )
    parser.add_argument("--image", required=True, help="Path to the input RGB image (e.g. frame_001.jpg)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save outputs in. Defaults to the input image's directory.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.image)) or "."

    device = get_device()
    processor, model = load_model(device)

    print(f"[INFO] Reading image: {args.image}")
    rgb_image = read_image(args.image)

    print("[INFO] Running depth inference...")
    raw_depth = run_inference(processor, model, device, rgb_image)

    print("[INFO] Normalizing depth to 0.0-1.0 range...")
    normalized_depth = normalize_depth(raw_depth)

    npy_path, png_path = save_outputs(normalized_depth, args.image, output_dir)

    print(f"[DONE] Saved raw depth array -> {npy_path}")
    print(f"[DONE] Saved depth visualization -> {png_path}")


if __name__ == "__main__":
    main()