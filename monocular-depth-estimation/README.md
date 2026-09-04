# Monocular Depth Estimation

This module is part of the **Spectra3D** pipeline. It handles **Stage 2: Depth Estimation** — converting a single 2D RGB image into a per-pixel depth map, which downstream modules (point cloud generation, 3D Gaussian Splatting) use to build the 3D scene.

## What it does

Given an RGB image, `depth_inference.py`:
1. Loads a pre-trained **Depth Anything V2 (Small)** model from Hugging Face
2. Runs the image through the model to estimate depth at every pixel
3. Normalizes the depth values to a `0.0 – 1.0` range
4. Saves two outputs:
   - a raw `.npy` array (for downstream 3D math)
   - a grayscale `.png` (for quick visual inspection)

## Folder structure

```
monocular-depth-estimation/
├── depth_inference.py     # main script
├── requirements.txt       # Python dependencies
├── README.md               # this file
├── .gitignore
├── inputs/                 # sample/test images go here
│   └── frame_001.jpg
└── outputs/                # generated depth maps (gitignored)
    ├── frame_001_depth.npy
    └── frame_001_depth_viz.png
```

## Setup

```bash
cd monocular-depth-estimation
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

## Usage

Place an input image inside `inputs/`, then run:

```bash
python depth_inference.py --image inputs/frame_001.jpg
```

Outputs are saved to `outputs/`:
- `frame_001_depth.npy` — raw float32 depth array
- `frame_001_depth_viz.png` — grayscale depth visualization

## Model

- **Model:** Depth Anything V2 (Small variant, `vits`)
- **Source:** [`depth-anything/Depth-Anything-V2-Small-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) via Hugging Face `transformers`
- **Device:** Uses CUDA (GPU) if available, otherwise falls back to CPU

## Status

🚧 Folder structure and script scaffold in place. Model integration and testing in progress.

## Next steps

- [ ] Test on sample video frames
- [ ] Hand off `.npy` outputs to the point cloud generation module
- [ ] Benchmark inference speed for real-time use
