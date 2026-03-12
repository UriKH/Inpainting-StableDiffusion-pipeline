# Stable Diffusion Inpainting: Implementation and Improvements

This repository contains the code for our Deep Learning Final Project, tackling the task of Diffusion Inpainting Implementation and Improvement. 
The project implements an inpainting pipeline using `stable-diffusion-2-base`, identifies issues with the vanilla approach, and iteratively introduces improvements across 14 pipeline versions to fill in missing parts of an image with new or missing content.

## Features and Pipeline Enhancements
The repository includes a vanilla baseline and progressively advanced pipelines located in the `pipelines/` directory. Key enhancements include:
* **Vanilla Pipeline:** A standard DDPM-based prompt based inpainting implementation using the `stable-diffusion-2-base` architecture.
* **Iterative Improvements (V1-V14):** Advanced masking, attention manipulation, and blending techniques, including:
  * **Mask Preprocessing:** Dilation, feathering, and negative prompting (`v5_improved.py`).
  * **Attention Masking:** Custom processors for soft masking in Cross-Attention (`v6_improved.py`, `v7_improved.py`) and Self-Attention (`v8_improved.py`, `v9_improved.py`).
  * **Time-Travel Resampling:** Dynamic and RePaint-style scheduling to improve structural coherence (`v10_improved.py`).
  * **FreeU Integration:** Prioritizing semantic structure over high-frequency background matching (`v11_improved.py`).
  * **Organic Masking:** Time-dependent Perlin noise masks to prevent visible seam lines (`v12_improved.py`).
  * **Dynamic Mask Blending:** Annealing mask blur over timesteps to ensure perfect background reconstruction (`v13_improved.py`, `v14_improved.py`).

## Setup and Requirements
To run the code, you will need the dependencies listed in `requierments.txt`.
```bash
pip install -r requierments.txt
```
The primary required packages include `diffusers`, `transformers`, `torchmetrics[image]`, `opencv-python`, `scipy`, and `pandas`.

## Usage Guide

### 1. Interactive Masking Tool
Create custom masks and crop your own images using the interactive OpenCV tool.
```bash
python masking_tool.py <path_to_image> -o <output_directory>
```
* **Instructions:** Click and drag to paint the mask, press `c` to clear the canvas, `s` to save the final image and mask, or `q` to quit without saving.

### 2. Generating Inpaintings
Run the `data_generator.py` script to generate inpaintings on the COCO dataset using specific pipeline versions.
```bash
python data_generator.py --version <version_number> <input_args> <output_args>
```
* Use `--version 0` for the vanilla pipeline, or `1`-`14` for the improved versions.
* Additional flags allow tweaking parameters like `--use_freeu`, `--sa_dilation_threshold`, `--use_negative_prompt`, and `--use_dynamic_schedule`.
* To view all flags run:
  ```bash
  python data_generator.py -h
  ```

### 3. Evaluating Metrics
Calculate quantitative metrics to evaluate your improvements. Supported metrics include FID, SSIM, LPIPS, CLIP Score, MSE, PSNR, DISTS, and DINOv2 distance.
```bash
python create_metrics_json.py <input_args> <output_args>
```
The script will compute the metrics and output a `metrics.json` file in the specified output directory.

### 4. Graphs
* Run `metric_graphs.py` to create bar graphs for each metric:
  ```bash
  python metrics_graph.py -j <input_dir>
  ```
  Where `input_dir` is the parent folder in which all the results folders are in.
* To create relevant graphs maching the results like heatmaps and radar charts use `draw_smart_visuals.py`:
  ```bash
  python draw_smart_visuals.py -j <input_dir> -c <res_dir 1> ...
  ```
  Where `res_dir`-s are list of result directories to create radar chart from the metrics stored in them.
