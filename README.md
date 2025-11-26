---
title: Model Optimization Lab
emoji: 😻
colorFrom: pink
colorTo: gray
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
---

# Model Optimization Lab

Interactive Gradio playground for comparing pruning and quantization on both ImageNet-classification and ADE20K-segmentation models. Upload any image and observe how latency, confidence, model size, and segmentation quality change when applying different compression recipes. Pretrained weights are loaded by default; set `MODEL_OPT_PRETRAINED=0` if you want random initialization for experimentation.

## Features
- **Classification Tasks**: Baseline FP32 inference using cached backbones (ResNet-50, MobileNetV3, EfficientNet-B0, ConvNeXt-Tiny, ViT-B/16, RegNetY-016, EfficientNet-Lite0).
- **Segmentation Tasks**: Pretrained ADE20K models (SegFormer B0/B4, DPT Large, UPerNet ConvNeXt-Tiny) with 150-class semantic segmentation.
- **Pruning tabs**: Structured/unstructured pruning with configurable sparsity and comprehensive size/latency comparison for both classification and segmentation.
- **Quantization tabs**: Dynamic, weight-only INT8, and FP16 passes with CPU-safe fallbacks for unsupported kernels, available for both task types.
- **Visual Comparisons**: 
  - Classification: Automated metric tables and Top-5 bar charts to visualize confidence shifts.
  - Segmentation: Image sliders for overlay/mask comparisons, class distribution tables, and mask agreement metrics.
- **Export Options**: TorchScript, ONNX, JSON reports, and state dictionaries for all optimization variants.
- Lightweight CLI mode for quick experiments without launching the UI.

## Requirements
- Python 3.9+
- PyTorch with CPU support (GPU optional but recommended for FP16 experiments).
- The packages listed in `requirements.txt`:
  - `torch`, `torchvision` - Core PyTorch framework
  - `timm` - Classification model architectures
  - `segmentation-models-pytorch` - Segmentation model architectures
  - `albumentations` - Image preprocessing for segmentation models
  - `gradio` - Web UI framework
  - `pandas`, `matplotlib`, `numpy`, `pillow` - Data processing and visualization

## Quick Start
1. Clone the repository:
	```bash
	git clone https://github.com/shriarul5273/model-optimization-lab.git
	cd model-optimization-lab
	```
2. Create and activate a virtual environment (optional but recommended).
3. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
4. Launch the Gradio app:
	```bash
	python app.py
	```
5. Open the local Gradio URL (printed in the terminal) in your browser.

## Using the App
1. **Upload an image** or pick one of the provided examples (ImageNet samples for classification, ADE20K validation images for segmentation).
2. Choose the **Base Model** dropdown:
   - **Classification**: ResNet-50, MobileNetV3-Large, EfficientNet-B0, ConvNeXt-Tiny, ViT-B/16, RegNetY-016, EfficientNet-Lite0
   - **Segmentation**: SegFormer B0/B4 (ADE20K 512x512), DPT Large (ADE20K), UPerNet ConvNeXt-Tiny (ADE20K)
3. Pick a **Hardware Preset** or keep `custom`:
	- Edge CPU — CPU, channels-last off, dynamic quantization, 30% pruning.
	- Datacenter GPU — CUDA, channels-last on, `torch.compile`, FP16 quantization, 20% pruning.
	- Apple MPS — MPS, FP16 quantization, 20% pruning.
4. Select a tab (Pruning-Classification, Quantization-Classification, Pruning-Segmentation, or Quantization-Segmentation), configure options, then click **Run**.

### Pruning tab options (Classification & Segmentation)
- `Pruning Method`: `structured` (LN-structured) or `unstructured` (L1). Applied to Conv2d weights before export.
- `Pruning Amount`: 0.1–0.9 sparsity. Higher numbers zero more weights; latency impact depends on kernel support.
- `Device`: `auto` picks CUDA → MPS → CPU. Channels-last is only honored on CUDA.
- `Channels-last input (CUDA)`: Converts tensors to channels-last for better CUDA kernel throughput.
- `Mixed precision (AMP)`: Enables CUDA autocast for FP16/FP32 mixes.
- `Torch compile (PyTorch 2)`: Wraps the pruned model in `torch.compile` when available.
- **Exports**:
  - Classification: `pruned_model.ts`, `pruned_model.onnx`, `pruned_report.json`, `pruned_state_dict.pth`
  - Segmentation: `pruned_seg_model.ts`, `pruned_seg_model.onnx`, `pruned_seg_report.json`, `pruned_seg_state_dict.pth`
- **Outputs**:
  - Classification: Comparison metrics, Top-5 bar chart, per-layer sparsity table, download list
  - Segmentation: Comparison metrics, class distribution table, overlay/mask sliders, per-layer sparsity table, download list

### Quantization tab options (Classification & Segmentation)
- `Quantization Type`: `dynamic`/`weight_only` (INT8 linear layers on CPU), or `fp16` (casts model to half precision).
- `Device`: `auto` picks CUDA → MPS → CPU; dynamic/weight-only runs force CPU execution for kernel support.
- `Channels-last input (CUDA)`: Same as pruning; ignored on CPU.
- `Mixed precision (AMP)`: Applies CUDA autocast to the quantized forward pass.
- `Torch compile (PyTorch 2)`: Compiles the quantized model when available.
- **Exports**:
  - Classification: `quantized_model.ts`, `quantized_model.onnx`, `quant_report.json`, `quantized_state_dict.pth`
  - Segmentation: `quant_seg_model.ts`, `quant_seg_model.onnx`, `quant_seg_report.json`, `quant_seg_state_dict.pth`
- **Outputs**:
  - Classification: Comparison metrics, Top-5 bar chart, download list
  - Segmentation: Comparison metrics, class distribution table, overlay/mask sliders, download list

### What gets exported
- Artifacts are written to `exports/`. JSON reports include the chosen options, metrics, and Top-5 results for both the baseline and optimized variants.
- TorchScript/ONNX exports run best on CPU inputs; failures are logged to the console and skipped.
- State dicts are always saved for reproducibility; disable or prune them manually if you are embedding this module elsewhere.

### Output Interpreting Tips
- **Top-1 Prediction (Classification)**: Labels come from ImageNet synsets, so some entries include multiple comma-separated synonyms (e.g., `chambered nautilus, pearly nautilus`).
- **Mask Agreement (Segmentation)**: Percentage of pixels where original and optimized models predict the same class. 100% means identical masks; lower values indicate divergence.
- **Class Distribution (Segmentation)**: Shows the top 25 most prevalent classes by pixel coverage, with percentages and counts for both models.
- **Image Sliders (Segmentation)**: Drag the slider to compare original vs. optimized overlays or raw masks side-by-side.
- **Latency (ms)**: Includes the reported inference latency for each pass. Large numbers for quantized runs may indicate preprocessing overhead rather than faster model execution—see [Performance Notes](#performance-notes).
- **Model Size (MB)**: Serialized state dictionary size after saving to disk.

## Performance Notes
- The current quantization pipeline rebuilds the optimized model on each request, so the reported latency includes setup time. Reusing pre-quantized instances will yield more realistic numbers.
- Dynamic and weight-only quantization only affect linear layers; ResNet-50 is dominated by convolution blocks that remain FP32, so speedups are modest on CPU. Unsupported static INT8 kernels automatically fall back to dynamic quantization.
- PyTorch default quantization backend may fall back to `qnnpack` on CPU. For x86 systems, set `torch.backends.quantized.engine = "fbgemm"` before quantization for best results.
- FP16 inference is beneficial on GPUs. On CPU, PyTorch often casts half tensors back to float32, introducing overhead.

## Extending the Lab
- **Classification**: Swap in different architectures by changing the `timm.create_model` call in `app.py`.
- **Segmentation**: Add new models from the [smp-hub](https://huggingface.co/smp-hub) collection by adding entries to `SEGMENTATION_MODEL_CONFIGS`.
- Add calibration data and static INT8 quantization to include convolution layers.
- Cache optimized models to avoid recomputation across requests.
- Integrate evaluation datasets to quantify accuracy drop (classification: top-1/top-5, segmentation: mIoU, pixel accuracy).

## CLI Mode
- Run without the UI: `python app.py --cli --image path/to/img.jpg --mode prune --model resnet50 --device auto`
- Modes: `--mode prune` (structured pruning @ 0.4 sparsity) or `--mode quant` (dynamic quantization). Both emit the metrics table and export artifacts list.
- Devices: `auto` chooses CUDA → MPS → CPU based on availability; `cpu`/`cuda`/`mps` force a device. Dynamic/weight-only quantization forces CPU for kernel support even if GPU is requested.
- Models: any entry from `MODEL_OPTIONS` in `app.py`.

## Troubleshooting
- **Slow downloads**: The first run downloads pretrained weights (~100 MB). Subsequent runs use cached files.
- **CUDA errors**: Ensure the correct CUDA-enabled PyTorch build is installed if you intend to run on GPU.
- **Quantized model larger than expected**: The state dictionary includes dequantized tensors for some paths (e.g., dynamic quantization). Consider TorchScript or ONNX export for compact deployment artifacts.

## License
This project inherits the default license of the repository. Replace or update this section if you add a specific license.
