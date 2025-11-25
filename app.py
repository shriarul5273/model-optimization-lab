import argparse
import io
import json
import os
import time
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from PIL import Image
from torchvision import transforms


# ---------------------------------------------
# Base Model Registry / Defaults
# ---------------------------------------------
MODEL_OPTIONS = [
    "resnet50",
    "mobilenetv3_large_100",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_base_patch16_224",
    "regnety_016",
    "efficientnet_lite0",
]
PRETRAINED_DEFAULT = os.getenv("MODEL_OPT_PRETRAINED", "1") == "1"
_PRETRAINED_DISABLED = False
_PRETRAINED_WARNED = False

PRESETS = {
    "Edge CPU": {
        "device": "cpu",
        "channels_last": False,
        "compile": False,
        "quant": "dynamic",
        "prune_amount": 0.3,
    },
    "Datacenter GPU": {
        "device": "cuda",
        "channels_last": True,
        "compile": True,
        "quant": "fp16",
        "prune_amount": 0.2,
    },
    "Apple MPS": {
        "device": "mps",
        "channels_last": False,
        "compile": False,
        "quant": "fp16",
        "prune_amount": 0.2,
    },
}

_MODEL_CACHE: dict[str, torch.nn.Module] = {}
_TRANSFORM_CACHE: dict[str, transforms.Compose] = {}


def select_device(device_str: str) -> torch.device:
    """Return a valid torch.device based on user selection."""
    device_str = (device_str or "auto").lower()
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_str == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform(model_name: str):
    if model_name in _TRANSFORM_CACHE:
        return _TRANSFORM_CACHE[model_name]

    model = get_fp32_model(model_name)

    if hasattr(timm.data, "resolve_model_data_config"):
        data_cfg = timm.data.resolve_model_data_config(model)
    else:
        # Fallback for older timm versions
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)

    _TRANSFORM_CACHE[model_name] = timm.data.create_transform(**data_cfg)
    return _TRANSFORM_CACHE[model_name]


def get_fp32_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        global _PRETRAINED_DISABLED, _PRETRAINED_WARNED
        use_pretrained = PRETRAINED_DEFAULT and not _PRETRAINED_DISABLED
        if not use_pretrained and not _PRETRAINED_WARNED:
            print("INFO: MODEL_OPT_PRETRAINED disabled; using randomly initialized weights.")
            _PRETRAINED_WARNED = True
        try:
            loaded = timm.create_model(model_name, pretrained=use_pretrained)
        except Exception as exc:
            print(f"Warning: pretrained weights unavailable ({exc}); using random init for {model_name}")
            _PRETRAINED_DISABLED = True
            loaded = timm.create_model(model_name, pretrained=False)
        loaded.eval()
        _MODEL_CACHE[model_name] = loaded
    return _MODEL_CACHE[model_name]


def clone_model(model_name: str):
    """Create a fresh model loaded from the cached FP32 weights to avoid re-downloads."""
    base = get_fp32_model(model_name)
    fresh = timm.create_model(model_name, pretrained=False)
    fresh.load_state_dict(base.state_dict())
    fresh.eval()
    return fresh


# ---------------------------------------------
# Image Preprocess
# ---------------------------------------------
imagenet_info = timm.data.ImageNetInfo()
labels = [imagenet_info.index_to_description(i) for i in range(1000)]


# ---------------------------------------------
# PRUNING FUNCTION (dynamic)
# ---------------------------------------------
def apply_pruning(model, amount=0.5, method="unstructured"):
    model = model.eval()

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            if method == "unstructured":
                prune.l1_unstructured(module, name="weight", amount=amount)
            elif method == "structured":
                prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)

    # Make pruning permanent to reflect true sparsity/size
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and hasattr(module, "weight_orig"):
            prune.remove(module, "weight")
    return model


# ---------------------------------------------
# QUANTIZATION FUNCTION (dynamic)
# ---------------------------------------------
def apply_quantization(model, q_type="dynamic"):
    q_type = q_type or "dynamic"
    if q_type in {"dynamic", "weight_only"}:  # dynamic quantization is weight-only by default
        return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    if q_type == "fp16":
        return model.half().eval()
    return model


def compute_sparsity(model: nn.Module) -> pd.DataFrame:
    rows = []
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            weight = module.weight.detach()
            total = weight.numel()
            zeros = (weight == 0).sum().item()
            sparsity = 100.0 * zeros / max(total, 1)
            rows.append({"Layer": name, "Params": total, "Sparsity %": round(sparsity, 2)})
    return pd.DataFrame(rows)


def _macs_conv2d(inp, module: nn.Conv2d, out):
    # inp: (N, C_in, H, W), out: (N, C_out, H_out, W_out)
    batch, c_in, h, w = inp.shape
    _, c_out, h_out, w_out = out.shape
    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (c_in / module.groups)
    return batch * h_out * w_out * c_out * kernel_ops


def _macs_linear(inp, module: nn.Linear, out):
    batch = inp.shape[0]
    return batch * module.in_features * module.out_features


def layer_profile(model: nn.Module, sample_input: torch.Tensor) -> pd.DataFrame:
    """Collect per-layer params, MACs, and forward time (single run)."""
    rows = []
    handles = []
    start_times = {}

    def pre_hook(name):
        def _pre(mod, inp):
            start_times[name] = time.time()
        return _pre

    def fwd_hook(name):
        def _fwd(mod, inp, out):
            end = time.time()
            duration_ms = (end - start_times.get(name, end)) * 1000
            inp0 = inp[0] if isinstance(inp, (tuple, list)) else inp
            macs = None
            if isinstance(mod, nn.Conv2d):
                macs = _macs_conv2d(inp0, mod, out)
            elif isinstance(mod, nn.Linear):
                macs = _macs_linear(inp0, mod, out)
            params = sum(p.numel() for p in mod.parameters())
            rows.append({
                "Layer": name,
                "Type": mod.__class__.__name__,
                "Params": params,
                "MACs": macs if macs is None else float(macs),
                "Latency (ms)": round(duration_ms, 3),
            })
        return _fwd

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf
            handles.append(module.register_forward_pre_hook(pre_hook(name)))
            handles.append(module.register_forward_hook(fwd_hook(name)))

    with torch.no_grad():
        model(sample_input)

    for h in handles:
        h.remove()

    return pd.DataFrame(rows)


def maybe_compile(model, use_compile: bool):
    if not use_compile:
        return model
    if not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model)
    except Exception:
        return model


def get_state_dict_size_mb(model: nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getbuffer()) / 1e6


def prepare_image(image, transform_fn):
    if image is None:
        raise ValueError("No image provided")

    if not isinstance(image, Image.Image):
        if isinstance(image, np.ndarray) and image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        image = Image.fromarray(image.astype("uint8"))

    image = image.convert("RGB")
    tensor = transform_fn(image).unsqueeze(0)
    return tensor


def grad_cam(image, model, device, transform_fn):
    model.eval()
    target_layer = None
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Conv2d):
            target_layer = m
            break
    if target_layer is None:
        raise ValueError("No Conv2d layer found for Grad-CAM")

    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["value"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    handle_fwd = target_layer.register_forward_hook(fwd_hook)
    handle_bwd = target_layer.register_full_backward_hook(bwd_hook)

    img_t = prepare_image(image, transform_fn).to(device)
    img_t.requires_grad_(True)

    with torch.no_grad():
        pass

    out = model(img_t)
    top1 = out.argmax(dim=1)
    score = out[0, top1]
    model.zero_grad()
    score.backward()

    act = activations["value"]
    grad = gradients["value"]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze().cpu().numpy()
    cam -= cam.min()
    cam /= cam.max() + 1e-8

    # Resize CAM to image size
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(image.size, resample=Image.BILINEAR)
    heatmap = np.array(cam_img)
    heatmap_rgb = np.stack([heatmap, np.zeros_like(heatmap), np.zeros_like(heatmap)], axis=-1)
    overlay = np.array(image.convert("RGB"), dtype=np.float32)
    alpha = 0.35
    blended = (overlay * (1 - alpha) + heatmap_rgb * alpha).clip(0, 255).astype("uint8")
    blended_img = Image.fromarray(blended)

    handle_fwd.remove()
    handle_bwd.remove()
    return blended_img


# ---------------------------------------------
# Inference Function (shared)
# ---------------------------------------------
def run_inference(model, image, device, transform_fn, channels_last=False, warmup=False, use_amp=False):
    print(f"  run_inference called, image type: {type(image)}")
    img = prepare_image(image, transform_fn)

    model = model.to(device)
    img = img.to(device)

    if channels_last and device.type == "cuda":
        img = img.to(memory_format=torch.channels_last)

    if next(model.parameters()).dtype == torch.float16:
        img = img.half()

    if warmup:
        with torch.no_grad():
            model(img)

    print("  Running model inference...")
    start = time.time()
    amp_ctx = torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda")
    with torch.no_grad(), amp_ctx:
        out = model(img)
    latency = (time.time() - start) * 1000

    out_cpu = out.detach().cpu()
    prob = torch.softmax(out_cpu, dim=1)[0]
    top5_prob, top5_idx = torch.topk(prob, 5)

    results = [(labels[i], float(top5_prob[j])) for j, i in enumerate(top5_idx)]
    print(f"  Inference complete. Top prediction: {results[0][0]}")
    return results, latency


def build_top5_plot(results_orig, results_other, other_label: str):
    classes = []
    for r in results_orig + results_other:
        if r[0] not in classes:
            classes.append(r[0])
    orig_map = {r[0]: r[1] for r in results_orig}
    other_map = {r[0]: r[1] for r in results_other}
    orig_vals = [orig_map.get(c, 0.0) for c in classes]
    other_vals = [other_map.get(c, 0.0) for c in classes]
    x = np.arange(len(classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, orig_vals, width, label="Original")
    ax.bar(x + width / 2, other_vals, width, label=other_label)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylabel("Confidence")
    ax.set_xlabel("Class")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------
# Gradio Functions With Options
# ---------------------------------------------
def run_pruned(
    img,
    model_name,
    method,
    amount,
    device_choice="auto",
    channels_last=False,
    use_compile=False,
    use_amp=False,
    export_ts=False,
    export_onnx=False,
    export_report=False,
    export_state=True,
    preset=None,
):
    print("\n=== RUN PRUNED CALLED ===")
    print(f"Image type: {type(img)}, Model: {model_name}, Method: {method}, Amount: {amount}")

    if img is None:
        print("ERROR: Image is None")
        return (
            {"Metric": ["Error"], "Original Model": ["No image uploaded"], "Pruned Model": [""]},
            {"Class": [], "Original": [], "Pruned": []},
            pd.DataFrame(),
            []
        )

    if preset in PRESETS:
        preset_cfg = PRESETS[preset]
        device_choice = preset_cfg["device"]
        channels_last = preset_cfg["channels_last"]
        use_compile = preset_cfg["compile"]
        use_amp = preset_cfg.get("amp", use_amp)
        amount = preset_cfg.get("prune_amount", amount)

    device = select_device(device_choice)
    transform_fn = get_transform(model_name)

    # Run original model
    print("Running original model...")
    fp32_model = get_fp32_model(model_name)
    results_orig, latency_orig = run_inference(fp32_model, img, device, transform_fn, channels_last, warmup=True, use_amp=use_amp)
    print(f"Original model done. Latency: {latency_orig:.2f}ms")

    # Run pruned model
    print("Creating fresh model...")
    fresh_model = clone_model(model_name)
    print("Applying pruning...")
    pruned_model = apply_pruning(fresh_model, amount=float(amount), method=method)
    pruned_model = maybe_compile(pruned_model, use_compile)
    print("Running pruned model...")
    results_pruned, latency_pruned = run_inference(pruned_model, img, device, transform_fn, channels_last, warmup=True, use_amp=use_amp)
    print(f"Pruned model done. Latency: {latency_pruned:.2f}ms")

    # Model sizes (in-memory)
    size_orig = get_state_dict_size_mb(fp32_model)
    size_pruned = get_state_dict_size_mb(pruned_model)
    print(f"Model sizes - Original: {size_orig:.2f}MB, Pruned: {size_pruned:.2f}MB")

    # Comparison metrics - as DataFrame for Gradio
    metrics_df = pd.DataFrame({
        "Metric": ["Top-1 Prediction", "Confidence", "Latency (ms)", "Model Size (MB)"],
        "Original Model": [
            results_orig[0][0],
            f"{results_orig[0][1]:.4f}",
            f"{latency_orig:.2f}",
            f"{size_orig:.2f}"
        ],
        "Pruned Model": [
            results_pruned[0][0],
            f"{results_pruned[0][1]:.4f}",
            f"{latency_pruned:.2f}",
            f"{size_pruned:.2f}"
        ]
    })

    chart_fig = build_top5_plot(results_orig, results_pruned, "Pruned")
    sparsity_df = compute_sparsity(pruned_model.cpu())

    downloads = []
    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)
    sample_cpu = prepare_image(img, transform_fn)

    if export_report:
        report_path = export_dir / "pruned_report.json"
        report = {
            "model": model_name,
            "pruning": {"method": method, "amount": float(amount)},
            "metrics": metrics_df.to_dict(),
            "top5_pruned": results_pruned,
            "top5_original": results_orig,
        }
        report_path.write_text(json.dumps(report, indent=2))
        downloads.append(str(report_path))

    # Always allow state_dict download for reproducibility
    if export_state:
        state_path = export_dir / "pruned_state_dict.pth"
        torch.save(pruned_model.state_dict(), state_path)
        downloads.append(str(state_path))

    if export_ts:
        ts_path = export_dir / "pruned_model.ts"
        try:
            scripted = torch.jit.trace(pruned_model.cpu(), sample_cpu)
            scripted.save(ts_path)
            downloads.append(str(ts_path))
        except Exception as exc:
            print(f"TorchScript export failed: {exc}")

    if export_onnx:
        onnx_path = export_dir / "pruned_model.onnx"
        try:
            torch.onnx.export(
                pruned_model.cpu(),
                sample_cpu,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=13,
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
            downloads.append(str(onnx_path))
        except Exception as exc:
            print(f"ONNX export failed: {exc}")

    print("=== RUN PRUNED COMPLETE ===")
    return metrics_df, chart_fig, sparsity_df, downloads


def run_quantized(
    img,
    model_name,
    q_type,
    device_choice="auto",
    channels_last=False,
    use_compile=False,
    use_amp=False,
    export_ts=False,
    export_onnx=False,
    export_report=False,
    export_state=True,
    preset=None,
):
    print("\n=== RUN QUANTIZED CALLED ===")
    print(f"Image type: {type(img)}, Model: {model_name}, Q-type: {q_type}")

    if img is None:
        print("ERROR: Image is None")
        return (
            {"Metric": ["Error"], "Original Model": ["No image uploaded"], "Quantized Model": [""]},
            {"Class": [], "Original": [], "Quantized": []},
            []
        )

    if preset in PRESETS:
        preset_cfg = PRESETS[preset]
        device_choice = preset_cfg["device"]
        channels_last = preset_cfg["channels_last"]
        use_compile = preset_cfg["compile"]
        use_amp = preset_cfg.get("amp", use_amp)
        q_type = preset_cfg.get("quant", q_type)

    device = select_device(device_choice)
    if q_type in {"dynamic", "weight_only"} and device.type != "cpu":
        print("Dynamic/weight-only quantization uses CPU kernels; switching device to CPU.")
        device = torch.device("cpu")
        channels_last = False
        use_amp = False
    transform_fn = get_transform(model_name)

    # Run original model
    print("Running original model...")
    fp32_model = get_fp32_model(model_name)
    results_orig, latency_orig = run_inference(fp32_model, img, device, transform_fn, channels_last, warmup=True, use_amp=use_amp)
    print(f"Original model done. Latency: {latency_orig:.2f}ms")

    # Run quantized model
    fresh_model = clone_model(model_name)
    quant_model = apply_quantization(fresh_model, q_type)
    quant_model = maybe_compile(quant_model, use_compile)
    results_quant, latency_quant = run_inference(quant_model, img, device, transform_fn, channels_last, warmup=True, use_amp=use_amp)

    size_orig = get_state_dict_size_mb(fp32_model)
    size_quant = get_state_dict_size_mb(quant_model)

    metrics_df = pd.DataFrame({
        "Metric": ["Top-1 Prediction", "Confidence", "Latency (ms)", "Model Size (MB)"],
        "Original Model": [
            results_orig[0][0],
            f"{results_orig[0][1]:.4f}",
            f"{latency_orig:.2f}",
            f"{size_orig:.2f}"
        ],
        "Quantized Model": [
            results_quant[0][0],
            f"{results_quant[0][1]:.4f}",
            f"{latency_quant:.2f}",
            f"{size_quant:.2f}"
        ]
    })

    chart_fig = build_top5_plot(results_orig, results_quant, "Quantized")

    downloads = []
    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)
    sample_cpu = prepare_image(img, transform_fn)

    if export_report:
        report_path = export_dir / "quant_report.json"
        report = {
            "model": model_name,
            "quantization": q_type,
            "metrics": metrics_df.to_dict(),
            "top5_quantized": results_quant,
            "top5_original": results_orig,
        }
        report_path.write_text(json.dumps(report, indent=2))
        downloads.append(str(report_path))

    if export_state:
        state_path = export_dir / "quantized_state_dict.pth"
        torch.save(quant_model.state_dict(), state_path)
        downloads.append(str(state_path))

    if export_ts:
        ts_path = export_dir / "quantized_model.ts"
        try:
            scripted = torch.jit.trace(quant_model.cpu(), sample_cpu)
            scripted.save(ts_path)
            downloads.append(str(ts_path))
        except Exception as exc:
            print(f"TorchScript export failed: {exc}")

    if export_onnx:
        onnx_path = export_dir / "quantized_model.onnx"
        try:
            torch.onnx.export(
                quant_model.cpu(),
                sample_cpu,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=13,
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
            downloads.append(str(onnx_path))
        except Exception as exc:
            print(f"ONNX export failed: {exc}")

    print("=== RUN QUANTIZED COMPLETE ===")
    return metrics_df, chart_fig, downloads


def run_profile(image, model_name, variant, device_choice="auto"):
    if image is None:
        return pd.DataFrame()
    device = select_device(device_choice)
    if variant == "quant" and device.type != "cpu":
        device = torch.device("cpu")
    transform_fn = get_transform(model_name)
    sample = prepare_image(image, transform_fn).to(device)

    if variant == "fp32":
        model = get_fp32_model(model_name).to(device)
    elif variant == "pruned":
        model = apply_pruning(clone_model(model_name), amount=0.4)
    else:
        model = apply_quantization(clone_model(model_name), "dynamic")

    profile_df = layer_profile(model, sample)
    return profile_df


def run_batch(images, model_name, mode, device_choice="auto"):
    """Batch runner: returns per-image metrics and aggregate stats."""
    if not images:
        return pd.DataFrame(), pd.DataFrame()

    device = select_device(device_choice)
    transform_fn = get_transform(model_name)

    per_image = []
    latencies = []
    labels_map = {}
    expanded_files = []
    temp_dirs = []

    for path in images:
        if isinstance(path, str) and path.endswith(".zip"):
            td = TemporaryDirectory()
            temp_dirs.append(td)  # keep alive until function ends
            with zipfile.ZipFile(path) as zf:
                zf.extractall(td.name)
            for root, _, files in os.walk(td.name):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        expanded_files.append(os.path.join(root, f))
                    if f.lower() in {"labels.txt", "labels.csv"}:
                        with open(os.path.join(root, f)) as lf:
                            for line in lf:
                                parts = line.strip().split(",")
                                if len(parts) >= 2:
                                    labels_map[parts[0]] = parts[1]
        else:
            expanded_files.append(path)

    for path in expanded_files:
        img = Image.open(path) if isinstance(path, str) else path
        if mode == "prune":
            metrics, _, _, _ = run_pruned(
                img,
                model_name,
                "structured",
                0.4,
                device_choice=device_choice,
                export_state=False,
            )
            latency = float(metrics.loc[metrics["Metric"] == "Latency (ms)", "Pruned Model"].values[0])
            top1 = metrics.loc[metrics["Metric"] == "Top-1 Prediction", "Pruned Model"].values[0]
        else:
            metrics, _, _ = run_quantized(
                img,
                model_name,
                "dynamic",
                device_choice=device_choice,
                export_state=False,
            )
            latency = float(metrics.loc[metrics["Metric"] == "Latency (ms)", "Quantized Model"].values[0])
            top1 = metrics.loc[metrics["Metric"] == "Top-1 Prediction", "Quantized Model"].values[0]

        fname = os.path.basename(getattr(path, "name", path))
        record = {"Image": fname, "Top-1": top1, "Latency (ms)": latency}
        if fname in labels_map:
            record["Label"] = labels_map[fname]
            record["Correct"] = labels_map[fname] == top1
        per_image.append(record)
        latencies.append(latency)

    per_image_df = pd.DataFrame(per_image)
    summary = {
        "count": len(latencies),
        "mean_latency": float(np.mean(latencies)),
        "median_latency": float(np.median(latencies)),
        "max_latency": float(np.max(latencies)),
    }
    if "Correct" in per_image_df.columns:
        summary["accuracy"] = float(per_image_df["Correct"].mean())

    summary_df = pd.DataFrame({"Metric": list(summary.keys()), "Value": list(summary.values())})
    return per_image_df, summary_df


def run_sweep(img, model_name, device_choice, experiments_json=None):
    if img is None:
        return pd.DataFrame(), pd.DataFrame()
    default_experiments = [
        {"mode": "prune", "amount": 0.2, "method": "structured"},
        {"mode": "prune", "amount": 0.5, "method": "structured"},
        {"mode": "quant", "q_type": "dynamic"},
        {"mode": "quant", "q_type": "fp16"},
    ]
    try:
        experiments = json.loads(experiments_json) if experiments_json else default_experiments
    except Exception:
        experiments = default_experiments

    rows = []
    for exp in experiments:
        if exp.get("mode") == "prune":
            metrics, _, _, _ = run_pruned(
                img,
                model_name,
                exp.get("method", "structured"),
                exp.get("amount", 0.4),
                device_choice=device_choice,
                export_state=False,
            )
            latency = float(metrics.loc[metrics["Metric"] == "Latency (ms)", "Pruned Model"].values[0])
            size = float(metrics.loc[metrics["Metric"] == "Model Size (MB)", "Pruned Model"].values[0])
            top1 = metrics.loc[metrics["Metric"] == "Top-1 Prediction", "Pruned Model"].values[0]
            rows.append({"mode": "prune", "amount": exp.get("amount"), "latency": latency, "size": size, "top1": top1})
        else:
            metrics, _, _ = run_quantized(
                img,
                model_name,
                exp.get("q_type", "dynamic"),
                device_choice=device_choice,
                export_state=False,
            )
            latency = float(metrics.loc[metrics["Metric"] == "Latency (ms)", "Quantized Model"].values[0])
            size = float(metrics.loc[metrics["Metric"] == "Model Size (MB)", "Quantized Model"].values[0])
            top1 = metrics.loc[metrics["Metric"] == "Top-1 Prediction", "Quantized Model"].values[0]
            rows.append({"mode": "quant", "q_type": exp.get("q_type"), "latency": latency, "size": size, "top1": top1})

    df = pd.DataFrame(rows)
    pareto_df = df.rename(columns={"latency": "Latency (ms)", "size": "Model Size (MB)", "top1": "Top-1"})
    return df, pareto_df


def fastapi_snippet():
    return """
from fastapi import FastAPI, UploadFile
from PIL import Image
import io, torch, timm
from app import get_transform, get_fp32_model, run_inference, select_device

app = FastAPI()
MODEL = "resnet50"
DEVICE = select_device("auto")
MODEL_OBJ = get_fp32_model(MODEL).to(DEVICE)
TRANSFORM = get_transform(MODEL)


@app.post('/predict')
async def predict(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    results, latency = run_inference(MODEL_OBJ, img, DEVICE, TRANSFORM)
    return {"top1": results[0][0], "confidence": results[0][1], "latency_ms": latency}
"""


def dockerfile_snippet():
    return """
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py", "--cli", "--image", "examples/cat.jpg"]
"""


# ---------------------------------------------
# GRADIO UI
# ---------------------------------------------
examples = [["examples/cat.jpg"], ["examples/dog.jpg"], ["examples/bird.jpg"], ["examples/car.jpg"], ["examples/elephant.jpg"]]


def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# 🧠 Model Optimization Lab — Compare, Export, Benchmark")

        device_opts = ["auto", "cpu"]
        if torch.cuda.is_available():
            device_opts.append("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device_opts.append("mps")
        preset_opts = list(PRESETS.keys()) + ["custom"]

        with gr.Tabs():
            # ---- PRUNING TAB ----
            with gr.Tab("Pruning"):
                with gr.Row():
                    with gr.Column():
                        img_p = gr.Image(label="Upload Image")
                        model_p = gr.Dropdown(MODEL_OPTIONS, value=MODEL_OPTIONS[0], label="Base Model")
                        preset_p = gr.Dropdown(preset_opts, value="custom", label="Hardware Preset")
                        method_p = gr.Dropdown(["unstructured", "structured"], value="structured", label="Pruning Method")
                        amount_p = gr.Slider(minimum=0.1, maximum=0.9, step=0.1, value=0.4, label="Pruning Amount")
                        device_p = gr.Dropdown(device_opts, value=device_opts[0], label="Device")
                        channels_last_p = gr.Checkbox(label="Channels-last input (CUDA)", value=True)
                        amp_p = gr.Checkbox(label="Mixed precision (AMP)", value=True)
                        compile_p = gr.Checkbox(label="Torch compile (PyTorch 2)")
                        export_ts_p = gr.Checkbox(label="Export TorchScript")
                        export_onnx_p = gr.Checkbox(label="Export ONNX")
                        export_report_p = gr.Checkbox(label="Export JSON report", value=True)
                        btn_p = gr.Button("Run Pruned Model")
                        gr.Examples(examples=examples, inputs=img_p)

                    with gr.Column():
                        metrics_p = gr.Dataframe(label="📊 Comparison Metrics", headers=["Metric", "Original Model", "Pruned Model"])
                        chart_p = gr.Plot(label="🎯 Top-5 Predictions Comparison")
                        sparsity_p = gr.Dataframe(label="Layer sparsity (%)")
                        downloads_p = gr.Files(label="Exports (state_dict / TorchScript / ONNX / report)")

                btn_p.click(
                    fn=run_pruned,
                    inputs=[
                        img_p,
                        model_p,
                        method_p,
                        amount_p,
                        device_p,
                        channels_last_p,
                        compile_p,
                        amp_p,
                        export_ts_p,
                        export_onnx_p,
                        export_report_p,
                        gr.State(True),
                        preset_p,
                    ],
                    outputs=[metrics_p, chart_p, sparsity_p, downloads_p],
                )

            # ---- QUANTIZATION TAB ----
            with gr.Tab("Quantization"):
                with gr.Row():
                    with gr.Column():
                        img_q = gr.Image(label="Upload Image")
                        model_q = gr.Dropdown(MODEL_OPTIONS, value=MODEL_OPTIONS[0], label="Base Model")
                        preset_q = gr.Dropdown(preset_opts, value="custom", label="Hardware Preset")
                        q_type = gr.Dropdown(["dynamic", "weight_only", "fp16"], value="dynamic", label="Quantization Type")
                        device_q = gr.Dropdown(device_opts, value=device_opts[0], label="Device")
                        channels_last_q = gr.Checkbox(label="Channels-last input (CUDA)", value=True)
                        amp_q = gr.Checkbox(label="Mixed precision (AMP)", value=True)
                        compile_q = gr.Checkbox(label="Torch compile (PyTorch 2)")
                        export_ts_q = gr.Checkbox(label="Export TorchScript")
                        export_onnx_q = gr.Checkbox(label="Export ONNX")
                        export_report_q = gr.Checkbox(label="Export JSON report", value=True)
                        btn_q = gr.Button("Run Quantized Model")
                        gr.Examples(examples=examples, inputs=img_q)

                    with gr.Column():
                        metrics_q = gr.Dataframe(label="📊 Comparison Metrics", headers=["Metric", "Original Model", "Quantized Model"])
                        chart_q = gr.Plot(label="🎯 Top-5 Predictions Comparison")
                        downloads_q = gr.Files(label="Exports (state_dict / TorchScript / ONNX / report)")

                btn_q.click(
                    fn=run_quantized,
                    inputs=[
                        img_q,
                        model_q,
                        q_type,
                        device_q,
                        channels_last_q,
                        compile_q,
                        amp_q,
                        export_ts_q,
                        export_onnx_q,
                        export_report_q,
                        gr.State(True),
                        preset_q,
                    ],
                    outputs=[metrics_q, chart_q, downloads_q],
                )

        return demo


def main():
    parser = argparse.ArgumentParser(description="Model optimization lab")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode (no UI)")
    parser.add_argument("--mode", choices=["prune", "quant"], default="prune", help="Optimization mode in CLI")
    parser.add_argument("--image", type=str, help="Path to an image for CLI mode")
    parser.add_argument("--model", type=str, default=MODEL_OPTIONS[0], choices=MODEL_OPTIONS)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.cli:
        if not args.image:
            raise SystemExit("--image is required in CLI mode")
        img = Image.open(args.image)
        if args.mode == "prune":
            metrics, _, downloads = run_pruned(img, args.model, "structured", 0.4, device_choice=args.device)
        else:
            metrics, _, downloads = run_quantized(img, args.model, "dynamic", device_choice=args.device)
        print(metrics)
        print("Exports:", downloads)
        return

    demo = create_demo()
    demo.launch()


if __name__ == "__main__":
    main()
