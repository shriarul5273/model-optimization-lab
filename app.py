import argparse
import copy
import io
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import segmentation_models_pytorch as smp
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_resnet50_fpn,
    ssdlite320_mobilenet_v3_large,
)
try:
    from ultralytics import YOLO as UltralyticsYOLO
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    UltralyticsYOLO = None
try:
    import albumentations as A
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    A = None


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

@dataclass(frozen=True)
class SegmentationModelConfig:
    name: str
    checkpoint: str
    classes: int = 150
    dataset: str = "ADE20K"


SEGMENTATION_MODEL_CONFIGS: tuple[SegmentationModelConfig, ...] = (
    SegmentationModelConfig("SegFormer B0 (ADE20K 512x512)", "smp-hub/segformer-b0-512x512-ade-160k"),
    SegmentationModelConfig("SegFormer B4 (ADE20K 512x512)", "smp-hub/segformer-b4-512x512-ade-160k"),
    SegmentationModelConfig("DPT Large (ADE20K)", "smp-hub/dpt-large-ade20k"),
    SegmentationModelConfig("UPerNet ConvNeXt-Tiny (ADE20K)", "smp-hub/upernet-convnext-tiny"),
)
SEGMENTATION_MODEL_MAP = {cfg.name: cfg for cfg in SEGMENTATION_MODEL_CONFIGS}

_SEG_BASE_PALETTE = np.array(
    [
        [0, 0, 0],
        [0, 114, 189],
        [217, 83, 25],
        [237, 177, 32],
        [126, 47, 142],
        [119, 172, 48],
        [77, 190, 238],
        [162, 20, 47],
        [163, 200, 236],
        [255, 127, 14],
        [255, 188, 121],
        [111, 118, 207],
        [204, 121, 167],
        [148, 103, 189],
        [44, 160, 44],
        [23, 190, 207],
        [31, 119, 180],
        [255, 152, 150],
        [214, 39, 40],
        [188, 189, 34],
    ],
    dtype=np.uint8,
)

_SEG_MODEL_CACHE: dict[str, torch.nn.Module] = {}
_SEG_TRANSFORM_CACHE: dict[str, object] = {}
_SEG_PALETTE_CACHE: dict[int, np.ndarray] = {}

ADE20K_CLASS_NAMES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway",
    "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench",
    "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel",
    "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television receiver",
    "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet",
    "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
    "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball",
    "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan",
    "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag"
]


# ---------------------------------------------
# Object Detection Registry / Defaults
# ---------------------------------------------
DETECTION_MODEL_CONFIGS = {
    "Faster R-CNN ResNet50 FPN (COCO)": {
        "builder": fasterrcnn_resnet50_fpn,
        "weights": FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        "backend": "torchvision",
    },
    "SSDlite320 MobileNetV3 (COCO)": {
        "builder": ssdlite320_mobilenet_v3_large,
        "weights": SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
        "backend": "torchvision",
    },
}
COCO_CATEGORIES = list(FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta.get("categories", []))

# Optional YOLOv12 variants (Ultralytics) with size options: n/s/m/l/x
for _size in ("n", "s", "m", "l", "x"):
    DETECTION_MODEL_CONFIGS[f"YOLO12-{_size} (COCO)"] = {
        "backend": "ultralytics",
        "weights": f"yolo12{_size}.pt",
        "imgsz": 640,
        "categories": COCO_CATEGORIES,
    }
DETECTION_MODEL_OPTIONS = list(DETECTION_MODEL_CONFIGS.keys())

_DET_MODEL_CACHE: dict[str, nn.Module] = {}
_DET_TRANSFORM_CACHE: dict[str, object] = {}
_DET_LABELS_CACHE: dict[str, list[str]] = {}


def _require_ultralytics():
    if UltralyticsYOLO is None:
        raise RuntimeError(
            "The 'ultralytics' package is required for YOLO12 models. "
            "Install it with `pip install ultralytics` to enable these options."
        )


def add_image_label(img: Image.Image, label: str) -> Image.Image:
    """Add a text label at the top of an image."""
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    
    # Create canvas with extra space at top for label
    canvas = np.ones((h + 40, w, 3), dtype=np.uint8) * 255
    canvas[40:, :] = img_array
    
    # Convert back to PIL for text drawing
    canvas_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas_img)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()
    
    # Get text size and center it
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (w - text_width) // 2
    
    # Draw text
    draw.text((text_x, 10), label, fill=(0, 0, 0), font=font)
    
    return canvas_img


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
# Segmentation Utilities
# ---------------------------------------------
def _require_albumentations():
    if A is None:
        raise RuntimeError(
            "Albumentations is required for pretrained segmentation models. "
            "Install it with `pip install albumentations` or add it to your environment."
        )


def get_segmentation_model(config: SegmentationModelConfig) -> nn.Module:
    key = config.checkpoint
    if key not in _SEG_MODEL_CACHE:
        model = smp.from_pretrained(config.checkpoint).eval()
        _SEG_MODEL_CACHE[key] = model
    return _SEG_MODEL_CACHE[key]


def clone_segmentation_model(config: SegmentationModelConfig) -> nn.Module:
    base = get_segmentation_model(config)
    fresh = smp.from_pretrained(config.checkpoint).eval()
    fresh.load_state_dict(base.state_dict())
    return fresh


def get_segmentation_transform(config: SegmentationModelConfig):
    key = config.checkpoint
    if key in _SEG_TRANSFORM_CACHE:
        return _SEG_TRANSFORM_CACHE[key]

    _require_albumentations()
    try:
        preprocessing = A.Compose.from_pretrained(config.checkpoint)
    except Exception as exc:  # pragma: no cover - depends on network availability
        raise RuntimeError(f"Failed to load preprocessing pipeline for {config.checkpoint}: {exc}") from exc

    def _transform(image):
        if image is None:
            raise ValueError("No image provided")
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                array = image
                if array.dtype != np.uint8:
                    array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
                image_rgb = Image.fromarray(array)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
        else:
            image_rgb = image

        image_rgb = image_rgb.convert("RGB")
        np_image = np.array(image_rgb)
        processed = preprocessing(image=np_image)["image"]
        if isinstance(processed, torch.Tensor):
            processed_np = processed.detach().cpu().numpy()
        else:
            processed_np = np.asarray(processed, dtype=np.float32)
        tensor = torch.from_numpy(processed_np.transpose(2, 0, 1)).float()
        return tensor, image_rgb

    _SEG_TRANSFORM_CACHE[key] = _transform
    return _transform


def get_segmentation_palette(class_count: int) -> np.ndarray:
    if class_count in _SEG_PALETTE_CACHE:
        return _SEG_PALETTE_CACHE[class_count]

    base_len = len(_SEG_BASE_PALETTE)
    if class_count <= base_len:
        palette = _SEG_BASE_PALETTE[:class_count]
    else:
        palette = np.zeros((class_count, 3), dtype=np.uint8)
        palette[:base_len] = _SEG_BASE_PALETTE
        rng = np.random.default_rng(1337)
        palette[base_len:] = rng.integers(0, 256, size=(class_count - base_len, 3), endpoint=False, dtype=np.uint8)
        palette[:, 0] |= 1  # ensure colors are not pure black except index 0
        palette[0] = np.array([0, 0, 0], dtype=np.uint8)

    _SEG_PALETTE_CACHE[class_count] = palette
    return palette


def colorize_mask(mask: np.ndarray, class_count: int) -> Image.Image:
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D for colorization")
    palette = get_segmentation_palette(class_count)
    indexed = np.mod(mask, class_count)
    colored = palette[indexed]
    return Image.fromarray(colored.astype(np.uint8))


def overlay_mask(image: Image.Image, mask_image: Image.Image, alpha: float = 0.5) -> Image.Image:
    base = np.array(image.convert("RGB"), dtype=np.float32)
    mask_resized = mask_image.resize(image.size, Image.NEAREST)
    mask_arr = np.array(mask_resized, dtype=np.float32)
    blended = (1.0 - alpha) * base + alpha * mask_arr
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def summarize_mask(mask: np.ndarray, class_count: int) -> list[dict[str, float]]:
    flat = mask.reshape(-1)
    counts = np.bincount(flat, minlength=class_count)
    total = float(flat.size)
    summary = []
    for idx in range(class_count):
        count = int(counts[idx])
        percent = (count / total * 100.0) if total else 0.0
        summary.append({"index": idx, "count": count, "percent": percent})
    return summary


def get_class_labels(config: SegmentationModelConfig) -> list[str]:
    # Try to get labels from model metadata first
    model = get_segmentation_model(config)
    meta = getattr(model, "meta", {}) or {}
    dataset_meta = meta.get("dataset", {}) or {}
    labels = dataset_meta.get("class_names") or dataset_meta.get("classes_names")
    
    # If not in metadata, use dataset-specific labels
    if not labels:
        if config.dataset == "ADE20K" and config.classes == 150:
            labels = ADE20K_CLASS_NAMES
        else:
            labels = [f"Class {idx}" for idx in range(config.classes)]
    else:
        labels = list(labels)
    
    # Ensure we have the right number of labels
    if len(labels) < config.classes:
        labels.extend(f"Class {len(labels) + i}" for i in range(config.classes - len(labels)))
    return labels[: config.classes]


# ---------------------------------------------
# Object Detection Utilities
# ---------------------------------------------
def get_detection_config(model_name: str) -> dict:
    if model_name not in DETECTION_MODEL_CONFIGS:
        raise ValueError(f"Unknown detection model: {model_name}")
    return dict(DETECTION_MODEL_CONFIGS[model_name])


def get_detection_labels(model_name: str) -> list[str]:
    if model_name in _DET_LABELS_CACHE:
        return _DET_LABELS_CACHE[model_name]
    cfg = get_detection_config(model_name)
    categories = cfg.get("categories")
    if categories:
        labels = categories
    else:
        weights = cfg.get("weights")
        labels = weights.meta.get("categories", []) if weights else []
    _DET_LABELS_CACHE[model_name] = list(labels)
    return _DET_LABELS_CACHE[model_name]


def get_detection_transform(model_name: str):
    if model_name in _DET_TRANSFORM_CACHE:
        return _DET_TRANSFORM_CACHE[model_name]
    cfg = get_detection_config(model_name)
    backend = cfg.get("backend", "torchvision")
    if backend == "ultralytics":
        transform = lambda img: img  # Ultralytics handles preprocessing internally
    else:
        weights = cfg.get("weights")
        transform = weights.transforms() if weights else transforms.Compose([transforms.ToTensor()])
    _DET_TRANSFORM_CACHE[model_name] = transform
    return transform


def get_detection_model(model_name: str) -> nn.Module:
    if model_name not in _DET_MODEL_CACHE:
        cfg = get_detection_config(model_name)
        backend = cfg.get("backend", "torchvision")
        if backend == "ultralytics":
            _require_ultralytics()
            weights = cfg.get("weights")
            try:
                model = UltralyticsYOLO(weights)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load YOLO12 weights '{weights}'. Download or place the checkpoint locally first."
                ) from exc
            if hasattr(model, "model"):
                model.model.eval()
        else:
            weights = cfg.get("weights")
            try:
                model = cfg["builder"](weights=weights)
            except Exception as exc:
                print(f"Warning: detection weights unavailable ({exc}); using random init for {model_name}")
                model = cfg["builder"](weights=None)
            model.eval()
        _DET_MODEL_CACHE[model_name] = model
    return _DET_MODEL_CACHE[model_name]


def clone_detection_model(model_name: str) -> nn.Module:
    base = get_detection_model(model_name)
    cfg = get_detection_config(model_name)
    backend = cfg.get("backend", "torchvision")
    if backend == "ultralytics":
        _require_ultralytics()
        fresh = copy.deepcopy(base)
        if hasattr(fresh, "model") and isinstance(fresh.model, nn.Module):
            fresh.model.eval()
        return fresh

    fresh = cfg["builder"](weights=None)
    fresh.load_state_dict(base.state_dict())
    fresh.eval()
    return fresh


def prepare_detection_input(image, transform_fn):
    if image is None:
        raise ValueError("No image provided")
    if not isinstance(image, Image.Image):
        if isinstance(image, np.ndarray) and image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        image = Image.fromarray(np.array(image).astype("uint8"))
    image_rgb = image.convert("RGB")
    tensor = transform_fn(image_rgb)
    if tensor.ndim == 3:
        tensor = tensor
    else:
        tensor = torch.as_tensor(tensor)
    return tensor, image_rgb


def draw_detections(image: Image.Image, detections: list[dict], max_dets: int = 30) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    colors = _SEG_BASE_PALETTE  # reuse palette for variety
    for idx, det in enumerate(detections[:max_dets]):
        box = det["box"]
        color = tuple(int(c) for c in colors[idx % len(colors)])
        draw.rectangle(box, outline=color, width=3)
        label = f"{det['label']} {det['score']:.2f}"
        draw.text((box[0] + 4, box[1] + 4), label, fill=color)
    return canvas


def run_detection_inference(
    model: nn.Module,
    image,
    device: torch.device,
    transform_fn,
    channels_last: bool,
    warmup: bool,
    use_amp: bool,
    score_thresh: float = 0.25,
    backend: str = "torchvision",
    imgsz: int | None = None,
):
    if backend == "ultralytics":
        if image is None:
            raise ValueError("No image provided")
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray) and image.dtype != np.uint8:
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            image = Image.fromarray(np.array(image).astype("uint8"))
        image_rgb = image.convert("RGB")

        device_arg = str(device) if isinstance(device, torch.device) else device
        half = use_amp and isinstance(device, torch.device) and device.type == "cuda"
        if hasattr(model, "model") and isinstance(model.model, nn.Module):
            model.model.to(device)

        if warmup:
            with torch.no_grad():
                model.predict(image_rgb, imgsz=imgsz, device=device_arg, verbose=False, half=half)

        start = time.time()
        with torch.no_grad():
            results = model.predict(image_rgb, imgsz=imgsz, device=device_arg, verbose=False, half=half)
        latency = (time.time() - start) * 1000

        dets: list[dict] = []
        if results:
            res = results[0]
            boxes = getattr(res, "boxes", None)
            if boxes is not None:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                labels = boxes.cls.detach().cpu().numpy()
                for box, score, label_idx in zip(xyxy, confs, labels):
                    if score < score_thresh:
                        continue
                    dets.append(
                        {
                            "label": str(int(label_idx)),
                            "score": float(score),
                            "box": [float(x) for x in box],
                        }
                    )

        return {"detections": dets, "latency": latency, "image": image_rgb}

    tensor, image_rgb = prepare_detection_input(image, transform_fn)
    model = model.to(device)

    batch_tensor = tensor.to(device)
    if channels_last and device.type == "cuda" and batch_tensor.dim() == 4:
        batch_tensor = batch_tensor.to(memory_format=torch.channels_last)
    elif channels_last and device.type == "cuda":
        # Channels-last requires NCHW (4D) input; detection tensors are 3D.
        pass

    if next(model.parameters()).dtype == torch.float16:
        batch_tensor = batch_tensor.half()

    inputs = [batch_tensor]

    if warmup:
        with torch.no_grad():
            model(inputs)

    amp_ctx = torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda")
    start = time.time()
    with torch.no_grad(), amp_ctx:
        outputs = model(inputs)
    latency = (time.time() - start) * 1000

    out = outputs[0]
    boxes = out["boxes"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy()

    dets = []
    for box, score, label_idx in zip(boxes, scores, labels):
        if score < score_thresh:
            continue
        dets.append(
            {
                "label": str(label_idx),
                "score": float(score),
                "box": [float(x) for x in box],
            }
        )

    return {
        "detections": dets,
        "latency": latency,
        "image": image_rgb,
    }


def attach_detection_labels(detections: list[dict], label_names: list[str]) -> list[dict]:
    labeled = []
    for det in detections:
        idx = int(det["label"])
        name = label_names[idx] if idx < len(label_names) else f"Class {idx}"
        labeled.append({**det, "label": name})
    return labeled


def get_detection_state_module(model, backend: str):
    if backend == "ultralytics" and hasattr(model, "model"):
        return model.model
    return model


def build_detection_metrics(
    original_result: dict,
    optimized_result: dict,
    size_original: float,
    size_optimized: float,
    optimized_label: str,
    score_thresh: float,
):
    orig_dets = original_result["detections"]
    opt_dets = optimized_result["detections"]
    mean_score_orig = float(np.mean([d["score"] for d in orig_dets])) if orig_dets else 0.0
    mean_score_opt = float(np.mean([d["score"] for d in opt_dets])) if opt_dets else 0.0

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Latency (ms)",
                f"Detections (score>={score_thresh})",
                "Mean Score",
                "Model Size (MB)",
            ],
            "Original Model": [
                f"{original_result['latency']:.2f}",
                str(len(orig_dets)),
                f"{mean_score_orig:.3f}",
                f"{size_original:.2f}",
            ],
            optimized_label: [
                f"{optimized_result['latency']:.2f}",
                str(len(opt_dets)),
                f"{mean_score_opt:.3f}",
                f"{size_optimized:.2f}",
            ],
        }
    )
    return metrics_df


def build_detection_comparison_df(
    orig_dets: list[dict],
    opt_dets: list[dict],
    optimized_label: str,
    max_rows: int = 50,
) -> pd.DataFrame:
    rows = []
    for det in orig_dets:
        rows.append(
            {
                "Model": "Original",
                "Class": det["label"],
                "Score": round(det["score"], 3),
                "Box [x1,y1,x2,y2]": [round(x, 1) for x in det["box"]],
            }
        )
    for det in opt_dets:
        rows.append(
            {
                "Model": optimized_label,
                "Class": det["label"],
                "Score": round(det["score"], 3),
                "Box [x1,y1,x2,y2]": [round(x, 1) for x in det["box"]],
            }
        )
    if max_rows and len(rows) > max_rows:
        rows = rows[:max_rows]
    return pd.DataFrame(rows)


def run_segmentation_inference(
    model: nn.Module,
    image,
    device: torch.device,
    transform_fn,
    channels_last: bool,
    warmup: bool,
    use_amp: bool,
    class_count: int,
):
    tensor, original_image = transform_fn(image)

    model = model.to(device)
    input_tensor = tensor.unsqueeze(0).to(device)

    if channels_last and device.type == "cuda":
        input_tensor = input_tensor.to(memory_format=torch.channels_last)

    if next(model.parameters()).dtype == torch.float16:
        input_tensor = input_tensor.half()

    if warmup:
        with torch.no_grad():
            model(input_tensor)

    amp_ctx = torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda")
    start = time.time()
    with torch.no_grad(), amp_ctx:
        logits = model(input_tensor)
    latency = (time.time() - start) * 1000

    if isinstance(logits, (list, tuple)):
        logits = logits[0]

    logits = logits.detach().cpu()
    probs = torch.softmax(logits, dim=1)
    mask_tensor = torch.argmax(probs, dim=1)[0]
    mask_processed = mask_tensor.cpu().numpy().astype(np.int64)

    mean_conf = float(probs.max(dim=1)[0].mean().item())

    mask_processed_image = colorize_mask(mask_processed, class_count)
    mask_original_l = Image.fromarray(mask_processed.astype(np.uint8), mode="L").resize(original_image.size, Image.NEAREST)
    mask_original_np = np.array(mask_original_l, dtype=np.int64)
    mask_original_image = colorize_mask(mask_original_np, class_count)
    overlay_original = overlay_mask(original_image, mask_original_image)
    class_summary = summarize_mask(mask_original_np, class_count)

    return {
        "latency": latency,
        "mask_processed": mask_processed,
        "mask_original": mask_original_np,
        "mask_image_processed": mask_processed_image,
        "mask_image_original": mask_original_image,
        "overlay_original": overlay_original,
        "mean_confidence": mean_conf,
        "class_summary": class_summary,
    }


def build_segmentation_metrics(
    original_result: dict,
    optimized_result: dict,
    size_original: float,
    size_optimized: float,
    optimized_label: str,
) -> pd.DataFrame:
    mask_original = original_result["mask_original"]
    mask_optimized = optimized_result["mask_original"]
    agreement = float((mask_original == mask_optimized).mean() * 100.0)

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Latency (ms)",
                "Mean Confidence",
                "Model Size (MB)",
                "Mask Agreement (%)",
            ],
            "Original Model": [
                f"{original_result['latency']:.2f}",
                f"{original_result['mean_confidence']:.4f}",
                f"{size_original:.2f}",
                "100.00",
            ],
            optimized_label: [
                f"{optimized_result['latency']:.2f}",
                f"{optimized_result['mean_confidence']:.4f}",
                f"{size_optimized:.2f}",
                f"{agreement:.2f}",
            ],
        }
    )
    return metrics_df


def build_class_distribution_df(
    original_summary: list[dict[str, float]],
    optimized_summary: list[dict[str, float]],
    labels: list[str],
    optimized_label: str,
    max_rows: int = 25,
) -> pd.DataFrame:
    rows = []
    for idx, label in enumerate(labels):
        orig_entry = original_summary[idx]
        opt_entry = optimized_summary[idx]
        if orig_entry["count"] == 0 and opt_entry["count"] == 0:
            continue
        rows.append(
            {
                "Class": label,
                "Original %": round(orig_entry["percent"], 2),
                f"{optimized_label} %": round(opt_entry["percent"], 2),
                "Original Pixels": orig_entry["count"],
                f"{optimized_label} Pixels": opt_entry["count"],
            }
        )

    rows.sort(key=lambda item: max(item["Original %"], item[f"{optimized_label} %"]), reverse=True)
    if max_rows and len(rows) > max_rows:
        rows = rows[:max_rows]

    return pd.DataFrame(rows)


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


def run_pruned_detection(
    img,
    model_choice,
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
    score_thresh=0.25,
):
    print("\n=== RUN DETECTION PRUNED CALLED ===")
    if img is None:
        print("ERROR: Image is None")
        empty_metrics = pd.DataFrame({"Metric": ["Error"], "Original Model": ["No image"], "Pruned Model": [""]})
        return empty_metrics, None, pd.DataFrame(), []

    if preset in PRESETS:
        preset_cfg = PRESETS[preset]
        device_choice = preset_cfg["device"]
        channels_last = preset_cfg["channels_last"]
        use_compile = preset_cfg["compile"]
        use_amp = preset_cfg.get("amp", use_amp)
        amount = preset_cfg.get("prune_amount", amount)

    device = select_device(device_choice)
    cfg = get_detection_config(model_choice)
    backend = cfg.get("backend", "torchvision")
    imgsz = cfg.get("imgsz")
    labels = get_detection_labels(model_choice)
    transform_fn = get_detection_transform(model_choice)

    base_model = get_detection_model(model_choice)
    original_result = run_detection_inference(
        base_model,
        img,
        device,
        transform_fn,
        channels_last=channels_last,
        warmup=True,
        use_amp=use_amp,
        score_thresh=score_thresh,
        backend=backend,
        imgsz=imgsz,
    )
    original_result["detections"] = attach_detection_labels(original_result["detections"], labels)

    fresh_model = clone_detection_model(model_choice)
    pruned_module = apply_pruning(get_detection_state_module(fresh_model, backend), amount=float(amount), method=method)
    pruned_module = maybe_compile(pruned_module, use_compile)
    if backend == "ultralytics" and hasattr(fresh_model, "model"):
        fresh_model.model = pruned_module
        pruned_model = fresh_model
    else:
        pruned_model = pruned_module
    pruned_result = run_detection_inference(
        pruned_model,
        img,
        device,
        transform_fn,
        channels_last=channels_last,
        warmup=True,
        use_amp=use_amp,
        score_thresh=score_thresh,
        backend=backend,
        imgsz=imgsz,
    )
    pruned_result["detections"] = attach_detection_labels(pruned_result["detections"], labels)

    size_orig = get_state_dict_size_mb(get_detection_state_module(base_model, backend))
    size_pruned = get_state_dict_size_mb(get_detection_state_module(pruned_model, backend))

    metrics_df = build_detection_metrics(
        original_result, pruned_result, size_orig, size_pruned, "Pruned Model", score_thresh
    )
    det_df = build_detection_comparison_df(original_result["detections"], pruned_result["detections"], "Pruned")
    overlay_orig = add_image_label(
        draw_detections(original_result["image"], original_result["detections"]),
        "Original Model",
    )
    overlay_pruned = add_image_label(
        draw_detections(pruned_result["image"], pruned_result["detections"]),
        "Pruned Model",
    )
    overlay_slider_value = (
        overlay_orig,
        overlay_pruned,
    )

    downloads: list[str] = []
    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)
    trace_inputs = None

    if backend != "ultralytics":
        sample_tensor, _ = prepare_detection_input(img, transform_fn)
        sample_batch = [sample_tensor]
        trace_inputs = (sample_batch,)
    else:
        if export_ts or export_onnx:
            print("TorchScript/ONNX export is not enabled for YOLO12 models in this app.")
            export_ts = False
            export_onnx = False

    if export_report:
        report_path = export_dir / "pruned_det_report.json"
        report = {
            "model": model_choice,
            "pruning": {"method": method, "amount": float(amount)},
            "score_threshold": score_thresh,
            "metrics": metrics_df.to_dict(),
            "detections": {
                "original": original_result["detections"],
                "pruned": pruned_result["detections"],
            },
        }
        report_path.write_text(json.dumps(report, indent=2))
        downloads.append(str(report_path))

    if export_state:
        state_path = export_dir / "pruned_det_state_dict.pth"
        torch.save(get_detection_state_module(pruned_model, backend).state_dict(), state_path)
        downloads.append(str(state_path))

    if export_ts and trace_inputs is not None:
        ts_path = export_dir / "pruned_det_model.ts"
        try:
            scripted = torch.jit.trace(pruned_model.cpu(), trace_inputs)
            scripted.save(ts_path)
            downloads.append(str(ts_path))
        except Exception as exc:  # pragma: no cover - export best effort
            print(f"TorchScript export failed: {exc}")

    if export_onnx and trace_inputs is not None:
        onnx_path = export_dir / "pruned_det_model.onnx"
        try:
            torch.onnx.export(
                pruned_model.cpu(),
                trace_inputs,
                onnx_path,
                input_names=["images"],
                output_names=["detections"],
                opset_version=13,
                dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"}},
            )
            downloads.append(str(onnx_path))
        except Exception as exc:  # pragma: no cover - export best effort
            print(f"ONNX export failed: {exc}")

    print("=== RUN DETECTION PRUNED COMPLETE ===")
    return metrics_df, overlay_slider_value, det_df, downloads


def run_quantized_detection(
    img,
    model_choice,
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
    score_thresh=0.25,
):
    print("\n=== RUN DETECTION QUANTIZED CALLED ===")
    if img is None:
        print("ERROR: Image is None")
        empty_metrics = pd.DataFrame({"Metric": ["Error"], "Original Model": ["No image"], "Quantized Model": [""]})
        return empty_metrics, None, pd.DataFrame(), []

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
    cfg = get_detection_config(model_choice)
    backend = cfg.get("backend", "torchvision")
    imgsz = cfg.get("imgsz")

    labels = get_detection_labels(model_choice)
    transform_fn = get_detection_transform(model_choice)
    base_model = get_detection_model(model_choice)

    original_result = run_detection_inference(
        base_model,
        img,
        device,
        transform_fn,
        channels_last=channels_last,
        warmup=True,
        use_amp=use_amp,
        score_thresh=score_thresh,
        backend=backend,
        imgsz=imgsz,
    )
    original_result["detections"] = attach_detection_labels(original_result["detections"], labels)

    fresh_model = clone_detection_model(model_choice)
    quant_module = apply_quantization(get_detection_state_module(fresh_model, backend), q_type)
    quant_module = maybe_compile(quant_module, use_compile)
    if backend == "ultralytics" and hasattr(fresh_model, "model"):
        fresh_model.model = quant_module
        quant_model = fresh_model
    else:
        quant_model = quant_module
    quant_result = run_detection_inference(
        quant_model,
        img,
        device,
        transform_fn,
        channels_last=channels_last,
        warmup=True,
        use_amp=use_amp,
        score_thresh=score_thresh,
        backend=backend,
        imgsz=imgsz,
    )
    quant_result["detections"] = attach_detection_labels(quant_result["detections"], labels)

    size_orig = get_state_dict_size_mb(get_detection_state_module(base_model, backend))
    size_quant = get_state_dict_size_mb(get_detection_state_module(quant_model, backend))
    metrics_df = build_detection_metrics(
        original_result, quant_result, size_orig, size_quant, "Quantized Model", score_thresh
    )
    det_df = build_detection_comparison_df(original_result["detections"], quant_result["detections"], "Quantized")
    overlay_orig = add_image_label(
        draw_detections(original_result["image"], original_result["detections"]),
        "Original Model",
    )
    overlay_quant = add_image_label(
        draw_detections(quant_result["image"], quant_result["detections"]),
        "Quantized Model",
    )
    overlay_slider_value = (
        overlay_orig,
        overlay_quant,
    )

    downloads: list[str] = []
    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)
    trace_inputs = None

    if backend != "ultralytics":
        sample_tensor, _ = prepare_detection_input(img, transform_fn)
        sample_batch = [sample_tensor]
        trace_inputs = (sample_batch,)
    else:
        if export_ts or export_onnx:
            print("TorchScript/ONNX export is not enabled for YOLO12 models in this app.")
            export_ts = False
            export_onnx = False

    if export_report:
        report_path = export_dir / "quant_det_report.json"
        report = {
            "model": model_choice,
            "quantization": q_type,
            "score_threshold": score_thresh,
            "metrics": metrics_df.to_dict(),
            "detections": {
                "original": original_result["detections"],
                "quantized": quant_result["detections"],
            },
        }
        report_path.write_text(json.dumps(report, indent=2))
        downloads.append(str(report_path))

    if export_state:
        state_path = export_dir / "quant_det_state_dict.pth"
        torch.save(get_detection_state_module(quant_model, backend).state_dict(), state_path)
        downloads.append(str(state_path))

    if export_ts and trace_inputs is not None:
        ts_path = export_dir / "quant_det_model.ts"
        try:
            scripted = torch.jit.trace(quant_model.cpu(), trace_inputs)
            scripted.save(ts_path)
            downloads.append(str(ts_path))
        except Exception as exc:  # pragma: no cover - export best effort
            print(f"TorchScript export failed: {exc}")

    if export_onnx and trace_inputs is not None:
        onnx_path = export_dir / "quant_det_model.onnx"
        try:
            torch.onnx.export(
                quant_model.cpu(),
                trace_inputs,
                onnx_path,
                input_names=["images"],
                output_names=["detections"],
                opset_version=13,
                dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"}},
            )
            downloads.append(str(onnx_path))
        except Exception as exc:  # pragma: no cover - export best effort
            print(f"ONNX export failed: {exc}")

    print("=== RUN DETECTION QUANTIZED COMPLETE ===")
    return metrics_df, overlay_slider_value, det_df, downloads


def run_pruned_segmentation(
    img,
    model_choice,
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
    print("\n=== RUN SEGMENTATION PRUNED CALLED ===")
    if img is None:
        print("ERROR: Image is None")
        empty_metrics = pd.DataFrame({"Metric": ["Error"], "Original Model": ["No image"], "Pruned Model": [""]})
        empty_dist = pd.DataFrame({"Class": [], "Original %": [], "Pruned %": []})
        return empty_metrics, empty_dist, None, None, pd.DataFrame(), []

    config = SEGMENTATION_MODEL_MAP.get(model_choice, SEGMENTATION_MODEL_CONFIGS[0])

    if preset in PRESETS:
        preset_cfg = PRESETS[preset]
        device_choice = preset_cfg["device"]
        channels_last = preset_cfg["channels_last"]
        use_compile = preset_cfg["compile"]
        use_amp = preset_cfg.get("amp", use_amp)
        amount = preset_cfg.get("prune_amount", amount)

    device = select_device(device_choice)

    base_model = get_segmentation_model(config)
    transform_fn = get_segmentation_transform(config)
    class_labels = get_class_labels(config)
    class_count = config.classes

    original_result = run_segmentation_inference(
        base_model,
        img,
        device,
        transform_fn,
        channels_last=channels_last,
        warmup=True,
        use_amp=use_amp,
        class_count=class_count,
    )

    fresh_model = clone_segmentation_model(config)
    pruned_model = apply_pruning(fresh_model, amount=float(amount), method=method)
    pruned_model = maybe_compile(pruned_model, use_compile)
    pruned_result = run_segmentation_inference(
        pruned_model,
        img,
        device,
        transform_fn,
        channels_last=channels_last,
        warmup=True,
        use_amp=use_amp,
        class_count=class_count,
    )

    size_orig = get_state_dict_size_mb(base_model)
    size_pruned = get_state_dict_size_mb(pruned_model)
    metrics_df = build_segmentation_metrics(original_result, pruned_result, size_orig, size_pruned, "Pruned Model")
    class_df = build_class_distribution_df(
        original_result["class_summary"],
        pruned_result["class_summary"],
        class_labels,
        "Pruned",
    )

    # Add labels to images for slider comparison
    overlay_orig_labeled = add_image_label(original_result["overlay_original"], "Original Model")
    overlay_pruned_labeled = add_image_label(pruned_result["overlay_original"], "Pruned Model")
    mask_orig_labeled = add_image_label(original_result["mask_image_original"], "Original Mask")
    mask_pruned_labeled = add_image_label(pruned_result["mask_image_original"], "Pruned Mask")
    
    overlay_slider_value = (
        overlay_orig_labeled,
        overlay_pruned_labeled,
    )
    mask_slider_value = (
        mask_orig_labeled,
        mask_pruned_labeled,
    )
    sparsity_df = compute_sparsity(pruned_model.cpu())

    downloads: list[str] = []
    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)

    if export_report:
        report_path = export_dir / "pruned_seg_report.json"
        report = {
            "model": config.name,
            "checkpoint": config.checkpoint,
            "dataset": config.dataset,
            "pruning": {"method": method, "amount": float(amount)},
            "metrics": metrics_df.to_dict(),
            "class_distribution": class_df.to_dict(),
        }
        report_path.write_text(json.dumps(report, indent=2))
        downloads.append(str(report_path))

    if export_state:
        state_path = export_dir / "pruned_seg_state_dict.pth"
        torch.save(pruned_model.state_dict(), state_path)
        downloads.append(str(state_path))

    sample_tensor, _ = transform_fn(img)
    sample_batch = sample_tensor.unsqueeze(0)

    if export_ts:
        ts_path = export_dir / "pruned_seg_model.ts"
        try:
            scripted = torch.jit.trace(pruned_model.cpu(), sample_batch)
            scripted.save(ts_path)
            downloads.append(str(ts_path))
        except Exception as exc:  # pragma: no cover - export best effort
            print(f"TorchScript export failed: {exc}")

    if export_onnx:
        onnx_path = export_dir / "pruned_seg_model.onnx"
        try:
            torch.onnx.export(
                pruned_model.cpu(),
                sample_batch,
                onnx_path,
                input_names=["input"],
                output_names=["mask"],
                opset_version=13,
                dynamic_axes={"input": {0: "batch"}, "mask": {0: "batch"}},
            )
            downloads.append(str(onnx_path))
        except Exception as exc:  # pragma: no cover - export best effort
            print(f"ONNX export failed: {exc}")

    return (
        metrics_df,
        class_df,
        overlay_slider_value,
        mask_slider_value,
        sparsity_df,
        downloads,
    )


def run_quantized_segmentation(
    img,
    model_choice,
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
    print("\n=== RUN SEGMENTATION QUANTIZED CALLED ===")
    if img is None:
        print("ERROR: Image is None")
        empty_metrics = pd.DataFrame({"Metric": ["Error"], "Original Model": ["No image"], "Quantized Model": [""]})
        empty_dist = pd.DataFrame({"Class": [], "Original %": [], "Quantized %": []})
        return empty_metrics, empty_dist, None, None, []

    config = SEGMENTATION_MODEL_MAP.get(model_choice, SEGMENTATION_MODEL_CONFIGS[0])

    if preset in PRESETS:
        preset_cfg = PRESETS[preset]
        device_choice = preset_cfg["device"]
        channels_last = preset_cfg["channels_last"]
        use_compile = preset_cfg["compile"]
        use_amp = preset_cfg.get("amp", use_amp)
        q_type = preset_cfg.get("quant", q_type)

    device = select_device(device_choice)
    if q_type in {"dynamic", "weight_only"} and device.type != "cpu":
        print("Dynamic quantization runs on CPU; switching device to CPU.")
        device = torch.device("cpu")
        channels_last = False
        use_amp = False

    base_model = get_segmentation_model(config)
    transform_fn = get_segmentation_transform(config)
    class_labels = get_class_labels(config)
    class_count = config.classes

    original_result = run_segmentation_inference(
        base_model,
        img,
        device,
        transform_fn,
        channels_last=channels_last,
        warmup=True,
        use_amp=use_amp,
        class_count=class_count,
    )

    fresh_model = clone_segmentation_model(config)
    quant_model = apply_quantization(fresh_model, q_type)
    quant_model = maybe_compile(quant_model, use_compile)

    quant_result = run_segmentation_inference(
        quant_model,
        img,
        device,
        transform_fn,
        channels_last=channels_last,
        warmup=True,
        use_amp=use_amp,
        class_count=class_count,
    )

    size_orig = get_state_dict_size_mb(base_model)
    size_quant = get_state_dict_size_mb(quant_model)
    metrics_df = build_segmentation_metrics(original_result, quant_result, size_orig, size_quant, "Quantized Model")
    class_df = build_class_distribution_df(
        original_result["class_summary"],
        quant_result["class_summary"],
        class_labels,
        "Quantized",
    )

    # Add labels to images for slider comparison
    overlay_orig_labeled = add_image_label(original_result["overlay_original"], "Original Model")
    overlay_quant_labeled = add_image_label(quant_result["overlay_original"], "Quantized Model")
    mask_orig_labeled = add_image_label(original_result["mask_image_original"], "Original Mask")
    mask_quant_labeled = add_image_label(quant_result["mask_image_original"], "Quantized Mask")
    
    overlay_slider_value = (
        overlay_orig_labeled,
        overlay_quant_labeled,
    )
    mask_slider_value = (
        mask_orig_labeled,
        mask_quant_labeled,
    )

    downloads: list[str] = []
    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)

    if export_report:
        report_path = export_dir / "quant_seg_report.json"
        report = {
            "model": config.name,
            "checkpoint": config.checkpoint,
            "dataset": config.dataset,
            "quantization": q_type,
            "metrics": metrics_df.to_dict(),
            "class_distribution": class_df.to_dict(),
        }
        report_path.write_text(json.dumps(report, indent=2))
        downloads.append(str(report_path))

    if export_state:
        state_path = export_dir / "quant_seg_state_dict.pth"
        torch.save(quant_model.state_dict(), state_path)
        downloads.append(str(state_path))

    sample_tensor, _ = transform_fn(img)
    sample_batch = sample_tensor.unsqueeze(0)

    if export_ts:
        ts_path = export_dir / "quant_seg_model.ts"
        try:
            scripted = torch.jit.trace(quant_model.cpu(), sample_batch)
            scripted.save(ts_path)
            downloads.append(str(ts_path))
        except Exception as exc:  # pragma: no cover - export best effort
            print(f"TorchScript export failed: {exc}")

    if export_onnx:
        onnx_path = export_dir / "quant_seg_model.onnx"
        try:
            torch.onnx.export(
                quant_model.cpu(),
                sample_batch,
                onnx_path,
                input_names=["input"],
                output_names=["mask"],
                opset_version=13,
                dynamic_axes={"input": {0: "batch"}, "mask": {0: "batch"}},
            )
            downloads.append(str(onnx_path))
        except Exception as exc:  # pragma: no cover - export best effort
            print(f"ONNX export failed: {exc}")

    return (
        metrics_df,
        class_df,
        overlay_slider_value,
        mask_slider_value,
        downloads,
    )
# ---------------------------------------------
# GRADIO UI
# ---------------------------------------------
examples = [["examples/cat.jpg"], ["examples/dog.jpg"], ["examples/bird.jpg"], ["examples/car.jpg"], ["examples/elephant.jpg"]]
ade_examples = [["examples/ADE_val_00000001.jpg"], ["examples/ADE_val_00000002.jpg"]]


def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# 🧠 Model Optimization Lab — Compare, Export, Benchmark")

        device_opts = ["auto", "cpu"]
        if torch.cuda.is_available():
            device_opts.append("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device_opts.append("mps")
        preset_opts = list(PRESETS.keys()) + ["custom"]
        seg_model_options = [cfg.name for cfg in SEGMENTATION_MODEL_CONFIGS]
        det_model_options = DETECTION_MODEL_OPTIONS.copy()

        with gr.Tabs():
            # ---- PRUNING TAB ----
            with gr.Tab("Pruning-Classification"):
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
                        gr.Markdown(
                            "### 📚 Classification Pruning Guide\n\n"
                            "**What is Pruning?**\n"
                            "Pruning removes less important weights from neural networks to reduce model size and potentially improve inference speed. "
                            "This tab applies pruning to ImageNet classification models.\n\n"
                            "**Options Explained:**\n"
                            "- **Base Model**: Select from 7 pretrained architectures (ResNet-50, MobileNetV3, EfficientNet-B0, ConvNeXt-Tiny, ViT-Base, RegNetY-016, EfficientNet-Lite0). Each has different size/accuracy tradeoffs.\n"
                            "- **Hardware Preset**: Quick configurations for common deployment scenarios:\n"
                            "  - *Edge CPU*: Optimized for resource-constrained devices (CPU-only, 30% pruning, dynamic quantization)\n"
                            "  - *Datacenter GPU*: Maximum performance on modern GPUs (CUDA, channels-last, compile, 20% pruning)\n"
                            "  - *Apple MPS*: Tuned for Apple Silicon (M1/M2/M3 chips with Metal Performance Shaders)\n"
                            "  - *Custom*: Manual control over all settings\n"
                            "- **Pruning Method**:\n"
                            "  - *Structured*: Removes entire filters/channels; better hardware support and actual speedups\n"
                            "  - *Unstructured*: Zeros individual weights; higher compression but needs specialized sparse kernels for speedup\n"
                            "- **Pruning Amount**: Percentage of weights to remove (0.1 = 10%, 0.9 = 90%). Higher values = smaller model but potential accuracy loss.\n"
                            "- **Device**: Inference hardware (auto-detects best available: CUDA → MPS → CPU)\n"
                            "- **Channels-last (CUDA only)**: Memory layout optimization for faster convolution operations on NVIDIA GPUs\n"
                            "- **Mixed Precision (AMP)**: Uses FP16 where safe, FP32 where needed; faster on modern GPUs with Tensor Cores\n"
                            "- **Torch Compile**: PyTorch 2.0+ graph optimization; can provide 20-40% speedup but adds compilation overhead\n\n"
                            "**Export Options:**\n"
                            "- *TorchScript*: Serialized model for C++ deployment or production serving\n"
                            "- *ONNX*: Cross-framework format (TensorRT, OpenVINO, ONNX Runtime, CoreML)\n"
                            "- *JSON Report*: Detailed metrics, settings, and Top-5 predictions for both models\n"
                            "- *State Dict*: Always saved; PyTorch checkpoint for loading pruned weights later\n\n"
                            "**Reading the Results:**\n"
                            "- *Comparison Metrics*: Side-by-side accuracy, speed, and size\n"
                            "- *Top-5 Chart*: Visual comparison of prediction confidence across models\n"
                            "- *Layer Sparsity*: Per-layer breakdown showing which parts were pruned most"
                        )

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
            with gr.Tab("Quantization-Classification"):
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
                        gr.Markdown(
                            "### 📚 Classification Quantization Guide\n\n"
                            "**What is Quantization?**\n"
                            "Quantization reduces model precision from 32-bit floats to lower bit-widths (INT8, FP16), decreasing memory usage and "
                            "enabling faster inference on hardware with specialized low-precision instructions.\n\n"
                            "**Options Explained:**\n"
                            "- **Base Model**: Choose from 7 pretrained ImageNet classifiers with varying complexity.\n"
                            "- **Hardware Preset**: Same presets as pruning tab, but with quantization-specific defaults.\n"
                            "- **Quantization Type**:\n"
                            "  - *Dynamic*: Post-training INT8 quantization on linear layers; activations quantized dynamically at runtime. **Forces CPU** (PyTorch's INT8 kernels are CPU-only). Best for transformers and MLP-heavy models.\n"
                            "  - *Weight-only*: Stores weights as INT8, computes in FP32. Reduces memory bandwidth, smaller model files. **CPU-optimized**.\n"
                            "  - *FP16*: Half-precision floating point; requires GPU with FP16 support (CUDA, MPS). Minimal accuracy loss, ~2x speedup on modern GPUs.\n"
                            "- **Device**: Hardware target (dynamic/weight-only auto-switch to CPU for kernel compatibility)\n"
                            "- **Channels-last**: CUDA memory layout optimization (ignored on CPU)\n"
                            "- **Mixed Precision (AMP)**: Can combine with FP16 quantization on GPUs\n"
                            "- **Torch Compile**: Graph-level optimizations from PyTorch 2.0+\n\n"
                            "**Export Options:** Same as pruning (TorchScript, ONNX, JSON report, state dict)\n\n"
                            "**Important Notes:**\n"
                            "⚠️ Dynamic/weight-only quantization automatically uses CPU even if GPU is selected (PyTorch limitation)\n"
                            "⚠️ ResNet-50 and similar CNN-heavy models see modest INT8 speedups because only linear layers are quantized\n"
                            "⚠️ FP16 on CPU often reverts to FP32 internally, adding overhead instead of speedup\n\n"
                            "**Reading the Results:**\n"
                            "- *Latency*: Dynamic quantization may show higher latency due to runtime overhead; production deployments should use cached models\n"
                            "- *Model Size*: FP16 ≈ 50% reduction, INT8 dynamic ≈ 75% reduction (varies by architecture)\n"
                            "- *Accuracy*: Watch for confidence drops; quantization can shift predictions slightly"
                        )
                        

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

            # ---- DETECTION PRUNING TAB ----
            with gr.Tab("Pruning-Detection"):
                with gr.Row():
                    with gr.Column():
                        img_dp = gr.Image(label="Upload Image")
                        model_dp = gr.Dropdown(det_model_options, value=det_model_options[0], label="Object Detector (COCO)")
                        preset_dp = gr.Dropdown(preset_opts, value="custom", label="Hardware Preset")
                        method_dp = gr.Dropdown(["unstructured", "structured"], value="structured", label="Pruning Method")
                        amount_dp = gr.Slider(minimum=0.1, maximum=0.9, step=0.1, value=0.3, label="Pruning Amount")
                        score_dp = gr.Slider(minimum=0.05, maximum=0.9, step=0.05, value=0.25, label="Score Threshold")
                        device_dp = gr.Dropdown(device_opts, value=device_opts[0], label="Device")
                        channels_last_dp = gr.Checkbox(label="Channels-last input (CUDA)", value=True)
                        amp_dp = gr.Checkbox(label="Mixed precision (AMP)", value=True)
                        compile_dp = gr.Checkbox(label="Torch compile (PyTorch 2)")
                        export_ts_dp = gr.Checkbox(label="Export TorchScript")
                        export_onnx_dp = gr.Checkbox(label="Export ONNX")
                        export_report_dp = gr.Checkbox(label="Export JSON report", value=True)
                        btn_dp = gr.Button("Run Detection Pruning")
                        gr.Examples(examples=examples, inputs=img_dp)
                        gr.Markdown(
                            "### 🦾 Detection Pruning Guide\n\n"
                            "**Models:**\n"
                            "- TorchVision: Faster R-CNN ResNet50 FPN, SSDlite320 MobileNetV3 (COCO pretrained)\n"
                            "- Ultralytics YOLO12: sizes n/s/m/l/x (COCO, auto-downloaded if missing)\n\n"
                            "**Core Options:**\n"
                            "- *Hardware Preset*: Same CPU/GPU defaults as classification; channels-last only applies on CUDA.\n"
                            "- *Pruning Method*: Structured is safest for detection heads; unstructured yields higher sparsity but rarely speeds up NMS.\n"
                            "- *Score Threshold*: Filters low-confidence boxes before metrics/overlays.\n"
                            "- *AMP / Torch Compile*: Only useful on GPU; compile adds startup cost but can speed up steady-state.\n"
                            "- *YOLO12 exports*: TorchScript/ONNX disabled here; state_dict still saved for the underlying torch model.\n\n"
                            "**Reading Results:**\n"
                            "- Metrics: latency, box count above threshold, mean score, model size.\n"
                            "- Overlay slider: drag to compare original vs pruned detections.\n"
                            "- Detections table: flattened list of boxes for quick scanning."
                        )

                    with gr.Column():
                        metrics_dp = gr.Dataframe(label="📊 Detection Metrics", headers=["Metric", "Original Model", "Pruned Model"])
                        overlay_dp = gr.ImageSlider(label="Overlay Comparison", type="pil")
                        dets_dp = gr.Dataframe(label="Detections (Original vs Pruned)")
                        downloads_dp = gr.Files(label="Exports (state_dict / TorchScript / ONNX / report)")

                btn_dp.click(
                    fn=run_pruned_detection,
                    inputs=[
                        img_dp,
                        model_dp,
                        method_dp,
                        amount_dp,
                        device_dp,
                        channels_last_dp,
                        compile_dp,
                        amp_dp,
                        export_ts_dp,
                        export_onnx_dp,
                        export_report_dp,
                        gr.State(True),
                        preset_dp,
                        score_dp,
                    ],
                    outputs=[
                        metrics_dp,
                        overlay_dp,
                        dets_dp,
                        downloads_dp,
                    ],
                )

            # ---- DETECTION QUANTIZATION TAB ----
            with gr.Tab("Quantization-Detection"):
                with gr.Row():
                    with gr.Column():
                        img_dq = gr.Image(label="Upload Image")
                        model_dq = gr.Dropdown(det_model_options, value=det_model_options[0], label="Object Detector (COCO)")
                        preset_dq = gr.Dropdown(preset_opts, value="custom", label="Hardware Preset")
                        q_type_dq = gr.Dropdown(["dynamic", "weight_only", "fp16"], value="dynamic", label="Quantization Type")
                        score_dq = gr.Slider(minimum=0.05, maximum=0.9, step=0.05, value=0.25, label="Score Threshold")
                        device_dq = gr.Dropdown(device_opts, value=device_opts[0], label="Device")
                        channels_last_dq = gr.Checkbox(label="Channels-last input (CUDA)", value=True)
                        amp_dq = gr.Checkbox(label="Mixed precision (AMP)", value=True)
                        compile_dq = gr.Checkbox(label="Torch compile (PyTorch 2)")
                        export_ts_dq = gr.Checkbox(label="Export TorchScript")
                        export_onnx_dq = gr.Checkbox(label="Export ONNX")
                        export_report_dq = gr.Checkbox(label="Export JSON report", value=True)
                        btn_dq = gr.Button("Run Detection Quantization")
                        gr.Examples(examples=examples, inputs=img_dq)
                        gr.Markdown(
                            "### ⚡ Detection Quantization Guide\n\n"
                            "**Models:** TorchVision detectors and YOLO12 n/s/m/l/x (Ultralytics). YOLO12 uses its internal preprocessing; other models use TorchVision transforms.\n\n"
                            "**Quantization Modes:**\n"
                            "- *Dynamic / Weight-only*: INT8 linear layers on CPU. UI auto-switches to CPU even if GPU selected (PyTorch limitation).\n"
                            "- *FP16*: Half precision for CUDA/MPS; keeps CPU in FP32. Pair with AMP + channels-last for best GPU speed.\n\n"
                            "**Tips:**\n"
                            "- Score threshold trims noisy boxes before metrics/overlays.\n"
                            "- TorchScript/ONNX exports are skipped for YOLO12; state_dict still saved. TorchVision exports remain enabled.\n"
                            "- For fastest runs, keep AMP + channels-last on CUDA; disable compile if you only run a single image.\n\n"
                            "**Outputs:** Metrics table, overlay slider, detections table, and exports in `exports/` with `_det` suffix."
                        )

                    with gr.Column():
                        metrics_dq = gr.Dataframe(label="📊 Detection Metrics", headers=["Metric", "Original Model", "Quantized Model"])
                        overlay_dq = gr.ImageSlider(label="Overlay Comparison", type="pil")
                        dets_dq = gr.Dataframe(label="Detections (Original vs Quantized)")
                        downloads_dq = gr.Files(label="Exports (state_dict / TorchScript / ONNX / report)")

                btn_dq.click(
                    fn=run_quantized_detection,
                    inputs=[
                        img_dq,
                        model_dq,
                        q_type_dq,
                        device_dq,
                        channels_last_dq,
                        compile_dq,
                        amp_dq,
                        export_ts_dq,
                        export_onnx_dq,
                        export_report_dq,
                        gr.State(True),
                        preset_dq,
                        score_dq,
                    ],
                    outputs=[
                        metrics_dq,
                        overlay_dq,
                        dets_dq,
                        downloads_dq,
                    ],
                )

            # ---- SEGMENTATION PRUNING TAB ----
            with gr.Tab("Pruning-Segmentation"):
                with gr.Row():
                    with gr.Column():
                        img_sp = gr.Image(label="Upload Image")
                        model_sp = gr.Dropdown(seg_model_options, value=seg_model_options[0], label="Pretrained ADE20K Model")
                        preset_sp = gr.Dropdown(preset_opts, value="custom", label="Hardware Preset")
                        method_sp = gr.Dropdown(["unstructured", "structured"], value="structured", label="Pruning Method")
                        amount_sp = gr.Slider(minimum=0.1, maximum=0.9, step=0.1, value=0.4, label="Pruning Amount")
                        device_sp = gr.Dropdown(device_opts, value=device_opts[0], label="Device")
                        channels_last_sp = gr.Checkbox(label="Channels-last input (CUDA)", value=True)
                        compile_sp = gr.Checkbox(label="Torch compile (PyTorch 2)")
                        amp_sp = gr.Checkbox(label="Mixed precision (AMP)", value=True)
                        export_ts_sp = gr.Checkbox(label="Export TorchScript")
                        export_onnx_sp = gr.Checkbox(label="Export ONNX")
                        export_report_sp = gr.Checkbox(label="Export JSON report", value=True)
                        btn_sp = gr.Button("Run Segmentation Pruning")
                        gr.Examples(examples=ade_examples, inputs=img_sp, label="ADE20K Samples")
                        gr.Markdown(
                            "### 🎨 Segmentation Pruning Guide\n\n"
                            "**What is Semantic Segmentation?**\n"
                            "Semantic segmentation assigns a class label to every pixel in an image (e.g., sky, road, person, car). "
                            "This tab uses ADE20K-pretrained models that recognize 150 scene categories.\n\n"
                            "**Available Models:**\n"
                            "- **SegFormer B0** (512x512): Lightweight transformer-based segmenter; efficient for edge deployment\n"
                            "- **SegFormer B4** (512x512): Larger variant with better accuracy; ~4x B0 parameters\n"
                            "- **DPT Large**: Vision-transformer-based dense prediction; state-of-the-art accuracy but slower\n"
                            "- **UPerNet ConvNeXt-Tiny**: Unified perceptual parsing with modern CNN backbone; balanced speed/accuracy\n\n"
                            "**Segmentation-Specific Options:**\n"
                            "- All pruning/device/compile options work the same as classification\n"
                            "- Models use [smp-hub](https://huggingface.co/smp-hub) pretrained checkpoints via `segmentation-models-pytorch`\n"
                            "- Preprocessing pipelines are model-specific (loaded from Hugging Face metadata)\n"
                            "- Images are resized based on model training resolution (usually 512x512 or 640x640)\n\n"
                            "**Understanding Segmentation Outputs:**\n"
                            "1. **Comparison Metrics Table**:\n"
                            "   - *Latency*: Inference time for full-image segmentation\n"
                            "   - *Mean Confidence*: Average softmax probability across all pixels\n"
                            "   - *Model Size*: State dict size in MB\n"
                            "   - *Mask Agreement*: % of pixels with identical class predictions (100% = perfect match)\n"
                            "2. **Class Distribution Table**:\n"
                            "   - Top 25 most prevalent classes by pixel coverage\n"
                            "   - Shows percentage and pixel counts for both models\n"
                            "   - Helps identify which objects dominate the scene\n"
                            "3. **Overlay Comparison Slider**:\n"
                            "   - Original image blended with colored segmentation masks\n"
                            "   - Drag slider to compare original vs. pruned predictions\n"
                            "   - Colors map to specific ADE20K classes (150 categories)\n"
                            "4. **Mask Comparison Slider**:\n"
                            "   - Raw segmentation masks without image overlay\n"
                            "   - Easier to spot subtle prediction differences\n"
                            "5. **Layer Sparsity Table**:\n"
                            "   - Per-layer pruning statistics showing compression levels\n\n"
                            "**Export Options:**\n"
                            "Files saved with `_seg` suffix: `pruned_seg_model.ts`, `pruned_seg_report.json`, etc.\n\n"
                            "**Tips:**\n"
                            "- Use ADE20K validation images (provided examples) for meaningful class diversity\n"
                            "- High mask agreement (>95%) indicates pruning preserved segmentation quality\n"
                            "- Check class distribution to ensure dominant objects aren't misclassified\n"
                            "- Structured pruning typically maintains better segmentation quality than unstructured"
                        )

                    with gr.Column():
                        metrics_sp = gr.Dataframe(label="📊 Comparison Metrics")
                        class_sp = gr.Dataframe(label="📈 Class Distribution")
                        overlay_slider_sp = gr.ImageSlider(label="Overlay Comparison", type="pil")
                        mask_slider_sp = gr.ImageSlider(label="Mask Comparison", type="pil")
                        sparsity_sp = gr.Dataframe(label="Layer sparsity (%)")
                        downloads_sp = gr.Files(label="Exports (state_dict / TorchScript / ONNX / report)")

                btn_sp.click(
                    fn=run_pruned_segmentation,
                    inputs=[
                        img_sp,
                        model_sp,
                        method_sp,
                        amount_sp,
                        device_sp,
                        channels_last_sp,
                        compile_sp,
                        amp_sp,
                        export_ts_sp,
                        export_onnx_sp,
                        export_report_sp,
                        gr.State(True),
                        preset_sp,
                    ],
                    outputs=[
                        metrics_sp,
                        class_sp,
                        overlay_slider_sp,
                        mask_slider_sp,
                        sparsity_sp,
                        downloads_sp,
                    ],
                )

            # ---- SEGMENTATION QUANTIZATION TAB ----
            with gr.Tab("Quantization-Segmentation"):
                with gr.Row():
                    with gr.Column():
                        img_sq = gr.Image(label="Upload Image")
                        model_sq = gr.Dropdown(seg_model_options, value=seg_model_options[0], label="Pretrained ADE20K Model")
                        preset_sq = gr.Dropdown(preset_opts, value="custom", label="Hardware Preset")
                        q_type_sq = gr.Dropdown(["dynamic", "weight_only", "fp16"], value="dynamic", label="Quantization Type")
                        device_sq = gr.Dropdown(device_opts, value=device_opts[0], label="Device")
                        channels_last_sq = gr.Checkbox(label="Channels-last input (CUDA)", value=True)
                        compile_sq = gr.Checkbox(label="Torch compile (PyTorch 2)")
                        amp_sq = gr.Checkbox(label="Mixed precision (AMP)", value=True)
                        export_ts_sq = gr.Checkbox(label="Export TorchScript")
                        export_onnx_sq = gr.Checkbox(label="Export ONNX")
                        export_report_sq = gr.Checkbox(label="Export JSON report", value=True)
                        btn_sq = gr.Button("Run Segmentation Quantization")
                        gr.Examples(examples=ade_examples, inputs=img_sq, label="ADE20K Samples")
                        gr.Markdown(
                            "### 🎨 Segmentation Quantization Guide\n\n"
                            "**Quantization for Dense Prediction:**\n"
                            "Semantic segmentation models are typically larger and slower than classifiers, making quantization especially valuable. "
                            "This tab applies the same quantization techniques as classification but evaluates pixel-level prediction quality.\n\n"
                            "**Available Models & Quantization:**\n"
                            "- **SegFormer B0/B4**: Transformer-based; dynamic quantization helps with attention/MLP layers (CPU-only)\n"
                            "- **DPT Large**: Vision-transformer backbone; benefits significantly from FP16 on GPU (~2x speedup)\n"
                            "- **UPerNet ConvNeXt-Tiny**: CNN-based; FP16 quantization provides best GPU acceleration\n\n"
                            "**Quantization Type Selection:**\n"
                            "- **Dynamic/Weight-only**: ⚠️ Automatically uses CPU (PyTorch INT8 limitation). Best for:  \n"
                            "  - Transformer-heavy models (SegFormer, DPT)\n"
                            "  - CPU-only deployment scenarios\n"
                            "  - Memory-constrained environments\n"
                            "- **FP16**: Recommended for GPU deployment (CUDA, MPS). Provides:\n"
                            "  - ~2x inference speedup on modern GPUs\n"
                            "  - 50% memory reduction\n"
                            "  - Minimal segmentation quality loss (<1% mIoU typically)\n\n"
                            "**Segmentation-Specific Metrics:**\n"
                            "1. **Mask Agreement**: Critical metric for segmentation; >95% is good, >98% is excellent\n"
                            "2. **Mean Confidence**: Should remain similar; large drops indicate quantization instability\n"
                            "3. **Class Distribution**: Compare pixel percentages; mismatches show which objects are affected\n\n"
                            "**Understanding the Outputs:**\n"
                            "- **Overlay Slider**: Drag to compare original vs. quantized predictions on the actual image\n"
                            "- **Mask Slider**: Raw segmentation masks for detailed comparison\n"
                            "- **Class Distribution**: Top 25 classes help identify systematic errors (e.g., 'road' → 'sidewalk' confusion)\n\n"
                            "**Performance Expectations:**\n"
                            "- **FP16 on CUDA**: Expect 1.5-2x speedup with <1% accuracy loss\n"
                            "- **Dynamic on CPU**: Model size ↓ 75%, latency may increase (first-run overhead)\n"
                            "- **Weight-only on CPU**: Model size ↓ 50%, latency similar to FP32\n\n"
                            "**Export Options:**\n"
                            "Files saved with `_seg` suffix: `quant_seg_model.onnx`, `quant_seg_state_dict.pth`, etc.\n\n"
                            "**Best Practices:**\n"
                            "✓ Use FP16 for GPU deployment (CUDA, MPS)\n"
                            "✓ Use dynamic quantization for CPU-bound transformer models\n"
                            "✓ Check mask agreement before deploying; <90% needs investigation\n"
                            "✓ Validate on multiple images; some scenes may be more sensitive to quantization\n"
                            "✗ Avoid FP16 on CPU (performance penalty, not benefit)\n"
                            "✗ Don't expect large speedups from dynamic quantization on CNN-heavy models (most layers are Conv2d, not Linear)"
                        )

                    with gr.Column():
                        metrics_sq = gr.Dataframe(label="📊 Comparison Metrics")
                        class_sq = gr.Dataframe(label="📈 Class Distribution")
                        overlay_slider_sq = gr.ImageSlider(label="Overlay Comparison", type="pil")
                        mask_slider_sq = gr.ImageSlider(label="Mask Comparison", type="pil")
                        downloads_sq = gr.Files(label="Exports (state_dict / TorchScript / ONNX / report)")

                btn_sq.click(
                    fn=run_quantized_segmentation,
                    inputs=[
                        img_sq,
                        model_sq,
                        q_type_sq,
                        device_sq,
                        channels_last_sq,
                        compile_sq,
                        amp_sq,
                        export_ts_sq,
                        export_onnx_sq,
                        export_report_sq,
                        gr.State(True),
                        preset_sq,
                    ],
                    outputs=[
                        metrics_sq,
                        class_sq,
                        overlay_slider_sq,
                        mask_slider_sq,
                        downloads_sq,
                    ],
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
    demo.launch(theme = gr.themes.Soft())


if __name__ == "__main__":
    main()
