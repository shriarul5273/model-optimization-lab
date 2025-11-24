import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import timm
import gradio as gr
from torchvision import transforms
from PIL import Image
import time
import os
import pandas as pd


# ---------------------------------------------
# Base FP32 Model (Loaded Once)
# ---------------------------------------------
fp32_model = timm.create_model("resnet50", pretrained=True)
fp32_model.eval()


# ---------------------------------------------
# Image Preprocess
# ---------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Get ImageNet labels - using class descriptions
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

    return model


# ---------------------------------------------
# QUANTIZATION FUNCTION (dynamic)
# ---------------------------------------------
def apply_quantization(model, q_type="dynamic"):
    if q_type == "dynamic":
        return torch.ao.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )

    elif q_type == "weight_only":
        model_int8 = model
        for name, module in model_int8.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = torch.quantize_per_tensor(
                    module.weight.data, scale=0.1, zero_point=0, dtype=torch.qint8
                ).dequantize()
        return model_int8

    elif q_type == "fp16":
        return model.half().eval()

    return model


# ---------------------------------------------
# Inference Function (shared)
# ---------------------------------------------
def run_inference(model, image):
    print(f"  run_inference called, image type: {type(image)}")
    if image is None:
        raise ValueError("No image provided")
    
    if not isinstance(image, Image.Image):
        print("  Converting numpy array to PIL Image")
        image = Image.fromarray(image.astype('uint8'))

    print("  Applying transforms...")
    img = transform(image).unsqueeze(0)

    if next(model.parameters()).dtype == torch.float16:
        img = img.half()

    print("  Running model inference...")
    start = time.time()
    with torch.no_grad():
        out = model(img)
    latency = (time.time() - start) * 1000

    print("  Processing results...")
    prob = torch.softmax(out, dim=1)[0]
    top5_prob, top5_idx = torch.topk(prob, 5)

    results = [(labels[i], float(top5_prob[j])) for j, i in enumerate(top5_idx)]
    print(f"  Inference complete. Top prediction: {results[0][0]}")
    return results, latency


# ---------------------------------------------
# Gradio Functions With Options
# ---------------------------------------------
def run_pruned(img, method, amount):
    print("\n=== RUN PRUNED CALLED ===")
    print(f"Image type: {type(img)}, Method: {method}, Amount: {amount}")
    
    if img is None:
        print("ERROR: Image is None")
        return {"Metric": ["Error"], "Original Model": ["No image uploaded"], "Pruned Model": [""]}, {"Class": [], "Original": [], "Pruned": []}
    
    # Run original model
    print("Running original model...")
    results_orig, latency_orig = run_inference(fp32_model, img)
    print(f"Original model done. Latency: {latency_orig:.2f}ms")
    
    # Run pruned model
    print("Creating fresh model...")
    fresh_model = timm.create_model("resnet50", pretrained=True).eval()
    print("Applying pruning...")
    pruned_model = apply_pruning(fresh_model, amount=float(amount), method=method)
    print("Running pruned model...")
    results_pruned, latency_pruned = run_inference(pruned_model, img)
    print(f"Pruned model done. Latency: {latency_pruned:.2f}ms")

    # Model sizes
    print("Saving models...")
    torch.save(fp32_model.state_dict(), "fp32_model.pth")
    
    # Make pruning permanent by removing the mask (this shows the actual reduced size)
    for module in pruned_model.modules():
        if isinstance(module, nn.Conv2d) and hasattr(module, 'weight_orig'):
            prune.remove(module, 'weight')
    
    torch.save(pruned_model.state_dict(), "pruned_model.pth")
    size_orig = os.path.getsize("fp32_model.pth") / 1e6
    size_pruned = os.path.getsize("pruned_model.pth") / 1e6
    print(f"Model sizes - Original: {size_orig:.2f}MB, Pruned: {size_pruned:.2f}MB")

    # Comparison metrics - as DataFrame for Gradio
    print("Creating metrics dataframe...")
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
    
    # Top-5 predictions chart data - as DataFrame for BarPlot
    print("Preparing chart data...")
    chart_df = pd.DataFrame({
        "Class": [results_orig[i][0] for i in range(5)],
        "Original": [results_orig[i][1] for i in range(5)],
        "Pruned": [results_pruned[i][1] for i in range(5)]
    })

    print("=== RUN PRUNED COMPLETE ===")
    return metrics_df, chart_df


def run_quantized(img, q_type):
    print("\n=== RUN QUANTIZED CALLED ===")
    print(f"Image type: {type(img)}, Q-type: {q_type}")
    
    if img is None:
        print("ERROR: Image is None")
        return {"Metric": ["Error"], "Original Model": ["No image uploaded"], "Quantized Model": [""]}, {"Class": [], "Original": [], "Quantized": []}
    
    # Run original model
    print("Running original model...")
    results_orig, latency_orig = run_inference(fp32_model, img)
    print(f"Original model done. Latency: {latency_orig:.2f}ms")
    
    # Run quantized model
    fresh_model = timm.create_model("resnet50", pretrained=True).eval()
    quant_model = apply_quantization(fresh_model, q_type)
    results_quant, latency_quant = run_inference(quant_model, img)

    # Model sizes
    torch.save(fp32_model.state_dict(), "fp32_model.pth")
    torch.save(quant_model.state_dict(), "quantized_model.pth")
    size_orig = os.path.getsize("fp32_model.pth") / 1e6
    size_quant = os.path.getsize("quantized_model.pth") / 1e6

    # Comparison metrics - as DataFrame for Gradio
    print("Creating metrics dataframe...")
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
    
    # Top-5 predictions chart data - as DataFrame for BarPlot
    print("Preparing chart data...")
    chart_df = pd.DataFrame({
        "Class": [results_orig[i][0] for i in range(5)],
        "Original": [results_orig[i][1] for i in range(5)],
        "Quantized": [results_quant[i][1] for i in range(5)]
    })

    print("=== RUN QUANTIZED COMPLETE ===")
    return metrics_df, chart_df


# ---------------------------------------------
# GRADIO UI
# ---------------------------------------------
# Example images
examples = [
    ["examples/cat.jpg"],
    ["examples/dog.jpg"],
    ["examples/bird.jpg"],
    ["examples/car.jpg"],
    ["examples/elephant.jpg"]
]

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 ResNet50 Optimization — Select Options to Compare")
    
    with gr.Tabs():

        # ---- PRUNING TAB ----
        with gr.Tab("Pruning"):
            with gr.Row():
                with gr.Column():
                    img_p = gr.Image(label="Upload Image")

                    method_p = gr.Dropdown(
                        ["unstructured", "structured"],
                        value="structured",
                        label="Pruning Method"
                    )

                    amount_p = gr.Slider(
                        minimum=0.1, maximum=0.9, step=0.1, value=0.4,
                        label="Pruning Amount"
                    )

                    btn_p = gr.Button("Run Pruned Model")
                    
                    gr.Examples(examples=examples, inputs=img_p)
                
                with gr.Column():
                    metrics_p = gr.Dataframe(label="📊 Comparison Metrics", headers=["Metric", "Original Model", "Pruned Model"])
                    chart_p = gr.BarPlot(
                        label="🎯 Top-5 Predictions Comparison",
                        x="Class",
                        y_title="Confidence",
                        height=400
                    )

            btn_p.click(fn=run_pruned, inputs=[img_p, method_p, amount_p], outputs=[metrics_p, chart_p])


        # ---- QUANTIZATION TAB ----
        with gr.Tab("Quantization"):
            with gr.Row():
                with gr.Column():
                    img_q = gr.Image(label="Upload Image")

                    q_type = gr.Dropdown(
                        ["dynamic", "weight_only", "fp16"],
                        value="dynamic",
                        label="Quantization Type"
                    )

                    btn_q = gr.Button("Run Quantized Model")
                    
                    gr.Examples(examples=examples, inputs=img_q)
                
                with gr.Column():
                    metrics_q = gr.Dataframe(label="📊 Comparison Metrics", headers=["Metric", "Original Model", "Quantized Model"])
                    chart_q = gr.BarPlot(
                        label="🎯 Top-5 Predictions Comparison",
                        x="Class",
                        y_title="Confidence",
                        height=400
                    )

            btn_q.click(fn=run_quantized, inputs=[img_q, q_type], outputs=[metrics_q, chart_q])


demo.launch()
