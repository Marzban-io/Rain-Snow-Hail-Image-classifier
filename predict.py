#!/usr/bin/env python3
"""
Command-line inference for the rain/snow/hail classifier.

Usage
-----
    python predict.py --image photo.jpg
    python predict.py --image photo.jpg --checkpoint models/rain_snow_hail_classifier.pth
    python predict.py --image folder_of_photos/ --topk 2

The checkpoint is the `.pth` file produced by
`notebooks/rain_snow_hail_classifier.ipynb`; it carries the class names and the
preprocessing settings alongside the weights, so nothing needs to be hardcoded
here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_model(num_classes: int) -> nn.Module:
    """Recreate the architecture used at training time (weights loaded separately)."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    return model


def load_classifier(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(num_classes=len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize(int(checkpoint["img_size"] * 1.14)),
        transforms.CenterCrop(checkpoint["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint["imagenet_mean"], checkpoint["imagenet_std"]),
    ])
    return model, preprocess, checkpoint["class_names"]


@torch.no_grad()
def predict(model, preprocess, class_names, image_path: Path, device: torch.device):
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    probabilities = torch.softmax(model(tensor), dim=1)[0]
    ranking = probabilities.argsort(descending=True)
    return [(class_names[i], probabilities[i].item()) for i in ranking]


def collect_images(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    return [path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a photo as rain, snow, or hail.")
    parser.add_argument("--image", required=True, type=Path,
                        help="Path to an image file, or a directory of images.")
    parser.add_argument("--checkpoint", default=Path("rain_snow_hail_classifier.pth"), type=Path,
                        help="Path to the .pth checkpoint (default: ./rain_snow_hail_classifier.pth).")
    parser.add_argument("--topk", default=3, type=int, help="How many classes to report per image.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table.")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train a model first with notebooks/rain_snow_hail_classifier.ipynb, "
            "or point --checkpoint at an existing .pth file."
        )

    images = collect_images(args.image)
    if not images:
        raise SystemExit(f"No images found at {args.image}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, class_names = load_classifier(args.checkpoint, device)

    results = {}
    for image_path in images:
        ranking = predict(model, preprocess, class_names, image_path, device)[: args.topk]
        results[str(image_path)] = {name: round(prob, 4) for name, prob in ranking}

        if not args.json:
            top_name, top_prob = ranking[0]
            print(f"\n{image_path}")
            print(f"  -> {top_name}  ({top_prob:.1%} confidence)")
            for name, prob in ranking:
                bar = "#" * int(round(prob * 30))
                print(f"     {name:<6} {prob:6.1%} {bar}")

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
