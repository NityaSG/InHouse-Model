"""Gender classification inference (EfficientNet-B0, 3 classes)."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

DEFAULT_CLASSES = ["men", "women", "other"]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class GenderClassifier(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def _base_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def _tta_transforms() -> list[transforms.Compose]:
    base = _base_transform()
    flip = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    crop_256 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    crop_240 = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    crop_flip = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    return [base, flip, crop_256, crop_240, crop_flip]


class GenderInference:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Gender checkpoint not found: {checkpoint_path}")

        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {"classes": DEFAULT_CLASSES})
        self.classes: list[str] = list(config.get("classes", DEFAULT_CLASSES))

        self.model = GenderClassifier(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self._base = _base_transform()
        self._tta = _tta_transforms()

    @torch.inference_mode()
    def predict(self, image: Image.Image, tta: bool = False) -> dict:
        return self.predict_batch([image], tta=tta)[0]

    @torch.inference_mode()
    def predict_batch(self, images: list[Image.Image], tta: bool = False) -> list[dict]:
        if not images:
            return []

        rgb_images = [img.convert("RGB") for img in images]
        transforms_list = self._tta if tta else [self._base]

        probs_sum = np.zeros((len(rgb_images), len(self.classes)), dtype=np.float32)
        for t in transforms_list:
            batch = torch.stack([t(img) for img in rgb_images]).to(self.device)
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_sum += probs
        probs_batch = probs_sum / len(transforms_list)

        results = []
        for probs in probs_batch:
            top_idx = int(np.argmax(probs))
            results.append({
                "prediction": self.classes[top_idx],
                "confidence": float(probs[top_idx]),
                "probabilities": {
                    cls: float(p) for cls, p in zip(self.classes, probs)
                },
                "tta": tta,
            })
        return results
