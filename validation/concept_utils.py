import math
from typing import Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

import clip


def auto_device(device=None):
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_concepts(path: str) -> List[str]:
    with open(path, "r") as f:
        concepts = [line.strip() for line in f.readlines()]
    return [c for c in concepts if c]


def make_subset_indices(length: int, subset_size: int, seed: int) -> List[int]:
    if subset_size <= 0 or subset_size >= length:
        return list(range(length))
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(length, generator=generator)[:subset_size]
    return perm.tolist()


def get_cifar10_datasets(backbone_preprocess, clip_preprocess, data_root: str):
    train_backbone = datasets.CIFAR10(root=data_root, download=True, train=True, transform=backbone_preprocess)
    test_backbone = datasets.CIFAR10(root=data_root, download=True, train=False, transform=backbone_preprocess)
    train_clip = datasets.CIFAR10(root=data_root, download=True, train=True, transform=clip_preprocess)
    test_clip = datasets.CIFAR10(root=data_root, download=True, train=False, transform=clip_preprocess)
    return train_backbone, test_backbone, train_clip, test_clip


def encode_clip_text_features(concepts: Sequence[str], clip_model, device: str, batch_size: int = 512):
    tokens = clip.tokenize([str(c) for c in concepts])
    outputs = []
    with torch.no_grad():
        for i in range(math.ceil(tokens.shape[0] / batch_size)):
            curr = tokens[i * batch_size : (i + 1) * batch_size].to(device)
            feat = clip_model.encode_text(curr).float()
            outputs.append(feat)
    text_features = torch.cat(outputs, dim=0)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    return text_features


def encode_clip_image_features(dataset, clip_model, device: str, batch_size: int = 256, num_workers: int = 4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    outputs = []
    with torch.no_grad():
        for images, _ in loader:
            feats = clip_model.encode_image(images.to(device)).float()
            feats = feats / feats.norm(dim=1, keepdim=True)
            outputs.append(feats.cpu())
    return torch.cat(outputs, dim=0)


def build_clip_targets(clip_dataset, concept_file: str, clip_model, device: str, batch_size: int = 256, num_workers: int = 4):
    concepts = load_concepts(concept_file)
    text_features = encode_clip_text_features(concepts, clip_model, device)
    image_features = encode_clip_image_features(
        clip_dataset,
        clip_model,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    targets = image_features @ text_features.cpu().T
    return targets, concepts


def extract_hierarchical_features(dataset, model, layer: str, device: str, batch_size: int = 256, num_workers: int = 4):
    if layer not in {"l1", "l4"}:
        raise ValueError(f"Unsupported layer: {layer}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    features = []
    labels = []

    model.eval()
    with torch.no_grad():
        for images, y in loader:
            out = model(images.to(device))[layer]
            features.append(out.cpu())
            labels.append(y)

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def subset_dataset(dataset, indices: Iterable[int]):
    return Subset(dataset, list(indices))
