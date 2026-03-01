import argparse
import csv
import datetime
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import clip
from models.hierarchical_resnet import HierarchicalResNet
from validation.concept_utils import (
    auto_device,
    build_clip_targets,
    extract_hierarchical_features,
    get_cifar10_datasets,
    make_subset_indices,
    subset_dataset,
)


def train_probe(train_x, train_t, test_x, test_t, out_dim, lr, epochs, batch_size, device):
    probe = torch.nn.Linear(train_x.shape[1], out_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    train_ds = TensorDataset(train_x, train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        probe.train()
        for x_batch, t_batch in train_loader:
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)

            pred = probe(x_batch)
            loss = 1.0 - F.cosine_similarity(pred, t_batch, dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        test_pred = probe(test_x.to(device)).cpu()

    sample_mean = F.cosine_similarity(test_pred, test_t, dim=1).mean().item()
    dim_mean = F.cosine_similarity(test_pred.T, test_t.T, dim=1).mean().item()

    return probe, sample_mean, dim_mean


def main(args):
    device = auto_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.torch_home, exist_ok=True)
    os.makedirs(args.clip_download_root, exist_ok=True)
    os.environ["TORCH_HOME"] = os.path.abspath(args.torch_home)

    backbone = HierarchicalResNet(device=device)
    clip_model, clip_preprocess = clip.load(
        args.clip_name,
        device=device,
        download_root=os.path.abspath(args.clip_download_root),
    )

    backbone_preprocess = clip_preprocess if args.backbone_preprocess == "clip" else None
    if backbone_preprocess is None:
        from torchvision import models

        weights = models.ResNet50_Weights.IMAGENET1K_V1
        backbone_preprocess = weights.transforms()

    train_backbone, test_backbone, train_clip, test_clip = get_cifar10_datasets(
        backbone_preprocess,
        clip_preprocess,
        data_root=args.data_root,
    )

    train_n = len(train_backbone)
    test_n = len(test_backbone)
    if args.full:
        train_subset = train_n
        test_subset = test_n
        epochs = 50
    else:
        train_subset = min(args.train_subset, train_n)
        test_subset = min(args.test_subset, test_n)
        epochs = args.epochs

    train_idx = make_subset_indices(train_n, train_subset, args.seed)
    test_idx = make_subset_indices(test_n, test_subset, args.seed + 1)

    train_backbone_sub = subset_dataset(train_backbone, train_idx)
    test_backbone_sub = subset_dataset(test_backbone, test_idx)
    train_clip_sub = subset_dataset(train_clip, train_idx)
    test_clip_sub = subset_dataset(test_clip, test_idx)

    train_l1, _ = extract_hierarchical_features(
        train_backbone_sub,
        backbone,
        layer="l1",
        device=device,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
    )
    train_l4, _ = extract_hierarchical_features(
        train_backbone_sub,
        backbone,
        layer="l4",
        device=device,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
    )
    test_l1, _ = extract_hierarchical_features(
        test_backbone_sub,
        backbone,
        layer="l1",
        device=device,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
    )
    test_l4, _ = extract_hierarchical_features(
        test_backbone_sub,
        backbone,
        layer="l4",
        device=device,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
    )

    train_targets_low, low_concepts = build_clip_targets(
        train_clip_sub,
        args.targets_low,
        clip_model,
        device=device,
        batch_size=args.clip_batch_size,
        num_workers=args.num_workers,
    )
    test_targets_low, _ = build_clip_targets(
        test_clip_sub,
        args.targets_low,
        clip_model,
        device=device,
        batch_size=args.clip_batch_size,
        num_workers=args.num_workers,
    )

    train_targets_high, high_concepts = build_clip_targets(
        train_clip_sub,
        args.targets_high,
        clip_model,
        device=device,
        batch_size=args.clip_batch_size,
        num_workers=args.num_workers,
    )
    test_targets_high, _ = build_clip_targets(
        test_clip_sub,
        args.targets_high,
        clip_model,
        device=device,
        batch_size=args.clip_batch_size,
        num_workers=args.num_workers,
    )

    runs = [
        ("L1_to_low", "layer1", train_l1, test_l1, train_targets_low, test_targets_low, len(low_concepts)),
        ("L4_to_low", "layer4", train_l4, test_l4, train_targets_low, test_targets_low, len(low_concepts)),
        ("L1_to_high", "layer1", train_l1, test_l1, train_targets_high, test_targets_high, len(high_concepts)),
        ("L4_to_high", "layer4", train_l4, test_l4, train_targets_high, test_targets_high, len(high_concepts)),
    ]

    rows = []
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    for run_name, layer_name, tr_x, te_x, tr_t, te_t, concept_count in runs:
        probe, sample_mean, dim_mean = train_probe(
            tr_x,
            tr_t,
            te_x,
            te_t,
            out_dim=tr_t.shape[1],
            lr=args.lr,
            epochs=epochs,
            batch_size=args.probe_batch_size,
            device=device,
        )

        ckpt_path = os.path.join(args.save_dir, f"{run_name}.pt")
        torch.save(
            {
                "run_name": run_name,
                "feature_layer": layer_name,
                "state_dict": probe.state_dict(),
                "in_dim": tr_x.shape[1],
                "out_dim": tr_t.shape[1],
                "seed": args.seed,
                "epochs": epochs,
            },
            ckpt_path,
        )

        target_set = "low" if "low" in run_name else "high"
        rows.append(
            {
                "run_name": run_name,
                "feature_layer": layer_name,
                "target_set": target_set,
                "train_epochs": epochs,
                "test_cosine_sample_mean": sample_mean,
                "test_cosine_dimension_mean": dim_mean,
                "seed": args.seed,
                "num_concepts": concept_count,
                "timestamp": timestamp,
            }
        )

    fieldnames = [
        "run_name",
        "feature_layer",
        "target_set",
        "train_epochs",
        "test_cosine_sample_mean",
        "test_cosine_dimension_mean",
        "seed",
        "num_concepts",
        "timestamp",
    ]

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results -> {args.output_csv}")
    for row in rows:
        print(
            row["run_name"],
            f"sample_mean={row['test_cosine_sample_mean']:.4f}",
            f"dimension_mean={row['test_cosine_dimension_mean']:.4f}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical concept aptitude validation")
    parser.add_argument("--clip_name", type=str, default="ViT-B/16")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--full", action="store_true", help="Run PRD full setting (50 epochs, full CIFAR-10)")
    parser.add_argument("--epochs", type=int, default=10, help="Quick-mode probe epochs")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe_batch_size", type=int, default=256)
    parser.add_argument("--clip_batch_size", type=int, default=256)
    parser.add_argument("--feature_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_subset", type=int, default=5000)
    parser.add_argument("--test_subset", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets_low", type=str, default="data/concept_sets/cifar10_layer1.txt")
    parser.add_argument("--targets_high", type=str, default="data/concept_sets/cifar10_layer4.txt")
    parser.add_argument("--backbone_preprocess", type=str, choices=["resnet", "clip"], default="resnet")
    parser.add_argument("--save_dir", type=str, default="checkpoints/validation")
    parser.add_argument("--output_csv", type=str, default="validation_results.csv")
    parser.add_argument("--data_root", type=str, default=".cache/cifar10")
    parser.add_argument("--torch_home", type=str, default=".cache/torch")
    parser.add_argument("--clip_download_root", type=str, default=".cache/clip")

    main(parser.parse_args())
