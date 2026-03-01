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


def center_rows(x):
    return x - x.mean(dim=1, keepdim=True)


def compute_metrics(pred, target):
    sample_mean_raw = F.cosine_similarity(pred, target, dim=1).mean().item()
    dim_mean_raw = F.cosine_similarity(pred.T, target.T, dim=1).mean().item()

    pred_c = center_rows(pred)
    target_c = center_rows(target)
    sample_mean_centered = F.cosine_similarity(pred_c, target_c, dim=1).mean().item()
    dim_mean_centered = F.cosine_similarity(pred_c.T, target_c.T, dim=1).mean().item()

    pred_var_mean = pred.var(dim=0, unbiased=False).mean().item()
    pred_var_min = pred.var(dim=0, unbiased=False).min().item()
    pred_var_max = pred.var(dim=0, unbiased=False).max().item()

    return {
        "test_cosine_sample_mean_raw": sample_mean_raw,
        "test_cosine_dimension_mean_raw": dim_mean_raw,
        "test_cosine_sample_mean_centered": sample_mean_centered,
        "test_cosine_dimension_mean_centered": dim_mean_centered,
        "pred_var_mean": pred_var_mean,
        "pred_var_min": pred_var_min,
        "pred_var_max": pred_var_max,
    }


def evaluate_probe(probe, test_x, test_t, device):
    probe.eval()
    with torch.no_grad():
        test_pred = probe(test_x.to(device)).cpu()
    return test_pred, compute_metrics(test_pred, test_t)


def train_probe(train_x, train_t, test_x, test_t, out_dim, lr, epochs, batch_size, device):
    probe = torch.nn.Linear(train_x.shape[1], out_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    _, untrained_metrics = evaluate_probe(probe, test_x, test_t, device)

    train_ds = TensorDataset(train_x, train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        probe.train()
        for x_batch, t_batch in train_loader:
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)

            pred = probe(x_batch)
            # Use centered cosine to avoid inflated scores from shared global bias.
            loss = 1.0 - F.cosine_similarity(center_rows(pred), center_rows(t_batch), dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_pred, trained_metrics = evaluate_probe(probe, test_x, test_t, device)
    return probe, test_pred, untrained_metrics, trained_metrics


def print_dataset_audit(train_backbone, test_backbone, train_idx, test_idx, train_backbone_sub, train_clip_sub):
    print("[Audit] Data separation checks")
    print(f"  train dataset length: {len(train_backbone)}")
    print(f"  test dataset length : {len(test_backbone)}")
    print(f"  train subset size   : {len(train_idx)}")
    print(f"  test subset size    : {len(test_idx)}")
    print(f"  train dataset split flag: {getattr(train_backbone, 'train', 'NA')}")
    print(f"  test dataset split flag : {getattr(test_backbone, 'train', 'NA')}")

    assert getattr(train_backbone, "train", True) is True, "Train dataset is not in train split"
    assert getattr(test_backbone, "train", False) is False, "Test dataset is not in test split"
    assert train_backbone_sub.indices == train_clip_sub.indices, "Backbone/CLIP train indices misaligned"
    print("  train backbone/CLIP index alignment: OK")


def check_target_alignment(name, image_features, text_features, targets):
    recon = image_features @ text_features.T
    max_abs = (recon - targets).abs().max().item()
    print(f"[Audit] Target matrix check ({name}): max |recomputed - stored| = {max_abs:.8f}")


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

    print_dataset_audit(train_backbone, test_backbone, train_idx, test_idx, train_backbone_sub, train_clip_sub)

    train_feats = {}
    test_feats = {}
    for layer in ("l1", "l2", "l3", "l4"):
        train_feats[layer], _ = extract_hierarchical_features(
            train_backbone_sub,
            backbone,
            layer=layer,
            device=device,
            batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
        )
        test_feats[layer], _ = extract_hierarchical_features(
            test_backbone_sub,
            backbone,
            layer=layer,
            device=device,
            batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
        )

    train_targets_low, low_concepts, train_img_low, text_low = build_clip_targets(
        train_clip_sub,
        args.targets_low,
        clip_model,
        device=device,
        batch_size=args.clip_batch_size,
        num_workers=args.num_workers,
        return_parts=True,
    )
    test_targets_low, _, test_img_low, _ = build_clip_targets(
        test_clip_sub,
        args.targets_low,
        clip_model,
        device=device,
        batch_size=args.clip_batch_size,
        num_workers=args.num_workers,
        return_parts=True,
    )

    train_targets_high, high_concepts, train_img_high, text_high = build_clip_targets(
        train_clip_sub,
        args.targets_high,
        clip_model,
        device=device,
        batch_size=args.clip_batch_size,
        num_workers=args.num_workers,
        return_parts=True,
    )
    test_targets_high, _, test_img_high, _ = build_clip_targets(
        test_clip_sub,
        args.targets_high,
        clip_model,
        device=device,
        batch_size=args.clip_batch_size,
        num_workers=args.num_workers,
        return_parts=True,
    )

    check_target_alignment("train-low", train_img_low, text_low, train_targets_low)
    check_target_alignment("test-low", test_img_low, text_low, test_targets_low)
    check_target_alignment("train-high", train_img_high, text_high, train_targets_high)
    check_target_alignment("test-high", test_img_high, text_high, test_targets_high)

    runs = []
    for layer in ("l1", "l2", "l3", "l4"):
        layer_name = f"layer{layer[-1]}"
        runs.append((f"{layer.upper()}_to_low", layer_name, layer, train_targets_low, test_targets_low, len(low_concepts), "low"))
        runs.append((f"{layer.upper()}_to_high", layer_name, layer, train_targets_high, test_targets_high, len(high_concepts), "high"))

    rows = []
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    for run_name, layer_name, layer_key, tr_t, te_t, concept_count, target_set in runs:
        tr_x = train_feats[layer_key]
        te_x = test_feats[layer_key]

        probe, test_pred, untrained_metrics, trained_metrics = train_probe(
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

        row = {
            "run_name": run_name,
            "feature_layer": layer_name,
            "target_set": target_set,
            "train_epochs": epochs,
            "seed": args.seed,
            "num_concepts": concept_count,
            "timestamp": timestamp,
        }
        row.update({f"untrained_{k}": v for k, v in untrained_metrics.items()})
        row.update({f"trained_{k}": v for k, v in trained_metrics.items()})
        rows.append(row)

        print(
            f"{run_name}: "
            f"untrained_centered={untrained_metrics['test_cosine_sample_mean_centered']:.4f}, "
            f"trained_centered={trained_metrics['test_cosine_sample_mean_centered']:.4f}, "
            f"trained_var_mean={trained_metrics['pred_var_mean']:.6f}"
        )

    fieldnames = [
        "run_name",
        "feature_layer",
        "target_set",
        "train_epochs",
        "seed",
        "num_concepts",
        "timestamp",
        "untrained_test_cosine_sample_mean_raw",
        "untrained_test_cosine_dimension_mean_raw",
        "untrained_test_cosine_sample_mean_centered",
        "untrained_test_cosine_dimension_mean_centered",
        "untrained_pred_var_mean",
        "untrained_pred_var_min",
        "untrained_pred_var_max",
        "trained_test_cosine_sample_mean_raw",
        "trained_test_cosine_dimension_mean_raw",
        "trained_test_cosine_sample_mean_centered",
        "trained_test_cosine_dimension_mean_centered",
        "trained_pred_var_mean",
        "trained_pred_var_min",
        "trained_pred_var_max",
    ]

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results -> {args.output_csv}")


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
