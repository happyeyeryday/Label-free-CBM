import argparse
import csv
import os
import re
import statistics
import sys
from collections import Counter


LAYER_FILES = {
    "l1": "cifar10_l1.txt",
    "l2": "cifar10_l2.txt",
    "l3": "cifar10_l3.txt",
    "l4": "cifar10_l4.txt",
}


def read_lines(path):
    with open(path, "r") as f:
        return [line.rstrip("\n") for line in f.readlines()]


def normalize(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def layer_index(layer_name):
    m = re.search(r"(\d)", layer_name.lower())
    if not m:
        return None
    return int(m.group(1))


def parse_levels_csv(value):
    valid = {"l1", "l2", "l3", "l4"}
    levels = [x.strip().lower() for x in value.split(",") if x.strip()]
    if not levels:
        raise ValueError("--levels must include at least one layer")
    invalid = [x for x in levels if x not in valid]
    if invalid:
        raise ValueError(f"--levels contains invalid layers: {invalid}")
    deduped = []
    seen = set()
    for level in levels:
        if level not in seen:
            seen.add(level)
            deduped.append(level)
    return deduped


def check_concept_files(concept_dir, active_levels):
    errors = []
    warnings = []
    per_layer = {}

    for key in active_levels:
        fname = LAYER_FILES[key]
        path = os.path.join(concept_dir, fname)
        if not os.path.exists(path):
            errors.append(f"Missing file: {path}")
            continue

        raw = read_lines(path)
        norm = [normalize(x) for x in raw if normalize(x)]

        if not raw:
            errors.append(f"{fname}: file is empty")
            continue

        if len(norm) == 0:
            errors.append(f"{fname}: no valid concepts after trimming")
            continue

        non_lower = [x for x in raw if x.strip() and x.strip() != x.strip().lower()]
        if non_lower:
            errors.append(f"{fname}: contains non-lowercase items (example: {non_lower[0]!r})")

        blanks = sum(1 for x in raw if not x.strip())
        if blanks > 0:
            warnings.append(f"{fname}: contains {blanks} blank lines")

        dup_counter = Counter(norm)
        dups = [k for k, v in dup_counter.items() if v > 1]
        if dups:
            errors.append(f"{fname}: duplicates found (example: {dups[0]!r})")

        per_layer[key] = norm

    if len(per_layer) == len(active_levels):
        all_seen = {}
        for layer, items in per_layer.items():
            for c in items:
                all_seen.setdefault(c, []).append(layer)

        overlaps = {k: v for k, v in all_seen.items() if len(v) > 1}
        if overlaps:
            sample_k = next(iter(overlaps.keys()))
            errors.append(
                f"Cross-layer duplicates found (example: {sample_k!r} in {overlaps[sample_k]})"
            )

        counts = {layer: len(items) for layer, items in per_layer.items()}
        min_count = min(counts.values())
        max_count = max(counts.values())
        if min_count < 10:
            warnings.append(f"Some layers are too small (<10 concepts). counts={counts}")
        if max_count > 4 * max(1, min_count):
            warnings.append(f"Layer size is highly imbalanced. counts={counts}")

    return errors, warnings, per_layer


def parse_results_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def analyze_layering_effect(rows):
    notes = []
    warns = []

    if not rows:
        warns.append("Results CSV is empty")
        return notes, warns

    required_cols = {
        "run_name",
        "trained_test_cosine_sample_mean_centered",
        "untrained_test_cosine_sample_mean_centered",
        "trained_pred_var_mean",
    }
    missing = [c for c in required_cols if c not in rows[0]]
    if missing:
        warns.append(f"Missing required columns for audit: {missing}")
        return notes, warns

    parsed = []
    for r in rows:
        run = r["run_name"]
        m = re.match(r"L(\d+)_to_([a-zA-Z0-9_]+)", run)
        if not m:
            continue
        layer = int(m.group(1))
        target = m.group(2).lower()

        try:
            tr = float(r["trained_test_cosine_sample_mean_centered"])
            un = float(r["untrained_test_cosine_sample_mean_centered"])
            var = float(r["trained_pred_var_mean"])
        except ValueError:
            continue

        parsed.append({"run": run, "layer": layer, "target": target, "trained": tr, "untrained": un, "var": var})

    if not parsed:
        warns.append("Could not parse run_name pattern like L1_to_xxx")
        return notes, warns

    # 1) Baseline sanity: trained should generally beat untrained
    deltas = [p["trained"] - p["untrained"] for p in parsed]
    notes.append(f"Mean(train-centered - untrained-centered) = {statistics.mean(deltas):.4f}")
    if statistics.mean(deltas) < 0.05:
        warns.append("Trained-vs-untrained gap is small (<0.05). Possible weak supervision or metric mismatch.")

    # 2) Variance collapse checks
    low_var_runs = [p for p in parsed if p["var"] < 1e-4]
    if low_var_runs:
        warns.append("Variance collapse suspected: " + ", ".join(x["run"] for x in low_var_runs))

    # 3) Target-wise ranking and trend
    targets = sorted(set(p["target"] for p in parsed))
    for target in targets:
        group = sorted([p for p in parsed if p["target"] == target], key=lambda x: x["layer"])
        if len(group) < 2:
            continue

        layers = [g["layer"] for g in group]
        scores = [g["trained"] for g in group]

        best = max(group, key=lambda x: x["trained"])
        worst = min(group, key=lambda x: x["trained"])
        notes.append(
            f"Target={target}: best={best['run']}({best['trained']:.4f}), "
            f"worst={worst['run']}({worst['trained']:.4f})"
        )

        # Simple monotonic tendency: compare first and last.
        gap = scores[-1] - scores[0]
        notes.append(f"Target={target}: layer{layers[-1]} - layer{layers[0]} gap = {gap:.4f}")

        if target in {"high", "l4", "global", "semantic"} and gap < 0.05:
            warns.append(
                f"Expected deep-layer advantage on high-level target '{target}' not clear (gap={gap:.4f})."
            )
        if target in {"low", "l1", "atomic", "shallow"} and gap > -0.02:
            warns.append(
                f"Expected shallow-layer advantage on low-level target '{target}' not clear (gap={gap:.4f})."
            )

    return notes, warns


def main(args):
    active_levels = parse_levels_csv(args.levels)
    errors, warnings, per_layer = check_concept_files(args.concept_dir, active_levels)

    print("=== Concept File Validation ===")
    for layer in active_levels:
        if layer in per_layer:
            print(f"{layer}: {len(per_layer[layer])} concepts")

    if errors:
        print("\n[ERRORS]")
        for e in errors:
            print(f"- {e}")
    else:
        print("\nNo hard errors found in concept files.")

    if warnings:
        print("\n[WARNINGS]")
        for w in warnings:
            print(f"- {w}")

    if args.results_csv:
        print("\n=== Layering Effect Audit ===")
        rows = parse_results_csv(args.results_csv)
        notes, audit_warns = analyze_layering_effect(rows)
        for n in notes:
            print(f"- {n}")
        if audit_warns:
            print("\n[AUDIT WARNINGS]")
            for w in audit_warns:
                print(f"- {w}")
        else:
            print("No obvious red flags from results CSV audit.")

    if errors:
        sys.exit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate hierarchical concept files and optionally audit results")
    parser.add_argument("--concept_dir", type=str, default="data/concept_sets")
    parser.add_argument(
        "--levels",
        type=str,
        default="l2,l3,l4",
        help="Comma-separated concept levels to validate (default: l2,l3,l4)",
    )
    parser.add_argument("--results_csv", type=str, default=None)
    main(parser.parse_args())
