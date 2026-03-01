import argparse
import json
import os
import re
from statistics import median


COLOR_WORDS = {
    "red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black", "white",
    "gray", "grey", "gold", "silver", "cyan", "magenta", "beige",
}

LOW_LEVEL_WORDS = {
    "striped", "spotted", "speckled", "furry", "fuzzy", "smooth", "rough", "shiny", "glossy",
    "matte", "metallic", "wooden", "plastic", "fabric", "texture", "pattern", "edge", "outline",
    "pixel", "bright", "dark", "light", "shadow", "blurred", "blurry", "noisy", "grainy",
}

HIGH_LEVEL_WORDS = {
    "animal", "vehicle", "object", "person", "scene", "tool", "machine", "transport", "device",
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
    "wing", "wheel", "engine", "window", "door", "beak", "tail", "head", "body", "face",
}


def tokenize(text):
    return re.findall(r"[a-zA-Z]+", text.lower())


def score_concept(concept):
    tokens = tokenize(concept)
    low_score = 0
    high_score = 0

    for token in tokens:
        if token in COLOR_WORDS:
            low_score += 3
        if token in LOW_LEVEL_WORDS:
            low_score += 2
        if token in HIGH_LEVEL_WORDS:
            high_score += 2

    if any(token.endswith("ish") for token in tokens):
        low_score += 1

    if len(tokens) >= 3:
        high_score += 1

    if "texture" in tokens or "pattern" in tokens:
        low_score += 2

    if "object" in tokens or "animal" in tokens or "vehicle" in tokens:
        high_score += 2

    delta = high_score - low_score
    return low_score, high_score, delta


def read_concepts(path):
    with open(path, "r") as f:
        concepts = [line.strip() for line in f.readlines()]
    return [c for c in concepts if c]


def write_list(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(items))


def main(args):
    concepts = read_concepts(args.input)
    if not concepts:
        raise ValueError("Input concept file is empty")

    scored = []
    for concept in concepts:
        low_score, high_score, delta = score_concept(concept)
        scored.append(
            {
                "concept": concept,
                "low_score": low_score,
                "high_score": high_score,
                "delta_high_minus_low": delta,
            }
        )

    deltas = [s["delta_high_minus_low"] for s in scored]
    split_point = median(deltas)

    low = [s["concept"] for s in scored if s["delta_high_minus_low"] <= split_point]
    high = [s["concept"] for s in scored if s["delta_high_minus_low"] > split_point]

    if not low or not high:
        scored_sorted = sorted(scored, key=lambda x: x["delta_high_minus_low"])
        half = max(1, len(scored_sorted) // 2)
        low = [s["concept"] for s in scored_sorted[:half]]
        high = [s["concept"] for s in scored_sorted[half:]]

    write_list(args.out_low, low)
    write_list(args.out_high, high)

    meta = {
        "input": args.input,
        "out_low": args.out_low,
        "out_high": args.out_high,
        "split_point": split_point,
        "num_total": len(scored),
        "num_low": len(low),
        "num_high": len(high),
        "scored": scored,
    }
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote low concepts: {len(low)} -> {args.out_low}")
    print(f"Wrote high concepts: {len(high)} -> {args.out_high}")
    print(f"Wrote metadata -> {args.meta_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split CIFAR10 concepts into low/high hierarchical sets")
    parser.add_argument("--input", type=str, default="data/concept_sets/cifar10_filtered.txt")
    parser.add_argument("--out_low", type=str, default="data/concept_sets/cifar10_layer1.txt")
    parser.add_argument("--out_high", type=str, default="data/concept_sets/cifar10_layer4.txt")
    parser.add_argument("--meta_out", type=str, default="data/concept_sets/layered_concepts_meta.json")
    main(parser.parse_args())
