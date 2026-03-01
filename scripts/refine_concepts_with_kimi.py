import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List


def read_concepts(path: str) -> List[str]:
    with open(path, "r") as f:
        concepts = [line.strip() for line in f.readlines()]
    return [c for c in concepts if c]


def normalize_concepts(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        clean = re.sub(r"\s+", " ", item.strip().lower())
        if not clean:
            continue
        if clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def extract_json_object(text: str) -> Dict:
    text = text.strip()

    # Handle fenced JSON blocks.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    # Try direct parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find first JSON object span.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")

    return json.loads(text[start : end + 1])


def build_sort_prompt(concepts: List[str]) -> str:
    payload = {
        "task": "Sort CIFAR-10 concepts into 4 hierarchical levels",
        "levels": {
            "l1": "atomic/shallow primitives: pure colors, micro-textures, basic local patterns",
            "l2": "local parts: small object components with simple shapes",
            "l3": "complex components: larger structural parts requiring wider receptive fields",
            "l4": "global semantic: abstract categories, habitat/behavior, full-object semantics",
        },
        "constraints": [
            "Output strict JSON only",
            "Use keys: l1, l2, l3, l4",
            "Each value must be a list of lowercase strings",
            "Every input concept must appear exactly once across l1-l4",
            "No extra commentary",
        ],
        "concepts": concepts,
    }
    return json.dumps(payload, ensure_ascii=False)


def build_augment_prompt(classes: List[str], existing_l1: List[str], existing_l2: List[str]) -> str:
    payload = {
        "task": "Generate additional low-level visual primitives for CIFAR-10",
        "goal": "If shallow levels are weak, propose strong visual primitives for shallow probes",
        "classes": classes,
        "constraints": [
            "Output strict JSON only",
            "Use keys: l1_extra, l2_extra",
            "Generate 20-30 items total",
            "Lowercase strings only",
            "Focus on visual primitives and local parts only",
            "Avoid full object names and high-level semantics",
        ],
        "existing_l1": existing_l1,
        "existing_l2": existing_l2,
    }
    return json.dumps(payload, ensure_ascii=False)


def call_kimi_json(client, model: str, prompt: str, temperature: float = 0.0) -> Dict:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are a strict JSON generator. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content
    return extract_json_object(content)


def write_concepts(path: str, concepts: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(concepts))


def validate_partition(original: List[str], grouped: Dict[str, List[str]]) -> None:
    levels = ["l1", "l2", "l3", "l4"]
    for level in levels:
        if level not in grouped:
            print(f"WARNING: Missing key from model output: {level}", file=sys.stderr)
            grouped[level] = []

    merged = []
    for level in levels:
        merged.extend(grouped[level])

    original_norm = normalize_concepts(original)
    merged_norm = normalize_concepts(merged)

    original_set = set(original_norm)
    merged_set = set(merged_norm)

    if original_set != merged_set:
        missing = sorted(original_set - merged_set)
        extra = sorted(merged_set - original_set)
        print(
            "WARNING: Model partition differs from original concepts. "
            f"Missing={missing[:20]}, Extra={extra[:20]}",
            file=sys.stderr
        )


def main(args):
    try:
        from openai import OpenAI
    except ImportError:
        print("Missing dependency: openai. Install with `pip install openai`.", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        print("MOONSHOT_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    concepts = read_concepts(args.input)
    if not concepts:
        raise ValueError("Input concept file is empty")

    client = OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")

    print(f"Loaded {len(concepts)} concepts from {args.input}")
    print("[1/2] Requesting 4-level hierarchical sorting from Kimi...")
    sort_result = call_kimi_json(client, args.model, build_sort_prompt(concepts), temperature=args.temperature)
    time.sleep(1.0)

    grouped = {
        "l1": normalize_concepts(sort_result.get("l1", [])),
        "l2": normalize_concepts(sort_result.get("l2", [])),
        "l3": normalize_concepts(sort_result.get("l3", [])),
        "l4": normalize_concepts(sort_result.get("l4", [])),
    }

    validate_partition(concepts, grouped)

    classes = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    ]

    need_augmentation = len(grouped["l1"]) < args.min_l1 or len(grouped["l2"]) < args.min_l2
    if need_augmentation:
        print("[2/2] L1/L2 deemed sparse, requesting Kimi low-level augmentation...")
        aug_result = call_kimi_json(
            client,
            args.model,
            build_augment_prompt(classes, grouped["l1"], grouped["l2"]),
            temperature=args.temperature,
        )
        time.sleep(1.0)

        l1_extra = normalize_concepts(aug_result.get("l1_extra", []))
        l2_extra = normalize_concepts(aug_result.get("l2_extra", []))

        grouped["l1"] = normalize_concepts(grouped["l1"] + l1_extra)
        grouped["l2"] = normalize_concepts(grouped["l2"] + l2_extra)

        print(f"Augmented L1 by {len(l1_extra)} and L2 by {len(l2_extra)} concepts")
    else:
        print("[2/2] L1/L2 already sufficiently populated; skip augmentation")

    out_l1 = os.path.join(args.output_dir, "cifar10_l1.txt")
    out_l2 = os.path.join(args.output_dir, "cifar10_l2.txt")
    out_l3 = os.path.join(args.output_dir, "cifar10_l3.txt")
    out_l4 = os.path.join(args.output_dir, "cifar10_l4.txt")

    write_concepts(out_l1, grouped["l1"])
    write_concepts(out_l2, grouped["l2"])
    write_concepts(out_l3, grouped["l3"])
    write_concepts(out_l4, grouped["l4"])

    meta = {
        "input": args.input,
        "model": args.model,
        "base_url": "https://api.moonshot.cn/v1",
        "counts": {k: len(v) for k, v in grouped.items()},
        "min_l1": args.min_l1,
        "min_l2": args.min_l2,
    }
    meta_path = os.path.join(args.output_dir, "cifar10_hierarchy_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    print(f"  L1 -> {out_l1} ({len(grouped['l1'])})")
    print(f"  L2 -> {out_l2} ({len(grouped['l2'])})")
    print(f"  L3 -> {out_l3} ({len(grouped['l3'])})")
    print(f"  L4 -> {out_l4} ({len(grouped['l4'])})")
    print(f"  Meta -> {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine CIFAR-10 concepts into 4 hierarchical levels with Kimi")
    parser.add_argument("--input", type=str, default="data/concept_sets/cifar10_filtered.txt")
    parser.add_argument("--output_dir", type=str, default="data/concept_sets")
    parser.add_argument("--model", type=str, default="moonshot-v1-8k")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--min_l1", type=int, default=35, help="If L1 count below this, trigger augmentation")
    parser.add_argument("--min_l2", type=int, default=30, help="If L2 count below this, trigger augmentation")
    main(parser.parse_args())
