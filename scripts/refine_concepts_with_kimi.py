import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple


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

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    return json.loads(text[start : end + 1])


def call_llm_json(client, model: str, prompt: str, temperature: float) -> Dict:
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


def build_missing_prompt(missing: List[str]) -> str:
    payload = {
        "task": "Assign missing concepts to one of l1/l2/l3/l4",
        "constraints": [
            "Output strict JSON only",
            "Use key: assignments",
            "assignments should be a list of objects: {concept, level}",
            "level must be one of l1,l2,l3,l4",
            "Every concept must be assigned exactly once",
        ],
        "missing_concepts": missing,
    }
    return json.dumps(payload, ensure_ascii=False)


def build_rebalance_prompt(
    grouped: Dict[str, List[str]],
    min_l1: int,
    min_l2: int,
    min_l3: int,
    min_l4: int,
) -> str:
    payload = {
        "task": "Rebalance hierarchical concept groups while preserving partition",
        "constraints": [
            "Output strict JSON only",
            "Use keys: l1,l2,l3,l4",
            "Only use provided concepts; no new concepts",
            "Each concept appears exactly once across l1-l4",
            f"l1 count >= {min_l1}",
            f"l2 count >= {min_l2}",
            f"l3 count >= {min_l3}",
            f"l4 count >= {min_l4}",
        ],
        "current_groups": grouped,
    }
    return json.dumps(payload, ensure_ascii=False)


def build_augment_prompt(classes: List[str], existing_l1: List[str], existing_l2: List[str]) -> str:
    payload = {
        "task": "Generate additional low-level visual primitives for CIFAR-10",
        "goal": "Strengthen shallow probe targets with visual primitives and local parts",
        "classes": classes,
        "constraints": [
            "Output strict JSON only",
            "Use keys: l1_extra, l2_extra",
            "Generate 20-30 items total",
            "Lowercase strings only",
            "Focus on visual primitives/local parts only",
            "Avoid full object names and high-level semantics",
        ],
        "existing_l1": existing_l1,
        "existing_l2": existing_l2,
    }
    return json.dumps(payload, ensure_ascii=False)


def grouped_from_result(result: Dict) -> Dict[str, List[str]]:
    return {
        "l1": normalize_concepts(result.get("l1", [])),
        "l2": normalize_concepts(result.get("l2", [])),
        "l3": normalize_concepts(result.get("l3", [])),
        "l4": normalize_concepts(result.get("l4", [])),
    }


def make_disjoint(grouped: Dict[str, List[str]], original_set: set) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    seen = set()
    disjoint = {k: [] for k in ("l1", "l2", "l3", "l4")}
    extras = []

    for level in ("l1", "l2", "l3", "l4"):
        for concept in grouped.get(level, []):
            if concept not in original_set:
                extras.append(concept)
                continue
            if concept in seen:
                continue
            seen.add(concept)
            disjoint[level].append(concept)

    missing = sorted(original_set - seen)
    return disjoint, missing, sorted(set(extras))


def strict_validate_partition(original: List[str], grouped: Dict[str, List[str]]) -> None:
    original_norm = normalize_concepts(original)
    original_set = set(original_norm)

    merged = []
    for level in ("l1", "l2", "l3", "l4"):
        merged.extend(grouped.get(level, []))
    merged_norm = normalize_concepts(merged)
    merged_set = set(merged_norm)

    if len(merged_norm) != len(merged):
        raise ValueError("Duplicates detected across hierarchical groups")

    if merged_set != original_set:
        missing = sorted(original_set - merged_set)
        extra = sorted(merged_set - original_set)
        raise ValueError(f"Partition mismatch: missing={missing[:20]}, extra={extra[:20]}")


def counts_ok(grouped: Dict[str, List[str]], min_l1: int, min_l2: int, min_l3: int, min_l4: int) -> bool:
    return (
        len(grouped["l1"]) >= min_l1
        and len(grouped["l2"]) >= min_l2
        and len(grouped["l3"]) >= min_l3
        and len(grouped["l4"]) >= min_l4
    )


def write_concepts(path: str, concepts: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(concepts))


def main(args):
    try:
        from openai import OpenAI
    except ImportError:
        print("Missing dependency: openai. Install with `pip install openai`.", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(f"{args.api_key_env} is not set.", file=sys.stderr)
        sys.exit(1)

    concepts = read_concepts(args.input)
    if not concepts:
        raise ValueError("Input concept file is empty")

    concepts = normalize_concepts(concepts)
    original_set = set(concepts)
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    print(f"Loaded {len(concepts)} concepts from {args.input}")

    print("[1/4] Initial 4-level sorting...")
    sort_result = call_llm_json(client, args.model, build_sort_prompt(concepts), temperature=args.temperature)
    time.sleep(1.0)
    grouped = grouped_from_result(sort_result)

    grouped, missing, extras = make_disjoint(grouped, original_set)
    if extras:
        print(f"Dropped {len(extras)} non-original concepts from initial partition")

    if missing:
        print(f"[2/4] Recovering {len(missing)} missing concepts...")
        recover = call_llm_json(client, args.model, build_missing_prompt(missing), temperature=args.temperature)
        time.sleep(1.0)
        assignments = recover.get("assignments", [])
        assign_map = {}
        for item in assignments:
            concept = normalize_concepts([str(item.get("concept", ""))])
            if not concept:
                continue
            c = concept[0]
            level = str(item.get("level", "")).strip().lower()
            if c in missing and level in {"l1", "l2", "l3", "l4"}:
                assign_map[c] = level

        for c in missing:
            level = assign_map.get(c, "l4")
            grouped[level].append(c)

        grouped, missing2, _ = make_disjoint(grouped, original_set)
        if missing2:
            raise ValueError(f"Failed to recover all missing concepts: {missing2[:20]}")

    strict_validate_partition(concepts, grouped)

    if not counts_ok(grouped, args.min_l1, args.min_l2, args.min_l3, args.min_l4):
        print("[3/4] Rebalancing groups to satisfy minimum counts...")
        rebalance = call_llm_json(
            client,
            args.model,
            build_rebalance_prompt(grouped, args.min_l1, args.min_l2, args.min_l3, args.min_l4),
            temperature=args.temperature,
        )
        time.sleep(1.0)
        grouped = grouped_from_result(rebalance)
        grouped, missing3, extras3 = make_disjoint(grouped, original_set)
        if missing3 or extras3:
            raise ValueError(
                f"Rebalance invalid. missing={missing3[:20]}, extras={extras3[:20]}"
            )
        strict_validate_partition(concepts, grouped)

    if not counts_ok(grouped, args.min_l1, args.min_l2, args.min_l3, args.min_l4):
        raise ValueError(
            f"Could not satisfy min counts after rebalance. counts={{k: len(v) for k, v in grouped.items()}}"
        )

    classes = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    ]

    print("[4/4] Augmenting shallow levels (L1/L2) with extra primitives...")
    aug_result = call_llm_json(
        client,
        args.model,
        build_augment_prompt(classes, grouped["l1"], grouped["l2"]),
        temperature=args.temperature,
    )
    time.sleep(1.0)

    l1_extra = normalize_concepts(aug_result.get("l1_extra", []))
    l2_extra = normalize_concepts(aug_result.get("l2_extra", []))

    # Keep extras distinct from original concepts.
    l1_extra = [x for x in l1_extra if x not in original_set]
    l2_extra = [x for x in l2_extra if x not in original_set and x not in set(l1_extra)]

    grouped["l1"] = normalize_concepts(grouped["l1"] + l1_extra)
    grouped["l2"] = normalize_concepts(grouped["l2"] + l2_extra)

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
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "counts": {k: len(v) for k, v in grouped.items()},
        "core_partition_counts": {
            "l1": len([x for x in grouped["l1"] if x in original_set]),
            "l2": len([x for x in grouped["l2"] if x in original_set]),
            "l3": len([x for x in grouped["l3"] if x in original_set]),
            "l4": len([x for x in grouped["l4"] if x in original_set]),
        },
        "extra_counts": {"l1_extra": len(l1_extra), "l2_extra": len(l2_extra)},
        "min_constraints": {
            "min_l1": args.min_l1,
            "min_l2": args.min_l2,
            "min_l3": args.min_l3,
            "min_l4": args.min_l4,
        },
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
    parser = argparse.ArgumentParser(description="Refine CIFAR-10 concepts into 4 hierarchical levels with LLM")
    parser.add_argument("--input", type=str, default="data/concept_sets/cifar10_filtered.txt")
    parser.add_argument("--output_dir", type=str, default="data/concept_sets")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com/v1")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--min_l1", type=int, default=25)
    parser.add_argument("--min_l2", type=int, default=25)
    parser.add_argument("--min_l3", type=int, default=20)
    parser.add_argument("--min_l4", type=int, default=20)
    main(parser.parse_args())
