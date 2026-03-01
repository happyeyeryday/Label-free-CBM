import argparse
import json
import os
import sys


def read_concepts(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def write_concepts(path, items):
    with open(path, "w") as f:
        f.write("\n".join(items))


def build_prompt(low, high):
    payload = {"low": low, "high": high}
    return (
        "You are refining hierarchical visual concepts for CIFAR-10. "
        "Return JSON only with keys low and high, each a list of strings. "
        "low should emphasize color/texture/local visual traits; high should emphasize semantic/object-level concepts. "
        "Do not drop concepts; every concept must appear exactly once across both lists.\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def main(args):
    try:
        from openai import OpenAI
    except ImportError:
        print("Missing dependency: openai. Install with `pip install openai`.", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    low = read_concepts(args.in_low)
    high = read_concepts(args.in_high)
    original = set(low + high)

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    resp = client.chat.completions.create(
        model=args.model,
        temperature=args.temperature,
        messages=[
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": build_prompt(low, high)},
        ],
    )

    content = resp.choices[0].message.content.strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        print("Model output is not valid JSON:", exc, file=sys.stderr)
        print(content, file=sys.stderr)
        sys.exit(1)

    new_low = parsed.get("low", [])
    new_high = parsed.get("high", [])

    if not isinstance(new_low, list) or not isinstance(new_high, list):
        print("JSON must contain list fields 'low' and 'high'.", file=sys.stderr)
        sys.exit(1)

    merged = set(new_low + new_high)
    if merged != original:
        missing = sorted(original - merged)
        extra = sorted(merged - original)
        print("Refined sets do not match original concept universe.", file=sys.stderr)
        print("Missing:", missing, file=sys.stderr)
        print("Extra:", extra, file=sys.stderr)
        sys.exit(1)

    if len(new_low) == 0 or len(new_high) == 0:
        print("Refined result has an empty group.", file=sys.stderr)
        sys.exit(1)

    write_concepts(args.out_low, new_low)
    write_concepts(args.out_high, new_high)

    print(f"Refined low concepts: {len(new_low)} -> {args.out_low}")
    print(f"Refined high concepts: {len(new_high)} -> {args.out_high}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional Kimi/OpenAI-compatible refinement for layered concepts")
    parser.add_argument("--in_low", type=str, default="data/concept_sets/cifar10_layer1.txt")
    parser.add_argument("--in_high", type=str, default="data/concept_sets/cifar10_layer4.txt")
    parser.add_argument("--out_low", type=str, default="data/concept_sets/cifar10_layer1.txt")
    parser.add_argument("--out_high", type=str, default="data/concept_sets/cifar10_layer4.txt")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    main(parser.parse_args())
