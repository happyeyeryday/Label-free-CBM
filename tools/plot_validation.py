import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def grouped_bar(df, y_col, title, out_path):
    pivot = (
        df.pivot(index="target_set", columns="feature_layer", values=y_col)
        .reindex(index=["low", "high"])
        .reindex(columns=["layer1", "layer4"])
    )

    ax = pivot.plot(kind="bar", figsize=(7, 5), rot=0)
    ax.set_xlabel("Target concept set")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(title)
    ax.legend(title="Feature layer")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    sample_path = os.path.join(args.out_dir, "validation_bar_sample_mean.png")
    dim_path = os.path.join(args.out_dir, "validation_bar_dimension_mean.png")

    grouped_bar(
        df,
        y_col="test_cosine_sample_mean",
        title="Hierarchical Concept Validation (Sample Mean)",
        out_path=sample_path,
    )
    grouped_bar(
        df,
        y_col="test_cosine_dimension_mean",
        title="Hierarchical Concept Validation (Dimension Mean)",
        out_path=dim_path,
    )

    print(f"Saved: {sample_path}")
    print(f"Saved: {dim_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot hierarchical validation results")
    parser.add_argument("--csv", type=str, default="validation_results.csv")
    parser.add_argument("--out_dir", type=str, default="figures")
    main(parser.parse_args())
