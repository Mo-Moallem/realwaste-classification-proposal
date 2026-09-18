"""Inspect RealWaste, create required visuals, and write a fixed split manifest.

Usage:
    python analysis.py --data-dir /path/to/realwaste-main/RealWaste
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


EXPECTED_COUNTS = {
    "Cardboard": 461,
    "Food Organics": 411,
    "Glass": 420,
    "Metal": 790,
    "Miscellaneous Trash": 495,
    "Paper": 500,
    "Plastic": 921,
    "Textile Trash": 318,
    "Vegetation": 436,
}
SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_images(data_dir: Path) -> pd.DataFrame:
    rows = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in VALID_EXTENSIONS:
                with Image.open(path) as image:
                    width, height = image.size
                    rgb = image.mode
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                rows.append(
                    {
                        "path": str(path.resolve()),
                        "relative_path": str(path.relative_to(data_dir)),
                        "label": class_dir.name,
                        "width": width,
                        "height": height,
                        "mode": rgb,
                        "sha256": digest,
                    }
                )
    if not rows:
        raise ValueError(f"No images found under {data_dir}")
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame, output_dir: Path) -> None:
    observed = df["label"].value_counts().sort_index().to_dict()
    print("Observed class counts:")
    print(pd.Series(observed).to_string())
    if observed != dict(sorted(EXPECTED_COUNTS.items())):
        print("\nWarning: observed counts differ from the UCI reference counts.")
    duplicates = df[df.duplicated("sha256", keep=False)].sort_values("sha256")
    print(f"\nExact duplicate files: {len(duplicates)}")
    if len(duplicates):
        duplicate_path = output_dir / "exact_duplicates.csv"
        duplicates.to_csv(duplicate_path, index=False)
        print(f"Review {duplicate_path} before model training.")
    print("\nImage dimensions:")
    print(df[["width", "height"]].value_counts().head(10).to_string())


def make_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df["label"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot.barh(ax=ax, color="#087E8B")
    ax.set_title("RealWaste class distribution")
    ax.set_xlabel("Number of images")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(ax.patches, counts.values):
        ax.text(value + 8, bar.get_y() + bar.get_height() / 2, str(value), va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=180)
    plt.close(fig)


def make_sample_grid(df: pd.DataFrame, output_dir: Path) -> None:
    samples = df.sort_values("relative_path").groupby("label", sort=True).head(1)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, (_, row) in zip(axes.flat, samples.iterrows()):
        with Image.open(row["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            ax.imshow(image)
        ax.set_title(row["label"], fontsize=10)
        ax.axis("off")
    fig.suptitle("One inspected RealWaste image per class", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "sample_grid.png", dpi=180)
    plt.close(fig)


def assign_splits(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    train, holdout = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label"],
    )
    validation, test = train_test_split(
        holdout,
        test_size=0.50,
        random_state=SEED,
        stratify=holdout["label"],
    )
    train = train.assign(split="train")
    validation = validation.assign(split="validation")
    test = test.assign(split="test")
    manifest = pd.concat([train, validation, test], ignore_index=True)
    manifest = manifest.sort_values(["split", "label", "relative_path"])
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    print("\nSplit counts:")
    print(pd.crosstab(manifest["label"], manifest["split"]).to_string())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = collect_images(args.data_dir)
    df.to_csv(args.output_dir / "image_inventory.csv", index=False)
    validate(df, args.output_dir)
    make_distribution(df, args.output_dir)
    make_sample_grid(df, args.output_dir)
    assign_splits(df, args.output_dir)
    print(f"\nOutputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
