from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = (
    BASE_DIR
    / "app"
    / "processing"
    / "evaluation_results"
)

THRESHOLD_CSV = RESULTS_DIR / "threshold_report.csv"
IMAGE_RESULTS_CSV = RESULTS_DIR / "image_similarity_scores.csv"

OUTPUT_DIR = RESULTS_DIR / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_columns(df):
    df.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in df.columns
    ]
    return df


def find_column(df, *candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def plot_threshold_metrics():
    df = normalize_columns(pd.read_csv(THRESHOLD_CSV))

    threshold_col = find_column(df, "threshold")
    precision_col = find_column(df, "precision")
    recall_col = find_column(df, "recall")
    f1_col = find_column(df, "f1", "f1_score")

    if not all([threshold_col, precision_col, recall_col, f1_col]):
        raise ValueError(
            f"Required threshold/precision/recall/F1 columns not found. "
            f"Available columns: {list(df.columns)}"
        )

    for column in [precision_col, recall_col, f1_col]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

        # Handle metrics stored as percentages, e.g. 95 instead of 0.95.
        if df[column].max() > 1:
            df[column] = df[column] / 100

    df = df.sort_values(threshold_col)

    plt.figure(figsize=(10, 6))
    plt.plot(
        df[threshold_col],
        df[precision_col],
        label="Precision",
        marker=".",
    )
    plt.plot(
        df[threshold_col],
        df[recall_col],
        label="Recall",
        marker=".",
    )
    plt.plot(
        df[threshold_col],
        df[f1_col],
        label="F1 Score",
        marker=".",
    )

    plt.axvline(
        x=0.52,
        linestyle="--",
        label="Selected threshold = 0.52",
    )

    plt.xlabel("Cosine Similarity Threshold")
    plt.ylabel("Score")
    plt.title("SnapTrace: Retrieval Metrics vs Threshold")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "threshold_metrics.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_counts():
    df = normalize_columns(pd.read_csv(THRESHOLD_CSV))

    threshold_col = find_column(df, "threshold")
    tp_col = find_column(df, "tp", "true_positives")
    fp_col = find_column(df, "fp", "false_positives")
    fn_col = find_column(df, "fn", "false_negatives")
    tn_col = find_column(df, "tn", "true_negatives")

    if not all([threshold_col, tp_col, fp_col, fn_col, tn_col]):
        print(
            "Skipping confusion-count graph: TP, FP, FN, TN columns "
            "were not all found."
        )
        print("Available columns:", list(df.columns))
        return

    df[threshold_col] = pd.to_numeric(
        df[threshold_col], errors="coerce"
    )
    selected = df.loc[
        (df[threshold_col] - 0.52).abs() < 0.005
    ]

    if selected.empty:
        selected = df.iloc[
            [(df[threshold_col] - 0.52).abs().argmin()]
        ]

    row = selected.iloc[0]

    labels = [
        "Correct matches\n(TP)",
        "False matches\n(FP)",
        "Missed matches\n(FN)",
        "Correct rejections\n(TN)",
    ]
    values = [
        int(row[tp_col]),
        int(row[fp_col]),
        int(row[fn_col]),
        int(row[tn_col]),
    ]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, values)

    plt.bar_label(bars, padding=3, fmt="%d")
    plt.ylabel("Number of photographs")
    plt.title(
        f"SnapTrace: Retrieval Outcomes at Threshold "
        f"{row[threshold_col]:.2f}"
    )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / "retrieval_outcomes.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_similarity_distribution():
    if not IMAGE_RESULTS_CSV.exists():
        print(
            "Skipping similarity distribution: "
            f"{IMAGE_RESULTS_CSV.name} was not found."
        )
        return

    df = normalize_columns(pd.read_csv(IMAGE_RESULTS_CSV))

    score_col = find_column(
        df,
        "similarity",
        "similarity_score",
        "max_similarity",
        "score",
    )

    if score_col is None:
        print(
            "Skipping similarity distribution: score column not found."
        )
        print("Available columns:", list(df.columns))
        return

    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    scores = df[score_col].dropna()

    if scores.empty:
        print("No valid similarity scores found.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, edgecolor="black")
    plt.axvline(
        x=0.52,
        linestyle="--",
        label="Selected threshold = 0.52",
    )

    plt.xlabel("Maximum Cosine Similarity per Photograph")
    plt.ylabel("Number of photographs")
    plt.title("SnapTrace: Gallery Similarity Score Distribution")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "similarity_distribution.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    if not THRESHOLD_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find {THRESHOLD_CSV}. "
            "Run evaluate_threshold.py first."
        )

    plot_threshold_metrics()
    plot_confusion_counts()
    plot_similarity_distribution()

    print("\nGraph generation completed.")
