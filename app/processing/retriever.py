import ast
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from detection import detectFaces
from alignment import alignFace
from embeding import getEmbedding


BASE_DIR = Path(__file__).resolve().parents[2]

SEARCH_DIR = BASE_DIR / "public" / "search"
UPLOAD_DIR = BASE_DIR / "public" / "upload"
EMBEDDING_CSV = BASE_DIR / "embeddings.csv"

RESULTS_DIR = BASE_DIR / "app" / "processing" / "evaluation_results"

TARGET_PERSON = "george_w_bush"
PRECISION_TARGET = 0.95

THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.01), 2)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".tif", ".tiff"
}


def normalize_name(name):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        name.lower()
    ).strip("_")


def find_query_image():
    if not SEARCH_DIR.is_dir():
        raise FileNotFoundError(
            f"Search directory not found: {SEARCH_DIR}"
        )

    images = sorted(
        path for path in SEARCH_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise FileNotFoundError(
            f"No supported images found in {SEARCH_DIR}"
        )

    preferred = [
        path for path in images
        if path.stem.lower() == "search"
    ]

    if preferred:
        return preferred[0]

    if len(images) == 1:
        return images[0]

    raise RuntimeError(
        "Multiple query images found. Keep one image in "
        "public/search or name the intended image search.jpg."
    )


def generate_query_embedding(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read query image: {image_path}"
        )

    faces = detectFaces(str(image_path))

    if len(faces) == 0:
        raise ValueError(
            "No face detected in the query image."
        )

    if len(faces) > 1:
        raise ValueError(
            f"Found {len(faces)} faces in the query image. "
            "Use a selfie containing only one person."
        )

    aligned_face = alignFace(
        image,
        faces[0]["landmarks"]
    )

    embedding = getEmbedding(aligned_face)

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    ).flatten()

    if not np.all(np.isfinite(embedding)):
        raise ValueError("Query embedding contains invalid values.")

    if np.linalg.norm(embedding) == 0:
        raise ValueError("Query embedding has zero norm.")

    return embedding


def parse_embedding(value):
    if isinstance(value, np.ndarray):
        embedding = value.astype(np.float32)

    else:
        try:
            parsed = json.loads(str(value))
        except (json.JSONDecodeError, TypeError):
            parsed = ast.literal_eval(str(value))

        embedding = np.asarray(
            parsed,
            dtype=np.float32
        )

    embedding = embedding.flatten()

    if not np.all(np.isfinite(embedding)):
        raise ValueError("Embedding contains invalid values.")

    if np.linalg.norm(embedding) == 0:
        raise ValueError("Embedding has zero norm.")

    return embedding


def cosine_similarity(a, b):
    if a.shape != b.shape:
        raise ValueError(
            f"Embedding dimensions differ: {a.shape} vs {b.shape}"
        )

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return -1.0

    return float(np.dot(a, b) / denominator)


def get_identity(filename):
    name = normalize_name(Path(filename).stem)

    if "george_w_bush" in name:
        return TARGET_PERSON

    if "colin_powell" in name:
        return "colin_powell"

    return None


def resolve_image_path(path_string):
    path = Path(str(path_string))

    if path.is_absolute() and path.exists():
        return path

    candidates = [
        BASE_DIR / path,
        UPLOAD_DIR / path.name
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def load_gallery_scores(query_embedding):
    if not EMBEDDING_CSV.is_file():
        raise FileNotFoundError(
            f"Embedding CSV not found: {EMBEDDING_CSV}\n"
            "Run your existing pipeline first to generate embeddings.csv."
        )

    df = pd.read_csv(EMBEDDING_CSV)

    required_columns = {"img_path", "embedding"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"embeddings.csv must contain {required_columns}. "
            f"Found columns: {list(df.columns)}"
        )

    # Keep the highest similarity for each photograph.
    # A group photo may have several face embeddings.
    image_scores = {}

    for _, row in df.iterrows():
        image_path = resolve_image_path(row["img_path"])
        image_name = image_path.name

        identity = get_identity(image_name)

        if identity is None:
            continue

        try:
            stored_embedding = parse_embedding(row["embedding"])

            score = cosine_similarity(
                query_embedding,
                stored_embedding
            )

            if (
                image_name not in image_scores
                or score > image_scores[image_name]["similarity"]
            ):
                image_scores[image_name] = {
                    "image": image_name,
                    "path": str(image_path),
                    "identity": identity,
                    "similarity": score,
                    "actual_positive": identity == TARGET_PERSON
                }

        except Exception as exc:
            print(
                f"Skipping embedding for {image_name}: {exc}"
            )

    return list(image_scores.values())


def calculate_metrics(gallery, threshold):
    tp = sum(
        item["actual_positive"]
        and item["similarity"] >= threshold
        for item in gallery
    )

    fp = sum(
        not item["actual_positive"]
        and item["similarity"] >= threshold
        for item in gallery
    )

    fn = sum(
        item["actual_positive"]
        and item["similarity"] < threshold
        for item in gallery
    )

    tn = sum(
        not item["actual_positive"]
        and item["similarity"] < threshold
        for item in gallery
    )

    total = tp + fp + fn + tn
    returned = tp + fp

    precision = tp / returned if returned else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "threshold": round(float(threshold), 2),
        "gallery_images": total,
        "returned": returned,
        "correct": tp,
        "wrong": fp,
        "missed": fn,
        "true_negative": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    query_path = find_query_image()

    print(f"Query image: {query_path}")
    print("Generating query embedding...")

    query_embedding = generate_query_embedding(query_path)

    print("Loading stored gallery embeddings...")

    gallery = load_gallery_scores(query_embedding)

    if not gallery:
        raise RuntimeError(
            "No labelled gallery images found. Check embeddings.csv "
            "and ensure filenames contain George_W_Bush or Colin_Powell."
        )

    positives = sum(
        item["actual_positive"] for item in gallery
    )

    negatives = len(gallery) - positives

    if positives == 0 or negatives == 0:
        raise RuntimeError(
            "The labelled gallery must contain both George W. Bush "
            "and Colin Powell photographs."
        )

    print(f"Unique gallery images evaluated: {len(gallery)}")
    print(f"George W. Bush images: {positives}")
    print(f"Colin Powell images: {negatives}")

    # Save per-image scores so results can be inspected.
    scores_path = RESULTS_DIR / "image_similarity_scores.csv"

    with scores_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "identity",
                "similarity",
                "actual_positive"
            ]
        )

        writer.writeheader()

        for item in sorted(
            gallery,
            key=lambda x: x["similarity"],
            reverse=True
        ):
            writer.writerow({
                "image": item["image"],
                "identity": item["identity"],
                "similarity": round(item["similarity"], 6),
                "actual_positive": item["actual_positive"]
            })

    # Evaluate every candidate threshold.
    rows = [
        calculate_metrics(gallery, threshold)
        for threshold in THRESHOLDS
    ]

    report_path = RESULTS_DIR / "threshold_report.csv"

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    # Choose the highest-recall threshold that meets
    # the configured precision target.
    eligible = [
        row for row in rows
        if row["precision"] >= PRECISION_TARGET
        and row["returned"] > 0
    ]

    if eligible:
        best = max(
            eligible,
            key=lambda row: (
                row["recall"],
                row["precision"],
                row["threshold"]
            )
        )

        selection_rule = (
            f"Highest recall with precision >= {PRECISION_TARGET:.0%}"
        )

    else:
        best = max(
            rows,
            key=lambda row: (
                row["f1"],
                row["precision"]
            )
        )

        selection_rule = (
            f"No threshold met the {PRECISION_TARGET:.0%} precision "
            "target; selected the threshold with the highest F1 score."
        )

    best_threshold = best["threshold"]

    matches = [
        item for item in gallery
        if item["similarity"] >= best_threshold
    ]

    matches_path = RESULTS_DIR / "best_threshold_results.csv"

    with matches_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        fields = [
            "image",
            "identity",
            "similarity",
            "correct_match"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for item in sorted(
            matches,
            key=lambda x: x["similarity"],
            reverse=True
        ):
            writer.writerow({
                "image": item["image"],
                "identity": item["identity"],
                "similarity": round(item["similarity"], 6),
                "correct_match": item["actual_positive"]
            })

    print("\n========== THRESHOLD EVALUATION ==========")
    print(f"Query image: {query_path.name}")
    print(f"Gallery size: {len(gallery)}")
    print(f"Target photographs: {positives}")
    print(f"Other-person photographs: {negatives}")
    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"Selection rule: {selection_rule}")
    print(f"Returned photographs: {best['returned']}")
    print(f"Correct photographs: {best['correct']}")
    print(f"Incorrect photographs: {best['wrong']}")
    print(f"Missed target photographs: {best['missed']}")
    print(f"True negatives: {best['true_negative']}")
    print(f"Accuracy: {best['accuracy']:.2%}")
    print(f"Precision: {best['precision']:.2%}")
    print(f"Recall: {best['recall']:.2%}")
    print(f"F1 score: {best['f1']:.4f}")

    print("\nSaved reports:")
    print(f"Threshold report: {report_path}")
    print(f"Image similarity scores: {scores_path}")
    print(f"Selected-threshold matches: {matches_path}")

    print(
        "\nIMPORTANT: This threshold is calibrated on this gallery "
        "and query. Validate it using other query images and identities "
        "before treating it as a general threshold."
    )


if __name__ == "__main__":
    main()
