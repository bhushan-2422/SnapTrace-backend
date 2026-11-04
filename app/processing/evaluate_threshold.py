import sys
import csv
import re
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "public" / "upload"
SEARCH_DIR = BASE_DIR / "public" / "search"
RESULTS_DIR = BASE_DIR / "app" / "processing" / "evaluation_results"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import processImage

TARGET_PERSON = "george_w_bush"
PRECISION_TARGET = 0.95

THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.01), 2)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
}


def normalize_name(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def find_query_image():
    if not SEARCH_DIR.is_dir():
        raise FileNotFoundError(f"Search directory not found: {SEARCH_DIR}")

    images = sorted(
        path for path in SEARCH_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise FileNotFoundError(
            f"No supported images found in {SEARCH_DIR}"
        )

    preferred = [
        image for image in images
        if image.stem.lower() == "search"
    ]

    if preferred:
        return preferred[0]

    if len(images) == 1:
        return images[0]

    raise RuntimeError(
        "Multiple images found in the search directory. "
        "Keep one query image or name the intended image search.jpg."
    )


def get_embeddings(image_path):
    results = processImage(str(image_path))
    embeddings = []

    for result in results:
        embedding = result.get("embedding")

        if embedding is None:
            continue

        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)

        if np.all(np.isfinite(vector)) and np.linalg.norm(vector) > 0:
            embeddings.append(vector)

    return embeddings


def cosine_similarity(a, b):
    if a.shape != b.shape:
        raise ValueError(
            f"Embedding dimension mismatch: {a.shape} vs {b.shape}"
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


def main():
    if not UPLOAD_DIR.is_dir():
        raise FileNotFoundError(
            f"Gallery directory not found: {UPLOAD_DIR}"
        )

    query_path = find_query_image()

    image_paths = sorted(
        path for path in UPLOAD_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(f"No gallery images found in {UPLOAD_DIR}")

    print(f"Query image: {query_path}")
    print(f"Gallery images found: {len(image_paths)}")

    print("\nGenerating query embedding...")
    query_embeddings = get_embeddings(query_path)

    if not query_embeddings:
        raise RuntimeError(
            "No face detected in the query image. "
            "Check the image and your existing pipeline."
        )

    if len(query_embeddings) != 1:
        raise RuntimeError(
            f"Expected one face in the query image, "
            f"but detected {len(query_embeddings)}. "
            "Use a selfie containing one person."
        )

    query_embedding = query_embeddings[0]
    gallery = []

    print("\nGenerating gallery embeddings...")

    for index, image_path in enumerate(image_paths, start=1):
        identity = get_identity(image_path.name)

        if identity is None:
            print(f"Skipping unlabelled filename: {image_path.name}")
            continue

        try:
            face_embeddings = get_embeddings(image_path)

            if not face_embeddings:
                print(f"No faces detected: {image_path.name}")
                continue

            similarity = max(
                cosine_similarity(query_embedding, face_embedding)
                for face_embedding in face_embeddings
            )

            gallery.append({
                "image": image_path.name,
                "identity": identity,
                "similarity": similarity,
                "actual_positive": identity == TARGET_PERSON
            })

        except Exception as exc:
            print(f"Failed: {image_path.name}: {exc}")

        if index % 10 == 0 or index == len(image_paths):
            print(f"Processed {index}/{len(image_paths)}")

    if not gallery:
        raise RuntimeError("No valid gallery images were evaluated.")

    total_positive = sum(item["actual_positive"] for item in gallery)
    total_negative = len(gallery) - total_positive

    if total_positive == 0 or total_negative == 0:
        raise RuntimeError(
            "The gallery must contain both target and other-person images."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for threshold in THRESHOLDS:
        tp = sum(
            item["actual_positive"] and item["similarity"] >= threshold
            for item in gallery
        )

        fp = sum(
            not item["actual_positive"] and item["similarity"] >= threshold
            for item in gallery
        )

        fn = sum(
            item["actual_positive"] and item["similarity"] < threshold
            for item in gallery
        )

        tn = sum(
            not item["actual_positive"] and item["similarity"] < threshold
            for item in gallery
        )

        returned = tp + fp
        precision = tp / returned if returned else 0.0
        recall = tp / total_positive
        accuracy = (tp + tn) / len(gallery)

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        rows.append({
            "threshold": round(float(threshold), 2),
            "total_gallery": len(gallery),
            "returned": returned,
            "correct": tp,
            "wrong": fp,
            "missed": fn,
            "true_negative": tn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    report_path = RESULTS_DIR / "threshold_report.csv"

    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [
        row for row in rows
        if row["precision"] >= PRECISION_TARGET and row["returned"] > 0
    ]

    if valid_rows:
        best = max(
            valid_rows,
            key=lambda row: (row["recall"], row["precision"])
        )
        reason = (
            f"Highest recall with precision >= {PRECISION_TARGET:.0%}"
        )
    else:
        best = max(
            rows,
            key=lambda row: (row["f1"], row["precision"])
        )
        reason = (
            f"No threshold achieved {PRECISION_TARGET:.0%} precision; "
            "selected the threshold with the highest F1 score."
        )

    matches_path = RESULTS_DIR / "best_threshold_results.csv"

    with matches_path.open("w", newline="", encoding="utf-8") as file:
        fields = ["image", "identity", "similarity", "correct_match"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for item in sorted(
            gallery,
            key=lambda item: item["similarity"],
            reverse=True
        ):
            if item["similarity"] >= best["threshold"]:
                writer.writerow({
                    "image": item["image"],
                    "identity": item["identity"],
                    "similarity": round(item["similarity"], 4),
                    "correct_match": item["actual_positive"]
                })

    print("\n========== RESULTS ==========")
    print(f"Target images evaluated: {total_positive}")
    print(f"Other-person images evaluated: {total_negative}")
    print(f"Selected threshold: {best['threshold']:.2f}")
    print(f"Selection rule: {reason}")
    print(f"Returned photographs: {best['returned']}")
    print(f"Correct photographs: {best['correct']}")
    print(f"Incorrect photographs: {best['wrong']}")
    print(f"Missed photographs: {best['missed']}")
    print(f"Accuracy: {best['accuracy']:.2%}")
    print(f"Precision: {best['precision']:.2%}")
    print(f"Recall: {best['recall']:.2%}")
    print(f"F1 score: {best['f1']:.4f}")

    print(f"\nFull threshold report: {report_path}")
    print(f"Results at selected threshold: {matches_path}")


if __name__ == "__main__":
    main()