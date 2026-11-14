"""Audit BANKING77 and build a group-aware development split."""

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE = (
    f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{REVISION}/banking_data"
)
CHECKSUMS = {
    "categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
SEED = 42
NEAR_SIMILARITY = 0.90
PII_PATTERNS = {
    "email_like": re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.I),
    "phone_like": re.compile(r"(?<!\w)\+?\d(?:[ ().-]*\d){9,14}(?![\w.-]*\d)"),
    "card_like": re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?![ -]?\d)"),
    "iban_like": re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b"),
}
REVIEW_PAIRS = [
    ("card_arrival", "card_delivery_estimate"),
    ("top_up_failed", "top_up_reverted"),
]


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames != ["text", "category"]:
            raise ValueError(f"Unexpected columns in {path.name}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed record in {path.name}")
    return rows


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def quantiles(values):
    return dict(
        zip(
            ("min", "p50", "p90", "p95", "p99", "max"),
            np.percentile(values, [0, 50, 90, 95, 99, 100]).tolist(),
            strict=True,
        )
    )


def near_groups(texts, labels):
    # Independent audit vocabulary. This matrix is never used for model training.
    matrix = TfidfVectorizer(analyzer="char", ngram_range=(3, 5)).fit_transform(texts)
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute").fit(matrix)
    parents = list(range(len(texts)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    same_label = 0
    cross_label_pairs = []
    for start in range(0, len(texts), 256):
        indices = neighbors.radius_neighbors(
            matrix[start : start + 256], radius=1 - NEAR_SIMILARITY, return_distance=False
        )
        for index, matches in enumerate(indices, start=start):
            for match in matches:
                if match <= index:
                    continue
                left, right = sorted((root(index), root(int(match))))
                parents[right] = left
                if labels[index] == labels[match]:
                    same_label += 1
                else:
                    cross_label_pairs.append((index, match))
    return np.array([root(index) for index in range(len(texts))]), same_label, cross_label_pairs


def make_splits(labels, groups):
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_ids = np.empty(len(labels), dtype=int)
    for fold, (_, indices) in enumerate(folds.split(labels, labels, groups)):
        fold_ids[indices] = fold
    split_ids = np.array(["selection", "policy", "fit", "fit", "fit"])[fold_ids]
    for split in ("fit", "selection", "policy"):
        if set(labels[split_ids == split]) != set(labels):
            raise ValueError(f"Missing intent in {split}")
    seen = defaultdict(set)
    for group, split in zip(groups, split_ids, strict=True):
        seen[group].add(split)
    if any(len(splits) != 1 for splits in seen.values()):
        raise ValueError("A text group spans multiple splits")
    return split_ids


def save_charts(output, counts, lengths, words, labels):
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

    ordered = sorted(counts, key=counts.get)
    fig, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
    ax.bar(range(1, len(ordered) + 1), [counts[label] for label in ordered], color="#316a8d")
    ax.axhline(np.median(list(counts.values())), color="#b44931", ls="--", label="Медиана")
    ax.set(
        title="Разброс размера классов BANKING77 train",
        xlabel="Класс в порядке роста числа обращений",
        ylabel="Число обращений",
    )
    ax.legend(frameon=False)
    fig.savefig(figures / "class_balance.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    for ax, values, unit, bins in zip(
        axes, (lengths, words), ("символов", "слов"), (32, 24), strict=True
    ):
        ax.hist(values, bins=bins, color="#316a8d", edgecolor="white", linewidth=0.5)
        p95 = np.percentile(values, 95)
        ax.axvline(p95, color="#b44931", ls="--", label=f"p95 = {p95:g}")
        ax.set(xlabel=f"Число {unit}", ylabel="Число обращений")
        ax.legend(frameon=False)
    fig.suptitle("Длина обращений в BANKING77 train")
    fig.savefig(figures / "text_length.png", dpi=160)
    plt.close(fig)

    pair_labels = [label for pair in REVIEW_PAIRS for label in pair]
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.boxplot(
        [lengths[labels == label] for label in pair_labels],
        tick_labels=pair_labels,
        orientation="horizontal",
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#d8e8f0"},
        medianprops={"color": "#b44931", "linewidth": 2},
    )
    ax.invert_yaxis()
    ax.set(title="Длина текста в двух парах близких тем train", xlabel="Число символов")
    fig.savefig(figures / "intent_pairs.png", dpi=160)
    plt.close(fig)


def run(data, output):
    for file, expected in CHECKSUMS.items():
        if hashlib.sha256((data / file).read_bytes()).hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch: {file}")
    categories = json.loads((data / "categories.json").read_text(encoding="utf-8"))
    if len(categories) != 77 or len(set(categories)) != 77:
        raise ValueError("Expected 77 distinct categories")

    train = read_rows(data / "train.csv")
    test = read_rows(data / "test.csv")
    if len(train) != 10003 or len(test) != 3080:
        raise ValueError("Unexpected train/test sizes")
    counts = Counter(row["category"] for row in train)
    unknown = set(counts) - set(categories)
    if unknown:
        raise ValueError(f"Unknown or empty categories: {sorted(unknown)}")

    exact = defaultdict(list)
    raw = Counter()
    for index, row in enumerate(train, start=1):
        key = normalize(row["text"])
        if key:
            exact[key].append((index, row["category"]))
            raw[row["text"]] += 1
    conflicts = {
        key: records for key, records in exact.items() if len({label for _, label in records}) > 1
    }
    repeated = sum(len(records) - 1 for key, records in exact.items() if key not in conflicts)
    texts = [key for key in exact if key not in conflicts]
    ids, labels = zip(*(exact[key][0] for key in texts), strict=True)
    labels = np.array(labels)
    groups, same_label, cross_label_pairs = near_groups(texts, labels)
    splits = make_splits(labels, groups)
    split_counts = {
        split: Counter(labels[splits == split].tolist()) for split in ("fit", "selection", "policy")
    }
    group_sizes = Counter(groups)
    group_labels = defaultdict(set)
    for group, label in zip(groups, labels, strict=True):
        group_labels[group].add(label)

    train_texts = [row["text"].strip() for row in train]
    lengths = np.array([len(text) for text in train_texts])
    words = [len(text.split()) for text in train_texts]
    train_labels = np.array([row["category"] for row in train])
    test_keys = [normalize(row["text"]) for row in test]
    summary = {
        "source": {"revision": REVISION, "url": SOURCE, "sha256": CHECKSUMS},
        "train": {
            "rows": len(train),
            "classes": len(counts),
            "per_class": dict(sorted(counts.items())),
            "empty_texts": len(train) - sum(raw.values()),
            "unknown_labels": len(unknown),
            "characters": quantiles(lengths),
            "words_by_whitespace": quantiles(words),
            "texts_above_2000_chars": int(np.count_nonzero(lengths > 2000)),
            "pii_pattern_hits_by_row": {
                name: sum(bool(pattern.search(text)) for text in train_texts)
                for name, pattern in PII_PATTERNS.items()
            },
            "raw_repeated_text_groups": sum(size > 1 for size in raw.values()),
            "raw_repeated_rows_beyond_first": sum(size - 1 for size in raw.values()),
            "normalized_repeated_text_groups": sum(len(records) > 1 for records in exact.values()),
            "same_label_repeated_rows_removed": repeated,
            "conflicts": [
                [{"sample_id": f"train:{index:05}", "intent": label} for index, label in records]
                for records in conflicts.values()
            ],
            "rows_removed_for_conflicts": sum(map(len, conflicts.values())),
        },
        "near_duplicates": {
            "cosine_threshold": NEAR_SIMILARITY,
            "pairs_same_label": same_label,
            "pairs_cross_label": [
                [
                    {"sample_id": f"train:{ids[index]:05}", "intent": str(labels[index])}
                    for index in pair
                ]
                for pair in cross_label_pairs
            ],
            "groups_with_multiple_texts": sum(size > 1 for size in group_sizes.values()),
            "mixed_label_groups": sum(len(values) > 1 for values in group_labels.values()),
            "largest_group": max(group_sizes.values()),
        },
        "test_text_audit": {
            "rows": len(test_keys),
            "empty_texts": test_keys.count(""),
            "exact_matches_with_raw_train": sum(key in exact for key in test_keys),
            "exact_matches_with_clean_train": sum(
                key in exact and key not in conflicts for key in test_keys
            ),
        },
        "development": {
            "rows": len(ids),
            "groups": len(group_sizes),
            "splits": {
                split: {
                    "rows": sum(per_class.values()),
                    "per_class": dict(sorted(per_class.items())),
                }
                for split, per_class in split_counts.items()
            },
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "split_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["sample_id", "text_group", "split"])
        for index, group, split in zip(ids, groups, splits, strict=True):
            writer.writerow([f"train:{index:05}", f"train:{ids[group]:05}", split])

    save_charts(output, counts, lengths, words, train_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/banking77"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)
