#!/usr/bin/env python3
"""Audit BANKING77 training labels with train-only out-of-fold predictions.

This is deliberately a data-audit step, not a model-promotion step. Every
probability is generated for a row by a classifier that did not train on that
row. Only the 9,233 training records are opened. Validation and test files are
not referenced or loaded anywhere in this script.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = PROJECT_ROOT / ".venv-encoder"
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp-012-label-quality-audit.json"
TRAIN_PATH = PROJECT_ROOT / "artifacts" / "encoder" / "data" / "train.jsonl"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "exp-012-label-audit"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
ARTIFACT_DIR = PROJECT_ROOT / CONFIG["outputs"]["artifact_directory"]
OOF_PATH = ARTIFACT_DIR / CONFIG["outputs"]["oof_predictions"]
REPORT_PATH = ARTIFACT_DIR / CONFIG["outputs"]["audit_report"]
MANIFEST_PATH = ARTIFACT_DIR / CONFIG["outputs"]["manifest"]
RESULT_PATH = EXPERIMENTS_DIR / "result.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"Experiment 012 label audit stopped: {message}")


if Path(sys.executable).resolve() != (EXPECTED_ENV / "bin" / "python").resolve():
    fail("use .venv-encoder/bin/python.")

if any(path.exists() for path in (OOF_PATH, REPORT_PATH, MANIFEST_PATH, RESULT_PATH)):
    fail("an output already exists; refusing to overwrite an audit run.")

if not TRAIN_PATH.is_file():
    fail(f"missing train-only source {TRAIN_PATH}.")

texts: list[str] = []
labels: list[int] = []
label_name_by_id: dict[int, str] = {}

with TRAIN_PATH.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        record = json.loads(line)
        text = record.get("text")
        label_id = record.get("label")
        label_name = record.get("label_name")
        if not isinstance(text, str) or not text.strip():
            fail(f"empty training text at line {line_number}.")
        if not isinstance(label_id, int) or not isinstance(label_name, str):
            fail(f"malformed training label at line {line_number}.")
        previous_name = label_name_by_id.setdefault(label_id, label_name)
        if previous_name != label_name:
            fail(f"label id {label_id} has inconsistent names.")
        texts.append(text)
        labels.append(label_id)

expected_rows = CONFIG["data"]["source_train_rows"]
if len(texts) != expected_rows:
    fail(f"expected {expected_rows} training rows, found {len(texts)}.")
if len(set(texts)) != len(texts):
    fail("duplicate training texts found; refusing ambiguous row indexing.")

observed_ids = sorted(label_name_by_id)
if observed_ids != list(range(len(observed_ids))):
    fail(f"training label ids are not contiguous: {observed_ids!r}.")
if len(observed_ids) != 77:
    fail(f"expected 77 labels, found {len(observed_ids)}.")

texts_array = np.asarray(texts, dtype=object)
labels_array = np.asarray(labels, dtype=np.int64)
n_rows = len(texts)
n_labels = len(observed_ids)
oof_probabilities = np.zeros((n_rows, n_labels), dtype=np.float32)
oof_folds = np.full(n_rows, -1, dtype=np.int64)

audit_config = CONFIG["audit"]
splitter = StratifiedKFold(
    n_splits=int(audit_config["folds"]),
    shuffle=True,
    random_state=int(audit_config["seed"]),
)

print("Open Model Training Lab — Experiment 012 train-only label audit")
print(f"source_train_rows: {n_rows}")
print(f"labels: {n_labels}")
print(f"folds: {audit_config['folds']}")
print(f"validation_rows_loaded: {CONFIG['data']['validation_rows_loaded']}")
print(f"test_rows_loaded: {CONFIG['data']['test_rows_loaded']}")
print(f"source_train_sha256: {sha256(TRAIN_PATH)}")
print("method: word TF-IDF + multinomial logistic regression")

for fold_number, (fit_indices, holdout_indices) in enumerate(
    splitter.split(texts_array, labels_array),
    start=1,
):
    vectorizer = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        max_features=int(audit_config["max_features"]),
        min_df=int(audit_config["min_df"]),
        ngram_range=tuple(audit_config["ngram_range"]),
        strip_accents="unicode",
        sublinear_tf=True,
        dtype=np.float32,
    )
    fit_matrix = vectorizer.fit_transform(texts_array[fit_indices])
    holdout_matrix = vectorizer.transform(texts_array[holdout_indices])
    classifier = LogisticRegression(
        C=float(audit_config["c"]),
        max_iter=int(audit_config["max_iter"]),
        random_state=int(audit_config["seed"]) + fold_number,
        solver=str(audit_config["solver"]),
        tol=1e-3,
    )
    classifier.fit(fit_matrix, labels_array[fit_indices])
    probabilities = classifier.predict_proba(holdout_matrix)
    if list(classifier.classes_) != observed_ids:
        fail(f"fold {fold_number} classifier classes differ from the label ids.")
    oof_probabilities[holdout_indices] = probabilities.astype(np.float32)
    oof_folds[holdout_indices] = fold_number
    print(
        f"fold_completed: {fold_number}/{audit_config['folds']} "
        f"fit_rows={len(fit_indices)} holdout_rows={len(holdout_indices)}"
    )

if np.any(oof_folds < 0):
    fail("some training rows did not receive an out-of-fold prediction.")
if not np.isfinite(oof_probabilities).all():
    fail("non-finite out-of-fold probabilities found.")

predicted_ids = np.argmax(oof_probabilities, axis=1)
row_indices = np.arange(n_rows)
given_probabilities = oof_probabilities[row_indices, labels_array]
predicted_probabilities = oof_probabilities[row_indices, predicted_ids]
disagreement = predicted_ids != labels_array

quality_threshold = float(audit_config["suspect_probability_threshold"])
candidate_order = sorted(
    range(n_rows),
    key=lambda index: (
        float(given_probabilities[index]),
        -float(predicted_probabilities[index] - given_probabilities[index]),
        index,
    ),
)

oof_accuracy = float(accuracy_score(labels_array, predicted_ids))
oof_macro_f1 = float(
    f1_score(labels_array, predicted_ids, average="macro", zero_division=0)
)
oof_log_loss = float(log_loss(labels_array, oof_probabilities, labels=observed_ids))

pair_counts: Counter[str] = Counter()
for expected_id, predicted_id in zip(labels_array, predicted_ids):
    if expected_id != predicted_id:
        pair_counts[
            f"{label_name_by_id[int(expected_id)]} -> "
            f"{label_name_by_id[int(predicted_id)]}"
        ] += 1

label_summaries: list[dict[str, object]] = []
for label_id in observed_ids:
    mask = labels_array == label_id
    label_summaries.append(
        {
            "label": label_name_by_id[label_id],
            "rows": int(mask.sum()),
            "oof_correct": int((predicted_ids[mask] == label_id).sum()),
            "oof_accuracy": float((predicted_ids[mask] == label_id).mean()),
            "mean_given_probability": float(given_probabilities[mask].mean()),
            "suspect_rows": int((given_probabilities[mask] < quality_threshold).sum()),
        }
    )
label_summaries.sort(key=lambda item: (item["oof_accuracy"], item["label"]))

top_count = min(int(audit_config["top_candidates"]), n_rows)
candidate_rows: list[dict[str, object]] = []
for rank, index in enumerate(candidate_order[:top_count], start=1):
    expected_id = int(labels_array[index])
    predicted_id = int(predicted_ids[index])
    candidate_rows.append(
        {
            "rank": rank,
            "row_index": index,
            "text": texts[index],
            "given_label": label_name_by_id[expected_id],
            "predicted_label": label_name_by_id[predicted_id],
            "given_probability": float(given_probabilities[index]),
            "predicted_probability": float(predicted_probabilities[index]),
            "margin": float(
                predicted_probabilities[index] - given_probabilities[index]
            ),
            "model_disagrees": bool(disagreement[index]),
            "fold": int(oof_folds[index]),
        }
    )

ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=False)

with OOF_PATH.open("w", encoding="utf-8") as handle:
    for index in range(n_rows):
        expected_id = int(labels_array[index])
        predicted_id = int(predicted_ids[index])
        handle.write(
            json.dumps(
                {
                    "row_index": index,
                    "text": texts[index],
                    "given_label": label_name_by_id[expected_id],
                    "predicted_label": label_name_by_id[predicted_id],
                    "given_probability": float(given_probabilities[index]),
                    "predicted_probability": float(predicted_probabilities[index]),
                    "model_disagrees": bool(disagreement[index]),
                    "fold": int(oof_folds[index]),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )

report = {
    "experiment": CONFIG["experiment"],
    "method": {
        "classifier": CONFIG["audit"]["classifier"],
        "vectorizer": CONFIG["audit"]["vectorizer"],
        "folds": int(CONFIG["audit"]["folds"]),
        "seed": int(CONFIG["audit"]["seed"]),
        "selection_policy": CONFIG["selection_policy"],
    },
    "data": {
        "source_train_rows": n_rows,
        "source_train_sha256": sha256(TRAIN_PATH),
        "validation_rows_loaded": 0,
        "test_rows_loaded": 0,
        "unique_train_texts": len(set(texts)),
    },
    "metrics": {
        "oof_accuracy": oof_accuracy,
        "oof_macro_f1": oof_macro_f1,
        "oof_log_loss": oof_log_loss,
        "disagreement_rows": int(disagreement.sum()),
        "disagreement_rate": float(disagreement.mean()),
        "rows_below_suspect_probability": int(
            (given_probabilities < quality_threshold).sum()
        ),
        "suspect_probability_threshold": quality_threshold,
    },
    "top_confusion_pairs": [
        {"pair": pair, "count": count}
        for pair, count in pair_counts.most_common(30)
    ],
    "weakest_labels": label_summaries[:20],
    "top_candidates": candidate_rows,
    "artifacts": {
        "oof_predictions": str(OOF_PATH),
        "audit_report": str(REPORT_PATH),
    },
}
REPORT_PATH.write_text(
    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

manifest = {
    "experiment": CONFIG["experiment"],
    "source_train_path": str(TRAIN_PATH),
    "source_train_sha256": sha256(TRAIN_PATH),
    "source_train_rows": n_rows,
    "validation_rows_loaded": 0,
    "test_rows_loaded": 0,
    "selection_policy": CONFIG["selection_policy"],
    "files": {
        "oof_predictions": {"path": str(OOF_PATH), "sha256": sha256(OOF_PATH)},
        "audit_report": {"path": str(REPORT_PATH), "sha256": sha256(REPORT_PATH)},
    },
}
MANIFEST_PATH.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

compact_result = {
    "experiment": CONFIG["experiment"],
    "status": "audit_complete",
    "source_train_rows": n_rows,
    "source_train_sha256": sha256(TRAIN_PATH),
    "folds": int(CONFIG["audit"]["folds"]),
    "oof_accuracy": oof_accuracy,
    "oof_macro_f1": oof_macro_f1,
    "oof_log_loss": oof_log_loss,
    "disagreement_rows": int(disagreement.sum()),
    "rows_below_suspect_probability": int(
        (given_probabilities < quality_threshold).sum()
    ),
    "validation_rows_loaded": 0,
    "test_rows_loaded": 0,
    "selection_policy": CONFIG["selection_policy"],
    "audit_report": str(REPORT_PATH),
    "manifest": str(MANIFEST_PATH),
}
RESULT_PATH.write_text(
    json.dumps(compact_result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"oof_accuracy: {oof_accuracy:.6f}")
print(f"oof_macro_f1: {oof_macro_f1:.6f}")
print(f"oof_log_loss: {oof_log_loss:.6f}")
print(f"disagreement_rows: {int(disagreement.sum())}")
print(
    "rows_below_suspect_probability: "
    f"{int((given_probabilities < quality_threshold).sum())}"
)
print(f"audit_report: {REPORT_PATH}")
print(f"manifest: {MANIFEST_PATH}")
print(f"result: {RESULT_PATH}")
print("exp_012_label_audit_ok: True")
