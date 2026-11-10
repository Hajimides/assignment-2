"""Fit on the training split only, evaluate once, and predict the separate novel."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.pipeline import Pipeline

from prepare_data import DATA, ROOT, download, read_csv, sentences, sha256, write_csv
from text_features import tokenize

OUT = ROOT / "output"
LABELS = ["negative", "positive"]
SEED = 42
(ROOT / ".cache").mkdir(exist_ok=True)


def make_model(kind="tfidf"):
    options = dict(tokenizer=tokenize, token_pattern=None, lowercase=False)
    if kind == "tfidf":
        features = TfidfVectorizer(**options, ngram_range=(1, 2), min_df=2,
                                   max_features=10000, sublinear_tf=True)
    else:
        features = CountVectorizer(**options, max_features=200)
    return Pipeline([("features", features), ("classifier", LogisticRegression(
        C=1.0, max_iter=2000, solver="liblinear", random_state=SEED))])


def evaluate(y_true, predicted):
    return {"accuracy": float(accuracy_score(y_true, predicted)),
            "confusion_matrix": confusion_matrix(y_true, predicted, labels=LABELS).tolist(),
            "confusion_matrix_order": LABELS,
            "classification_report": classification_report(y_true, predicted, labels=LABELS,
                                                             output_dict=True, zero_division=0)}


def normalized(text):
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)


def near_duplicates(train, test):
    def grams(text):
        t = normalized(text)
        return {t[i:i+5] for i in range(max(1, len(t)-4))}
    a = [(r, grams(r["text"])) for r in train]
    flagged = []
    for r in test:
        right = grams(r["text"])
        for left, bag in a:
            score = len(right & bag) / max(1, len(right | bag))
            if score >= .8:
                flagged.append({"train_id": left["item_id"], "test_id": r["item_id"], "jaccard_5gram": round(score, 6)})
    return flagged


def explain(model, text, top=4):
    vector = model["features"].transform([text]).tocsr()
    names = model["features"].get_feature_names_out()
    coefs = model["classifier"].coef_[0]
    parts = [(names[i], float(v*coefs[i])) for i, v in zip(vector.indices, vector.data)]
    parts.sort(key=lambda x: -abs(x[1]))
    return " | ".join(f"{name}:{value:+.3f}" for name, value in parts[:top])


def predictions(model, rows):
    texts = [r["text"] for r in rows]
    probs = model.predict_proba(texts)
    positive_index = list(model.classes_).index("positive")
    predicted = model.predict(texts)
    matrix = model["features"].transform(texts)
    out = []
    for i, r in enumerate(rows):
        p = float(probs[i, positive_index])
        out.append({**r, "predicted_label": predicted[i], "prob_positive": round(p, 8),
                    "prob_negative": round(1-p, 8), "confidence": round(max(p, 1-p), 8),
                    "low_confidence": int(.4 <= p <= .6), "nonzero_features": int(matrix[i].nnz),
                    "top_contributions": explain(model, r["text"])})
    return out


def wilson(correct, total):
    p = correct/total
    z = 1.96
    center = (p+z*z/(2*total))/(1+z*z/total)
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return [center-half, center+half]


def make_review_files(rows, analysis):
    # Once created, never overwrite the student's entries.
    rng = np.random.default_rng(3242)
    if not (DATA / "human_review.csv").exists():
        sample = []
        for source in sorted({r["source"] for r in rows}):
            for label in LABELS:
                candidates = [r for r in rows if r["source"] == source and r["label"] == label]
                for i in rng.choice(len(candidates), min(5, len(candidates)), replace=False):
                    r = candidates[i]
                    sample.append({"candidate_id": r["item_id"], "source": r["source"],
                                   "text": r["text"], "ai_label": r["label"],
                                   "human_label": "", "human_note": ""})
        write_csv(DATA / "human_review.csv", sample)
    if not (DATA / "analysis_human_review.csv").exists():
        # Uniform in text order, independent of predictions; keep gold blind.
        indices = np.linspace(0, len(analysis)-1, 40, dtype=int)
        write_csv(DATA / "analysis_human_review.csv", [
            {"sentence_id": analysis[i]["sentence_id"], "line_number": analysis[i]["line_number"],
             "text": analysis[i]["text"], "human_label": "", "human_note": ""} for i in indices])


def review_status():
    training = read_csv(DATA / "human_review.csv")
    completed = [r for r in training if r["human_label"].strip()]
    gold = read_csv(DATA / "analysis_human_review.csv")
    valid = {"positive", "negative", "neutral", "mixed", "unclear"}
    for r in gold:
        if r["human_label"].strip() and r["human_label"].strip() not in valid:
            raise ValueError(f"Invalid analysis review label: {r['human_label']}")
    return {"training_sample": len(training), "training_human_completed": len(completed),
            "training_agreement_with_initial_AI": (sum(r["human_label"] == r["ai_label"] for r in completed)/len(completed)) if completed else None,
            "analysis_sample": len(gold), "analysis_human_completed": sum(bool(r["human_label"].strip()) for r in gold)}


def plot_results(metrics, analysis, bins):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)
    cm = np.array(metrics["confusion_matrix"])
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
    for (i,j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=23,
                color="white" if value > cm.max()*.55 else "#17324b")
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=LABELS, yticklabels=LABELS,
           xlabel="Predicted label", ylabel="Reference label", title="Held-out test set · TF-IDF + Logistic Regression")
    fig.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11,4.8), constrained_layout=True)
    ax.plot([r["midpoint_percent"] for r in bins], [r["mean_prob_positive"] for r in bins],
            "o-", color="#0b756c", lw=2, ms=4)
    ax.axhline(.5, color="#999", ls="--", lw=1)
    ax.set(xlim=(0,100), ylim=(0,1), xlabel="Position in text (% of body characters)",
           ylabel="Mean model P(positive)", title="To Live · 20 equal character-position bins")
    ax.text(.01, .02, "Model tendency; not a measure of emotional intensity or literary value.", transform=ax.transAxes, fontsize=9, color="#666")
    fig.savefig(OUT / "huozhe_sentiment_curve.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-text", type=Path, default=ROOT / "余华_活着.txt")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    rows = read_csv(DATA / "labeled_sentences.csv")
    keys = [normalized(r["text"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate normalized training texts")
    if min(Counter(r["label"] for r in rows).values()) < 200:
        raise ValueError("Need at least 200 items per class")
    train, test = train_test_split(rows, test_size=.2, random_state=SEED,
                                   stratify=[r["label"] for r in rows])
    write_csv(OUT / "train_split.csv", train)
    write_csv(OUT / "test_split.csv", test)
    tx, ty = [r["text"] for r in train], [r["label"] for r in train]
    vx, vy = [r["text"] for r in test], [r["label"] for r in test]
    model = make_model()
    model.fit(tx, ty)  # TF-IDF's vocabulary and IDF see the 80% training split only.
    result = predictions(model, test)
    result = [{**r, "correct": int(r["label"] == r["predicted_label"])} for r in result]
    write_csv(OUT / "test_predictions.csv", result)
    errors = sorted([r for r in result if not r["correct"]], key=lambda r: -r["confidence"])
    write_csv(OUT / "misclassified.csv", errors, list(result[0]))
    metrics = evaluate(vy, [r["predicted_label"] for r in result])
    metrics["accuracy_wilson_95_interval"] = wilson(sum(r["correct"] for r in result), len(result))
    comparisons = []
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train),1)), ty)
    for name, predicted in [("majority_class", dummy.predict(np.zeros((len(test),1)))),
                            ("count_top_200_logreg", make_model("count").fit(tx,ty).predict(vx)),
                            ("tfidf_logreg_primary", [r["predicted_label"] for r in result])]:
        m = evaluate(vy, predicted)
        comparisons.append({"model": name, "accuracy": m["accuracy"],
                            "macro_f1": m["classification_report"]["macro avg"]["f1-score"]})
    write_csv(OUT / "model_comparison.csv", comparisons)
    # A separate diagnostic on TRAINING ONLY: each fold holds out an entire work.
    groups = np.array([r["source"] for r in train])
    fold_results = []
    tx_arr, ty_arr = np.array(tx), np.array(ty)
    for a,b in LeaveOneGroupOut().split(tx_arr, ty_arr, groups):
        fold = clone(model).fit(tx_arr[a],ty_arr[a])
        m = evaluate(ty_arr[b], fold.predict(tx_arr[b]))
        fold_results.append({"held_out_work": groups[b][0], "validation_n": len(b),
                             "accuracy": m["accuracy"], "macro_f1": m["classification_report"]["macro avg"]["f1-score"]})
    write_csv(OUT / "cross_work_training_cv.csv", fold_results)
    leakage = near_duplicates(train, test)
    write_csv(OUT / "near_duplicate_audit.csv", leakage, ["train_id", "test_id", "jaccard_5gram"])
    if leakage:
        raise ValueError("Near duplicates across train/test; resolve before reporting")
    names = model["features"].get_feature_names_out()
    weights = sorted(zip(names,model["classifier"].coef_[0]), key=lambda x:x[1])
    write_csv(OUT / "feature_weights.csv", [{"feature": n, "positive_log_odds_weight": float(w)} for n,w in weights])

    path = args.analysis_text
    if not path.exists():
        path, _ = download("余华_活着")
    text = path.read_text(encoding="utf-8-sig")
    # Preserve the original file; strip only its title and terminal marker.
    target = [r for r in sentences(text)
              if not r["text"].startswith("『活着／作者：") and r["text"] != "＝已完结＝"]
    beginning, end = target[0]["char_start"], target[-1]["char_end"]
    for i,r in enumerate(target,1):
        r["sentence_id"] = i
        r["position_percent"] = round(100*(r["char_start"]-beginning)/max(1,end-beginning),6)
        r["short_sentence"] = int(len(normalized(r["text"])) < 6)
        r["suspected_text_issue"] = int(bool(re.search(r"[＊�]|from:|ｆｒｏｍ", r["text"], re.I)))
        if text[r["char_start"]:r["char_end"]] != r["text"]:
            raise AssertionError("Original-text offset mismatch")
    target_keys = {normalized(r["text"]) for r in target}
    overlap = [r["item_id"] for r in rows if normalized(r["text"]) in target_keys]
    if overlap:
        raise ValueError(f"Training/analysis sentence overlap: {overlap}")
    analysis = predictions(model, target)
    # Public predictions retain offsets and scores; the full novel stays local.
    write_csv(OUT / "private/huozhe_full_predictions.csv", analysis)
    write_csv(OUT / "huozhe_predictions.csv", [{k:v for k,v in r.items() if k != "text"} for r in analysis])
    bins = []
    for i in range(20):
        section = [r for r in analysis if min(19,int(r["position_percent"]//5)) == i]
        bins.append({"bin": i+1, "start_percent": 5*i, "end_percent": 5*(i+1),
                     "midpoint_percent": 5*i+2.5, "sentence_count": len(section),
                     "mean_prob_positive": float(np.mean([r["prob_positive"] for r in section])),
                     "positive_label_fraction": float(np.mean([r["predicted_label"] == "positive" for r in section])),
                     "low_confidence_count": sum(r["low_confidence"] for r in section)})
    write_csv(OUT / "huozhe_position_bins.csv", bins)
    make_review_files(rows, analysis)
    status = review_status()
    # Optional target accuracy only after the student supplies actual reference labels.
    reviewed = read_csv(DATA / "analysis_human_review.csv")
    analysis_by_id = {str(r["sentence_id"]): r for r in analysis}
    seen_review_ids = set()
    review_predictions = []
    for r in reviewed:
        key = r["sentence_id"]
        if key in seen_review_ids or key not in analysis_by_id:
            raise ValueError(f"Duplicate or unknown analysis review sentence: {key}")
        seen_review_ids.add(key)
        prediction = analysis_by_id[key]
        if r["text"] != prediction["text"] or int(r["line_number"]) != prediction["line_number"]:
            raise ValueError(f"Analysis review reference no longer matches the text: {key}")
        label = r["human_label"].strip()
        included = label in LABELS
        review_predictions.append({**r, "predicted_label": prediction["predicted_label"],
                                   "prob_positive": prediction["prob_positive"],
                                   "included_in_binary_evaluation": int(included),
                                   "correct": int(label == prediction["predicted_label"]) if included else "",
                                   "top_contributions": prediction["top_contributions"]})
    write_csv(OUT / "analysis_review_predictions.csv", review_predictions)
    gold = {r["sentence_id"]: r["human_label"].strip() for r in reviewed if r["human_label"].strip() in LABELS}
    scored = [r for r in analysis if str(r["sentence_id"]) in gold]
    target_eval = evaluate([gold[str(r["sentence_id"])] for r in scored], [r["predicted_label"] for r in scored]) if scored else None
    if target_eval:
        target_eval["accuracy_wilson_95_interval"] = wilson(sum(gold[str(r["sentence_id"])] == r["predicted_label"] for r in scored), len(scored))
        baseline_prediction = dummy.predict(np.zeros((len(scored),1)))
        baseline = evaluate([gold[str(r["sentence_id"])] for r in scored], baseline_prediction)
        target_eval["training_majority_baseline"] = {
            "predicted_label": str(baseline_prediction[0]), "accuracy": baseline["accuracy"],
            "macro_f1": baseline["classification_report"]["macro avg"]["f1-score"]}
    reviewed_labels = Counter(r["human_label"].strip() for r in reviewed if r["human_label"].strip())
    summary = {"sentence_count": len(analysis), "predicted_label_counts": dict(Counter(r["predicted_label"] for r in analysis)),
               "mean_prob_positive": float(np.mean([r["prob_positive"] for r in analysis])),
               "low_confidence_count": sum(r["low_confidence"] for r in analysis),
               "zero_feature_count": sum(r["nonzero_features"] == 0 for r in analysis),
               "short_sentence_count": sum(r["short_sentence"] for r in analysis),
               "suspected_text_issue_count": sum(r["suspected_text_issue"] for r in analysis),
               "analysis_file_sha256": sha256(path), "analysis_reference_evaluation": target_eval,
               "analysis_reference_scored_n": len(scored),
               "analysis_review_label_counts": dict(reviewed_labels),
               "analysis_review_completed_n": sum(reviewed_labels.values()),
               "analysis_binary_fraction_of_reviewed": len(scored)/sum(reviewed_labels.values()) if reviewed_labels else None,
               "warning": "Binary predictions include neutral/mixed sentences. Reference accuracy applies only to reviewed positive/negative items; full-novel accuracy remains unknown. Scores are not calibrated emotional intensity."}
    (OUT / "analysis_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    metadata = {"random_seed": SEED, "split": "stratified 80/20 before fit", "total_n": len(rows),
                "class_counts": dict(Counter(r["label"] for r in rows)), "train_n": len(train), "test_n": len(test),
                "train_class_counts": dict(Counter(ty)), "test_class_counts": dict(Counter(vy)),
                "features": {"kind": "TF-IDF", "tokenizer": "jieba, HMM=False", "ngram_range": [1,2], "min_df": 2,
                             "max_features": 10000, "sublinear_tf": True, "actual_vocabulary_size": len(names), "stop_words": None},
                "classifier": {"type": "LogisticRegression", "C": 1.0, "penalty": "l2", "solver": "liblinear", "max_iter": 2000},
                "reference_labels": "AI-initialized; student-reviewed items identified by annotation_method; review coverage tracked separately", "human_review": status,
                "exact_train_test_duplicates": 0, "near_train_test_duplicates": len(leakage), "analysis_training_exact_overlaps": len(overlap),
                "cross_work_cv_scope": "training 80% only, no test or analysis text used for feature fitting or model selection",
                "hyperparameters_selected_before_first_test_evaluation": True,
                "data_sha256": sha256(DATA / "labeled_sentences.csv"),
                "python_version": sys.version.split()[0],
                "package_versions": {p: importlib.metadata.version(p) for p in ["scikit-learn","jieba","numpy","scipy","matplotlib","joblib"]}}
    metrics["experiment"] = metadata
    (OUT / "metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    plot_results(metrics, analysis, bins)
    joblib.dump(model, OUT / "model.joblib")
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["classification_report"]["macro avg"]["f1-score"],
                      "confusion_matrix": metrics["confusion_matrix"], "training_n": len(train), "test_n": len(test),
                      "analysis_sentences": len(analysis), "human_review": status},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
