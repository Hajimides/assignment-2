"""Create a self-contained result explorer; full novel text is never embedded."""
import json
from pathlib import Path
from prepare_data import ROOT, read_csv


def main():
    out = ROOT / "output"
    payload = {
        "metrics": json.loads((out / "metrics.json").read_text()),
        "summary": json.loads((out / "analysis_summary.json").read_text()),
        "test": read_csv(out / "test_predictions.csv"),
        "novel": read_csv(out / "huozhe_predictions.csv"),
        "bins": read_csv(out / "huozhe_position_bins.csv"),
        "weights": read_csv(out / "feature_weights.csv"),
        "comparisons": read_csv(out / "model_comparison.csv"),
        "training_review": read_csv(ROOT / "data/human_review.csv"),
        "analysis_review": read_csv(ROOT / "data/analysis_human_review.csv"),
    }
    template = (ROOT / "scripts/explorer_template.html").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    (out / "explorer.html").write_text(template.replace("__EXPERIMENT_DATA__", data), encoding="utf-8")
    print("Created output/explorer.html (offline, no external libraries, no embedded full novel)")


if __name__ == "__main__":
    main()
