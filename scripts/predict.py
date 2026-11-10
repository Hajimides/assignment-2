"""Apply the saved training-only pipeline to another local UTF-8 text."""
import argparse
from pathlib import Path
import joblib
from prepare_data import ROOT, sentences, write_csv
from run_experiment import predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text_file", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "output/private/new_predictions.csv")
    args = parser.parse_args()
    # Only load the model you generated yourself with run_experiment.py.
    model = joblib.load(ROOT / "output/model.joblib")
    rows = list(sentences(args.text_file.read_text(encoding="utf-8-sig")))
    if not rows:
        parser.error("No Chinese sentences found")
    write_csv(args.output, predictions(model, rows))
    print(f"Wrote {len(rows)} predictions to {args.output}")
