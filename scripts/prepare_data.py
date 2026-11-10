"""Download pinned sources, split sentences, and materialize reviewed annotations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REVISION = "4f4f2b61b560d5daff3d920eb2dfa72a7a47d7b4"
SOURCES = ["鲁迅_呐喊", "老舍_骆驼祥子", "巴金_寒夜", "沈从文_边城", "钱钟书_围城", "茅盾_子夜"]
# Only candidate retrieval uses these hints; labels are separately reviewed by AI.
POS_HINTS = re.compile("高兴|快乐|欢喜|欢乐|愉快|幸福|欣慰|喜悦|兴奋|感激|感谢|满意|微笑|喜爱|热爱|安慰|温暖|希望|快活|喜欢|佩服|赞美")
NEG_HINTS = re.compile("悲哀|悲伤|伤心|痛苦|难过|恐惧|害怕|愤怒|失望|绝望|忧愁|忧虑|苦恼|寂寞|孤独|憎恨|讨厌|愤恨|羞耻|惭愧|烦恼|发愁|难受|哭|眼泪|惶恐|悲惨|惊慌|伤感|厌恶|可怕|可怜|发怒")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sentences(text):
    """Yield each nonempty sentence with original line and Python character offsets.

    Hard boundaries: 。！？!? or six-dot/Chinese ellipsis, and physical newlines.
    Consume closing quotes with the preceding sentence. Do not join across lines.
    """
    offset = 0
    n = 0
    pattern = re.compile(r"[^。！？!?…\n]+(?:[。！？!?]+|…{2,}|…|$)[’”\"』」]*")
    for line_no, raw in enumerate(text.splitlines(keepends=True), 1):
        for match in pattern.finditer(raw.rstrip("\r\n")):
            value = match.group().strip()
            if not re.search(r"[\u4e00-\u9fff]", value):
                continue
            start = offset + match.start() + len(match.group()) - len(match.group().lstrip())
            n += 1
            yield {"sentence_id": n, "line_number": line_no,
                   "char_start": start, "char_end": start + len(value), "text": value}
        offset += len(raw)


def download(name):
    path = DATA / "raw" / f"{name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/mcjkurz/qhchina-data/{REVISION}/" + urllib.parse.quote(f"corpora/modern/{name}.txt")
    if not path.exists():
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
                path.write_bytes(data)
                break
            except Exception:
                if attempt == 2:
                    raise
    return path, url


def candidates():
    rng = random.Random(3242)
    rows, manifest, supplemental = [], [], []
    seen = set()
    for source in SOURCES:
        path, url = download(source)
        manifest.append({"source": source, "url": url, "sha256": sha256(path), "role": "training_candidate_source"})
        items = list(sentences(path.read_text(encoding="utf-8-sig")))
        for bucket, hints in [("positive_hint", POS_HINTS), ("negative_hint", NEG_HINTS)]:
            pool = [s for s in items if 12 <= len(s["text"]) <= 90 and hints.search(s["text"])]
            rng.shuffle(pool)
            if bucket == "positive_hint":
                supplemental.append((source, pool))
            selected = 0
            for s in pool:
                key = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", s["text"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"candidate_id": f"C{len(rows)+1:04d}", "source": source, **s,
                             "retrieval_bucket": bucket})
                selected += 1
                if selected == 50:
                    break
            if selected < 50:
                print(f"Using all {selected} available candidates: {source}, {bucket}")
    # The first annotation pass found too few clear positive sentences. Before
    # splitting/fitting, append at most 25 additional positive-hint candidates
    # per work, then review their actual sentiment separately as well.
    for source, pool in supplemental:
        selected = 0
        for s in pool:
            key = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", s["text"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({"candidate_id": f"C{len(rows)+1:04d}", "source": source, **s,
                         "retrieval_bucket": "supplemental_positive_hint"})
            selected += 1
            if selected == 25:
                break
    write_csv(DATA / "candidates.csv", rows)
    (DATA / "sources.json").write_text(json.dumps({"repository": "https://github.com/mcjkurz/qhchina-data", "revision": REVISION, "files": manifest}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Prepared {len(rows)} candidates (retrieval hints are not gold labels).")


def materialize():
    items = {r["candidate_id"]: r for r in read_csv(DATA / "candidates.csv")}
    annotations = read_csv(DATA / "annotation_decisions.csv")
    if len({a['candidate_id'] for a in annotations}) != len(annotations):
        raise ValueError("Repeated annotation IDs")
    if set(items) != {a['candidate_id'] for a in annotations}:
        raise ValueError("Every candidate needs an explicit decision")
    overrides = {}
    review_path = DATA / "human_review.csv"
    if review_path.exists():
        for r in read_csv(review_path):
            label = r["human_label"].strip()
            if label:
                if label not in {"positive", "negative", "exclude"}:
                    raise ValueError(f"Invalid human label: {label}")
                overrides[r["candidate_id"]] = label
    rows = []
    for a in annotations:
        label = overrides.get(a["candidate_id"], a["label"])
        if label not in {"positive", "negative", "exclude"}:
            raise ValueError(f"Invalid label: {label}")
        if label == "exclude":
            continue
        s = items[a["candidate_id"]]
        rows.append({"item_id": s["candidate_id"], "text": s["text"], "label": label,
                     "source": s["source"], "source_sentence_id": s["sentence_id"],
                     "line_number": s["line_number"], "char_start": s["char_start"], "char_end": s["char_end"],
                     "annotation_method": "human_reviewed" if a["candidate_id"] in overrides else "AI_reviewed",
                     "annotation_note": a["reason"]})
    counts = {label: sum(r['label'] == label for r in rows) for label in ["negative", "positive"]}
    if min(counts.values()) < 200:
        raise ValueError(f"At least 200 per class needed: {counts}")
    write_csv(DATA / "labeled_sentences.csv", rows)
    print(f"Labeled items: {counts}; human decisions: {len(overrides)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", action="store_true", help="download sources and regenerate candidate pool")
    args = p.parse_args()
    candidates() if args.candidates else materialize()
