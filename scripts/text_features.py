"""Shared tokenizer so the persisted pipeline also loads from predict.py."""
import logging
from pathlib import Path
import re
import jieba

jieba.setLogLevel(logging.ERROR)
cache = Path(__file__).resolve().parents[1] / ".cache"
cache.mkdir(exist_ok=True)
jieba.dt.tmp_dir = str(cache)


def tokenize(text):
    # Retain negators and one-character words; don't discard stop words.
    return [word for word in jieba.lcut(text, HMM=False)
            if re.search(r"[\u4e00-\u9fff]", word)]
