"""Check methodological invariants and independently recompute reported scores."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from prepare_data import ROOT, DATA, read_csv, sentences, sha256
from text_features import tokenize


def main():
    out = ROOT / "output"
    data = read_csv(DATA / "labeled_sentences.csv")
    train = read_csv(out / "train_split.csv")
    test = read_csv(out / "test_split.csv")
    predicted = read_csv(out / "test_predictions.csv")
    metrics = json.loads((out / "metrics.json").read_text())
    ids = lambda rows: {r["item_id"] for r in rows}
    assert ids(train).isdisjoint(ids(test)), "Training/test overlap"
    assert ids(train) | ids(test) == ids(data), "Split does not cover dataset"
    assert len(test) == int(np.ceil(.2 * len(data))), "Unexpected test size"
    assert min(Counter(r['label'] for r in data).values()) >= 200
    assert metrics['experiment']['data_sha256'] == sha256(DATA / "labeled_sentences.csv")
    for label in ['negative', 'positive']:
        assert abs(sum(r['label'] == label for r in test)/len(test)
                   -sum(r['label'] == label for r in data)/len(data)) <= 1/len(test)
    labels = ['negative', 'positive']
    cm = [[sum(r['label'] == a and r['predicted_label'] == b for r in predicted)
           for b in labels] for a in labels]
    assert cm == metrics['confusion_matrix']
    assert np.isclose((cm[0][0]+cm[1][1])/len(test), metrics['accuracy'])
    for i,label in enumerate(labels):
        tp = cm[i][i]
        precision = tp / max(1, sum(cm[j][i] for j in range(2)))
        recall = tp / max(1, sum(cm[i]))
        f1 = 2*precision*recall / max(1e-15, precision+recall)
        for key,value in [('precision',precision),('recall',recall),('f1-score',f1)]:
            assert np.isclose(metrics['classification_report'][label][key],value)
    assert len(read_csv(out / "misclassified.csv")) == len(test)-sum(cm[i][i] for i in range(2))
    assert not read_csv(out / "near_duplicate_audit.csv")
    model = joblib.load(out / "model.joblib")
    probs = model.predict_proba([r['text'] for r in predicted])[:,list(model.classes_).index('positive')]
    assert np.allclose(probs,[float(r['prob_positive']) for r in predicted],atol=1e-8)
    assert list(model.predict([r['text'] for r in predicted])) == [r['predicted_label'] for r in predicted]
    # Check every IDF against document frequencies from TRAINING ONLY.
    vectorizer = model['features']
    matrix = vectorizer.transform([r['text'] for r in train])
    df = (matrix > 0).sum(axis=0).A1
    expected_idf = np.log((1+len(train))/(1+df))+1
    assert np.allclose(vectorizer.idf_, expected_idf), "IDF not consistent with training split"
    assert '不' in tokenize('不高兴'), "Negation was discarded"
    sample = '她说：“真好！”\n他却哭了……后来笑了。'
    split = list(sentences(sample))
    assert [r['text'] for r in split] == ['她说：“真好！”','他却哭了……','后来笑了。']
    assert all(sample[r['char_start']:r['char_end']] == r['text'] for r in split)
    manifest = json.loads((DATA / "sources.json").read_text())
    source_text = {}
    for entry in manifest['files']:
        path = DATA / 'raw' / (entry['source']+'.txt')
        if path.exists():
            assert sha256(path) == entry['sha256']
            source_text[entry['source']] = path.read_text(encoding='utf-8-sig')
    for r in data:
        if r['source'] in source_text:
            assert source_text[r['source']][int(r['char_start']):int(r['char_end'])] == r['text']
    novel = read_csv(out / "huozhe_predictions.csv")
    local_novel = read_csv(out / "private/huozhe_full_predictions.csv")
    summary = json.loads((out / "analysis_summary.json").read_text())
    assert len(novel) == len(local_novel) == summary['sentence_count']
    assert all('text' not in r for r in novel), "Public output contains full novel text"
    for r in novel + predicted:
        p,q = float(r['prob_positive']),float(r['prob_negative'])
        assert 0 <= p <= 1 and 0 <= q <= 1 and np.isclose(p+q,1)
        assert r['predicted_label'] == ('positive' if p > .5 else 'negative')
    assert summary['predicted_label_counts'] == dict(Counter(r['predicted_label'] for r in novel))
    assert summary['low_confidence_count'] == sum(int(r['low_confidence']) for r in novel)
    assert all(not r['human_label'] or r['human_label'] in ['positive','negative','exclude']
               for r in read_csv(DATA / "human_review.csv"))
    reviewed = read_csv(DATA / "analysis_human_review.csv")
    review_predictions = read_csv(out / "analysis_review_predictions.csv")
    assert {r['sentence_id'] for r in reviewed} == {r['sentence_id'] for r in review_predictions}
    gold = {r['sentence_id']: r for r in reviewed}
    for r in review_predictions:
        assert r['human_label'] == gold[r['sentence_id']]['human_label']
        assert r['text'] == gold[r['sentence_id']]['text']
        included = r['human_label'] in labels
        assert int(r['included_in_binary_evaluation']) == int(included)
        assert r['correct'] == str(int(r['human_label'] == r['predicted_label'])) if included else r['correct'] == ''
    binary = [r for r in review_predictions if int(r['included_in_binary_evaluation'])]
    assert len(binary) == summary['analysis_reference_scored_n']
    assert dict(Counter(r['human_label'] for r in reviewed if r['human_label'])) == summary['analysis_review_label_counts']
    if binary:
        accuracy = sum(int(r['correct']) for r in binary)/len(binary)
        assert np.isclose(accuracy, summary['analysis_reference_evaluation']['accuracy'])
        cm_review = [[sum(r['human_label'] == a and r['predicted_label'] == b for r in binary)
                      for b in labels] for a in labels]
        assert cm_review == summary['analysis_reference_evaluation']['confusion_matrix']
        baseline = summary['analysis_reference_evaluation']['training_majority_baseline']
        baseline_accuracy = sum(r['human_label'] == baseline['predicted_label'] for r in binary)/len(binary)
        assert np.isclose(baseline_accuracy, baseline['accuracy'])
    for path in ['metrics.json','analysis_summary.json','explorer.html','confusion_matrix.png','huozhe_sentiment_curve.png']:
        assert (out / path).stat().st_size > 0
    print('PASS: split balance and coverage; no overlap; independently recalculated metrics;')
    print('      saved-model predictions; training-only IDF; sentence boundaries and offsets;')
    print('      source checksums and quotations; probabilities; no full novel in public CSV.')
    print('      human review alignment; binary-only reference evaluation and denominator.')


if __name__ == '__main__':
    main()
