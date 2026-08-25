"""Evaluation metrics for extractive QA.

We use the two official SQuAD v1.1 metrics:

* Exact Match (EM)  - 1 if the normalized predicted span equals the
  normalized gold span, else 0.
* Token F1          - harmonic mean of precision/recall over the token bags
  of prediction and gold answer (rewards partial overlap).

Normalization (lowercase, strip punctuation & articles) makes both metrics
robust to trivial differences like "The Taj Mahal" vs "taj mahal".
"""

import re
import string


def normalize_answer(text_tokens):
    text = " ".join(text_tokens).lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred_tokens, gold_tokens):
    return float(normalize_answer(pred_tokens) == normalize_answer(gold_tokens))


def token_f1(pred_tokens, gold_tokens):
    pred = normalize_answer(pred_tokens).split()
    gold = normalize_answer(gold_tokens).split()
    if not pred or not gold:
        return float(pred == gold)
    common = set(pred) & set(gold)
    if not common:
        return 0.0
    precision = len(common) / len(pred)
    recall = len(common) / len(gold)
    return 2 * precision * recall / (precision + recall)


def span_metrics(predicted_spans, examples):
    """Aggregate EM / F1 over a validation set.

    predicted_spans: list of (start_index, end_index) token spans.
    examples:        list of dicts containing 'context_tokens' and
                     'answer_tokens' produced by preprocessing.
    """
    em_total, f1_total = 0.0, 0.0
    for (s, e), ex in zip(predicted_spans, examples):
        pred_tokens = ex["context_tokens"][s:e + 1]
        em_total += exact_match(pred_tokens, ex["answer_tokens"])
        f1_total += token_f1(pred_tokens, ex["answer_tokens"])
    n = max(1, len(examples))
    return {"em": em_total / n, "f1": f1_total / n}
