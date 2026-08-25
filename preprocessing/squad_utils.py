"""SQuAD / SQuAD-style preprocessing.

Converts raw SQuAD-format JSON into tokenized training records:

    context + question + answer(char offsets)
        -> question tokens
        -> context tokens (possibly windowed so the answer fits)
        -> start/end TOKEN positions of the answer  <- what the model predicts

Answer-span mapping
-------------------
SQuAD gives the answer as `answer_start` (a CHARACTER offset) plus the answer
text. We locate the first and last tokens that overlap the character range
[answer_start, answer_start + len(answer)) using the offsets produced by
`tokenize_with_offsets`. This is the classic "char-to-token alignment" step of
extractive QA preprocessing.

Long contexts are handled with a sliding window anchored on the answer: we
keep the last `max_context_len` tokens ending at or after the answer whenever
the answer would be cut off by simple truncation.
"""

import bisect
import json
import os

from .tokenizer import tokenize, tokenize_with_offsets


def load_squad_records(path):
    """Flatten a SQuAD v1.1 file into (title, context, question, answer_text,
    answer_start) tuples."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for article in data["data"]:
        title = article.get("title", "unknown")
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                if not qa.get("answers"):
                    continue  # SQuAD v2 style "no answer" questions are skipped
                ans = qa["answers"][0]
                records.append(
                    {
                        "title": title,
                        "context": context,
                        "question": qa["question"],
                        "answer_text": ans["text"],
                        "answer_start": int(ans["answer_start"]),
                        # Optional: examples sharing a "group" (e.g. augmented
                        # paraphrases of one question) must stay in the SAME
                        # train/val split to avoid data leakage.
                        "group": qa.get("group"),
                    }
                )
    return records


def _token_index_at(starts, ends, char_pos):
    """Index of the token containing char position `char_pos`.

    `starts`/`ends` are the sorted character-offset arrays of the context
    tokens. If the position falls between tokens (e.g. inside a space), the
    next token starting at or after it is used instead.
    """
    idx = bisect.bisect_right(starts, char_pos) - 1
    if idx < 0:
        idx = 0
    if idx < len(starts) and ends[idx] <= char_pos:
        idx += 1  # position was after this token ended -> advance
    return min(idx, len(starts) - 1)


def _window_start(answer_tok_idx, num_tokens, max_len):
    """Choose a context window of length `max_len` that contains the answer.

    Simple right truncation would silently delete answers that appear late in
    a paragraph (very common in full SQuAD). Instead the window slides back
    just enough to cover the answer, keeping a little left-side context.
    """
    if num_tokens <= max_len:
        return 0
    ideal = answer_tok_idx - max(4, max_len // 6)
    return max(0, min(ideal, num_tokens - max_len))


def build_training_records(records, max_context_len=160, max_question_len=24):
    """Convert raw records into model-ready tokenized examples.

    Returns (examples, stats). Each example is a dict:
        question_tokens : list[str]           (truncated to max_question_len)
        context_tokens  : list[str]           (windowed to max_context_len)
        start_index     : int                 answer start TOKEN index
        end_index       : int                 answer end TOKEN index (inclusive)
        answer_tokens   : list[str]           gold answer tokens (for EM/F1)
    """
    examples = []
    stats = {"total": len(records), "kept": 0, "skipped_answer_mismatch": 0,
             "skipped_too_long": 0}

    for rec in records:
        q_tokens = tokenize(rec["question"])[:max_question_len]
        if not q_tokens:
            continue

        ctx_tokens_with_pos = tokenize_with_offsets(rec["context"])
        starts = [t.start for t in ctx_tokens_with_pos]
        ends = [t.end for t in ctx_tokens_with_pos]
        c_tokens_all = [t.text for t in ctx_tokens_with_pos]
        if not c_tokens_all:
            continue

        ans_char_start = rec["answer_start"]
        ans_char_end = ans_char_start + len(rec["answer_text"])

        # ---- char -> token alignment ------------------------------------
        s_tok = _token_index_at(starts, ends, ans_char_start)
        e_tok = _token_index_at(starts, ends, max(ans_char_end - 1, ans_char_start))

        # Sanity check: the reconstructed span should match the gold answer
        # (guards against tokenizer quirks / misaligned SQuAD files).
        rebuilt = "".join(c_tokens_all[s_tok:e_tok + 1])
        gold = "".join(tokenize(rec["answer_text"]))
        if rebuilt != gold:
            stats["skipped_answer_mismatch"] += 1
            continue

        # ---- windowing --------------------------------------------------
        w = _window_start(s_tok, len(c_tokens_all), max_context_len)
        ctx_tokens = c_tokens_all[w:w + max_context_len]
        s_idx, e_idx = s_tok - w, e_tok - w
        if e_idx >= max_context_len:  # answer longer than the window itself
            stats["skipped_too_long"] += 1
            continue

        examples.append(
            {
                "title": rec["title"],
                "question_tokens": q_tokens,
                "context_tokens": ctx_tokens,
                "start_index": int(s_idx),
                "end_index": int(e_idx),
                "answer_tokens": c_tokens_all[s_tok:e_tok + 1],
                **({"group": rec["group"]} if rec.get("group") else {}),
            }
        )
        stats["kept"] += 1

    return examples, stats


def _source_fingerprint(path):
    """(mtime_ns, size) of the source file, used to invalidate stale caches."""
    st = os.stat(path)
    return [st.st_mtime_ns, st.st_size]


def load_preprocessed(path, cache_path=None, rebuild_cache=False,
                      max_context_len=160, max_question_len=24):
    """Load + tokenize a SQuAD file, optionally caching the result.

    Tokenizing full SQuAD (~90k answers) takes a minute or two, so the result
    is cached to `cache_path` when provided. The cache automatically
    invalidates when the source file changes or when `max_*_len` differ;
    `rebuild_cache=True` forces a rebuild regardless.
    """
    fingerprint = _source_fingerprint(path)
    if cache_path and os.path.exists(cache_path) and not rebuild_cache:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if (payload.get("fingerprint") == fingerprint
                    and payload.get("max_context_len") == max_context_len
                    and payload.get("max_question_len") == max_question_len):
                return payload["examples"], payload["stats"]
        except (json.JSONDecodeError, KeyError):
            pass  # corrupted cache -> rebuild below

    records = load_squad_records(path)
    examples, stats = build_training_records(records, max_context_len,
                                             max_question_len)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"examples": examples, "stats": stats,
                       "fingerprint": fingerprint,
                       "max_context_len": max_context_len,
                       "max_question_len": max_question_len},
                      f, ensure_ascii=False)
    return examples, stats
