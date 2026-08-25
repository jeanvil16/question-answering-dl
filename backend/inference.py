"""Backend inference engine.

Loads the trained CNN+BiLSTM+Attention model once at startup, then answers
new questions by predicting answer span start/end positions and extracting
the corresponding substring from the raw context text.
"""

import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "saved_models"

from model.qa_model import ExtractiveQAModel, QAModelConfig, decode_best_span
from preprocessing.vocabulary import Vocabulary
from preprocessing.tokenizer import tokenize_with_offsets


class QAEngine:
    """Stateful wrapper around a loaded checkpoint."""

    def __init__(self, model_dir=SAVED):
        self.loaded = False
        self.model = None
        self.vocab = None
        self.config = None
        self.device = None
        self.best_metrics = {}

        # ---- guard: no model yet -----------------------------------------
        config_path = Path(model_dir) / "model_config.json"
        vocab_path = Path(model_dir) / "vocab.json"
        weights_path = Path(model_dir) / "qa_model.pt"
        metrics_path = Path(model_dir) / "metrics.json"

        if not all(p.exists() for p in [config_path, vocab_path, weights_path]):
            return  # gracefully left as not-loaded

        self.config = QAModelConfig.load(config_path)
        self.vocab = Vocabulary.load(vocab_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ExtractiveQAModel(self.config).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device,
                                weights_only=True)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.model.load_state_dict(state_dict)
        self.model.eval()

        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.best_metrics = json.load(f)
        self.loaded = True

    # ---------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, context, question):
        if not self.loaded:
            raise RuntimeError(
                "No trained model found. Run  python training/train.py  first."
            )

        t0 = time.perf_counter()

        # ---- tokenize ---------------------------------------------------
        q_tokens = tokenize_with_offsets(question)[:self.config.max_question_len]
        c_tokens = tokenize_with_offsets(context)[:self.config.max_context_len]
        q_ids = self.vocab.encode([t.text for t in q_tokens])
        c_ids = self.vocab.encode([t.text for t in c_tokens])

        q_len = len(q_ids)
        c_len = len(c_ids)

        # ---- tensorize (batch=1) ----------------------------------------
        q_ids_t = torch.tensor([q_ids], device=self.device)
        q_mask = torch.ones_like(q_ids_t, dtype=torch.bool)
        c_ids_t = torch.tensor([c_ids], device=self.device)
        c_mask = torch.ones_like(c_ids_t, dtype=torch.bool)

        # Pad to fixed config length if below (avoids shape mismatch on some
        # build configs). NOT longer than max — truncation already done above.
        pad_len_q = self.config.max_question_len - q_len
        pad_len_c = self.config.max_context_len - c_len
        if pad_len_q > 0:
            q_ids_t = torch.nn.functional.pad(q_ids_t, (0, pad_len_q))
            q_mask = torch.nn.functional.pad(q_mask, (0, pad_len_q))
        if pad_len_c > 0:
            c_ids_t = torch.nn.functional.pad(c_ids_t, (0, pad_len_c))
            c_mask = torch.nn.functional.pad(c_mask, (0, pad_len_c))

        # ---- forward pass ------------------------------------------------
        start_logits, end_logits = self.model(
            q_ids_t, q_mask, c_ids_t, c_mask
        )
        # Remove padding dims
        start_logits = start_logits[0, :c_len]
        end_logits = end_logits[0, :c_len]

        # ---- decode span -------------------------------------------------
        best_s, best_e = decode_best_span(
            start_logits.cpu(), end_logits.cpu(), self.config.max_answer_len
        )

        # ---- confidence: geometric mean of the softmax probs at s and e ---
        probs = torch.nn.functional.softmax(
            torch.stack([start_logits, end_logits]), dim=-1
        )
        start_prob = float(probs[0, best_s])
        end_prob = float(probs[1, best_e])
        confidence = (start_prob * end_prob) ** 0.5

        # ---- extract answer substring from original context ---------------
        if best_s < len(c_tokens) and best_e < len(c_tokens):
            ans_start_char = c_tokens[best_s].start
            ans_end_char = c_tokens[best_e].end
            answer = context[ans_start_char:ans_end_char]
        else:
            ans_start_char = ans_end_char = 0
            answer = ""

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "answer": answer,
            "confidence": round(confidence, 4),
            "start_prob": round(start_prob, 4),
            "end_prob": round(end_prob, 4),
            "start_char": ans_start_char,
            "end_char": ans_end_char,
            "start_token": best_s,
            "end_token": best_e,
            "latency_ms": latency_ms,
            "context_tokens": c_len,
            "question_tokens": q_len,
            "truncated": (c_len >= self.config.max_context_len
                          or q_len >= self.config.max_question_len),
        }


# ---- singleton created at import so Flask app can use it immediately -------
engine = None


def load_engine(model_dir=SAVED):
    global engine
    engine = QAEngine(model_dir)
    return engine
