"""Training script for the extractive QA model.

Runs the full supervised pipeline:

    1. load SQuAD(-style) JSON            (full dataset or built-in sample)
    2. tokenize + align answer spans      (preprocessing package, with cache)
    3. build the word vocabulary          -> embeddings can be looked up
    4. create PyTorch Dataset / loaders   (dynamic padding per batch)
    5. train with joint start/end cross-entropy
    6. validate every epoch               (loss + SQuAD EM/F1)
    7. save best checkpoint + vocab + config + training history

Quick demo on any laptop:
    python training/train.py                       # built-in sample dataset
Full SQuAD v1.1 (download first via dataset/download_squad.py):
    python training/train.py --train-file dataset/train-v1.1.json \
        --val-file dataset/dev-v1.1.json --epochs 3 --batch-size 64 \
        --max-context-len 320 --max-question-len 32 --learning-rate 1e-3
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.qa_model import ExtractiveQAModel, QAModelConfig, decode_best_span, count_parameters
from preprocessing.squad_utils import load_preprocessed
from preprocessing.vocabulary import Vocabulary
from training.metrics import span_metrics

DEFAULT_SAMPLE = ROOT / "dataset" / "sample_squad.json"
ARTIFACTS = ROOT / "saved_models"


# --------------------------------------------------------------------------- #
#  Data plumbing
# --------------------------------------------------------------------------- #
class QADataset(Dataset):
    """Tensor wrapper around preprocessed token records."""

    def __init__(self, examples, vocab):
        self.examples = examples
        self.vocab = vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "question_ids": self.vocab.encode(ex["question_tokens"]),
            "context_ids": self.vocab.encode(ex["context_tokens"]),
            "start_index": ex["start_index"],
            "end_index": ex["end_index"],
        }


def collate(batch):
    """Dynamically pad a batch to its own longest sequences (saves compute)."""
    q_max = max(len(b["question_ids"]) for b in batch)
    c_max = max(len(b["context_ids"]) for b in batch)
    pad = 0

    def pad_to(ids, length):
        return ids + [pad] * (length - len(ids))

    questions = torch.tensor([pad_to(b["question_ids"], q_max) for b in batch])
    contexts = torch.tensor([pad_to(b["context_ids"], c_max) for b in batch])
    return {
        "question_ids": questions,
        "question_mask": questions != pad,
        "context_ids": contexts,
        "context_mask": contexts != pad,
        "start_index": torch.tensor([b["start_index"] for b in batch]),
        "end_index": torch.tensor([b["end_index"] for b in batch]),
    }


def make_loaders(train_examples, val_examples, vocab, batch_size):
    train_loader = DataLoader(QADataset(train_examples, vocab),
                              batch_size=batch_size, shuffle=True,
                              collate_fn=collate)
    val_loader = DataLoader(QADataset(val_examples, vocab),
                            batch_size=batch_size, shuffle=False,
                            collate_fn=collate)
    # Ordered view of the TRAINING set: used to measure training EM/F1.
    # (The shuffled loader would mis-align predictions with examples.)
    train_eval_loader = DataLoader(QADataset(train_examples, vocab),
                                   batch_size=batch_size, shuffle=False,
                                   collate_fn=collate)
    return train_loader, train_eval_loader, val_loader


# --------------------------------------------------------------------------- #
#  Train / evaluate epochs
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, device, optimizer=None):
    """One pass over `loader`. If optimizer is given this is training."""
    training = optimizer is not None
    model.train() if training else model.eval()
    loss_ce = torch.nn.CrossEntropyLoss()
    total_loss, batches = 0.0, 0
    all_spans = []

    with torch.set_grad_enabled(training):
        for batch in loader:
            question_ids = batch["question_ids"].to(device)
            question_mask = batch["question_mask"].to(device)
            context_ids = batch["context_ids"].to(device)
            context_mask = batch["context_mask"].to(device)
            start_idx = batch["start_index"].to(device)
            end_idx = batch["end_index"].to(device)

            start_logits, end_logits = model(question_ids, question_mask,
                                             context_ids, context_mask)

            # Joint loss: the model must localise BOTH boundaries correctly.
            loss = loss_ce(start_logits, start_idx) + loss_ce(end_logits, end_idx)

            if training:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=5.0)  # stability
                optimizer.step()

            total_loss += loss.item()
            batches += 1

            if not training:
                for s_logit, e_logit in zip(start_logits, end_logits):
                    span = decode_best_span(s_logit.cpu(), e_logit.cpu(),
                                            model.config.max_answer_len)
                    all_spans.append(span)

    mean_loss = total_loss / max(1, batches)
    return mean_loss, all_spans


# --------------------------------------------------------------------------- #
#  Persistence
# --------------------------------------------------------------------------- #
def save_artifacts(model, vocab, history, best_scores, args, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Refresh runtime fields of the config before saving.
    cfg = model.config
    cfg.dataset_name = args.train_file.stem
    cfg.save(out_dir / "model_config.json")
    vocab.save(out_dir / "vocab.json")
    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_scores, f, indent=2)
    torch.save(model.state_dict(), out_dir / "qa_model.pt")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Train the CNN+BiLSTM QA model")
    p.add_argument("--train-file", type=Path, default=DEFAULT_SAMPLE,
                   help="SQuAD-format JSON used for training")
    p.add_argument("--val-file", type=Path, default=None,
                   help="Optional separate validation file; otherwise a "
                        "random split of the training data is used")
    p.add_argument("--sample-size", type=int, default=0,
                   help="Use only the first N examples (0 = all). Quick mode.")
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=96)
    p.add_argument("--cnn-num-filters", type=int, default=96)
    p.add_argument("--cnn-kernels", type=str, default="2,3,4",
                   help="Comma-separated CNN kernel sizes (n-gram widths)")
    p.add_argument("--cnn-out-dim", type=int, default=128)
    p.add_argument("--rnn-hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--max-question-len", type=int, default=24)
    p.add_argument("--max-context-len", type=int, default=160)
    p.add_argument("--max-answer-len", type=int, default=40)
    p.add_argument("--min-token-freq", type=int, default=1,
                   help="Raise to 2+ when training on full SQuAD")
    p.add_argument("--patience", type=int, default=12,
                   help="Early-stopping patience on validation F1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None,
                   help="'cpu', 'cuda' or empty for auto")
    p.add_argument("--rebuild-cache", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 64)
    print(" Question Answering System - training (CNN + BiLSTM + Attention)")
    print("=" * 64)

    # ---- 1/2. load + preprocess -----------------------------------------
    cache_train = ROOT / "dataset" / (
        f"processed_{args.train_file.stem}_c{args.max_context_len}"
        f"_q{args.max_question_len}.json")

    print(f"[data] tokenizing {args.train_file.name} ...")
    examples, stats = load_preprocessed(
        args.train_file, cache_path=str(cache_train),
        rebuild_cache=args.rebuild_cache,
        max_context_len=args.max_context_len,
        max_question_len=args.max_question_len)
    if args.sample_size > 0:
        examples = examples[:args.sample_size]
    print(f"[data] kept {stats['kept']}/{stats['total']} answers "
          f"(skipped {stats['skipped_answer_mismatch']} mismatched, "
          f"{stats['skipped_too_long']} too long)")

    # ---- train/validation split ------------------------------------------
    if args.val_file and args.val_file.exists():
        cache_val = ROOT / "dataset" / (
            f"processed_{args.val_file.stem}_c{args.max_context_len}"
            f"_q{args.max_question_len}.json")
        val_examples, _ = load_preprocessed(
            args.val_file, cache_path=str(cache_val),
            rebuild_cache=args.rebuild_cache,
            max_context_len=args.max_context_len,
            max_question_len=args.max_question_len)
    elif examples and "group" in examples[0]:
        # Augmented datasets: ALL variants of one question stay on the same
        # side of the split (otherwise paraphrases would leak into val).
        groups = sorted({ex["group"] for ex in examples})
        random.Random(args.seed).shuffle(groups)
        n_val = max(1, int(len(groups) * args.val_split))
        val_groups = set(groups[:n_val])
        val_examples = [ex for ex in examples if ex["group"] in val_groups]
        examples = [ex for ex in examples if ex["group"] not in val_groups]
    else:
        shuffled = examples[:]
        random.Random(args.seed).shuffle(shuffled)
        n_val = max(2, int(len(shuffled) * args.val_split))
        val_examples = shuffled[:n_val]
        examples = shuffled[n_val:]
    print(f"[split] train={len(examples)}  val={len(val_examples)}")

    # ---- 3. vocabulary ----------------------------------------------------
    vocab = Vocabulary.build(
        [ex["question_tokens"] for ex in examples]
        + [ex["context_tokens"] for ex in examples],
        min_freq=args.min_token_freq)
    print(f"[vocab] size={len(vocab)} (min_freq={args.min_token_freq})")

    train_loader, train_eval_loader, val_loader = make_loaders(
        examples, val_examples, vocab, args.batch_size)

    # ---- 4. model ----------------------------------------------------------
    kernels = [int(k) for k in args.cnn_kernels.split(",")]
    config = QAModelConfig(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        cnn_kernel_sizes=kernels,
        cnn_num_filters=args.cnn_num_filters,
        cnn_out_dim=args.cnn_out_dim,
        rnn_hidden_dim=args.rnn_hidden_dim,
        dropout=args.dropout,
        max_question_len=args.max_question_len,
        max_context_len=args.max_context_len,
        max_answer_len=args.max_answer_len,
    )
    model = ExtractiveQAModel(config).to(device)
    print(f"[model] trainable parameters: {count_parameters(model):,}")
    print(f"[model] device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # ---- 5/6. training loop -----------------------------------------------
    history, best_f1, bad_epochs = [], -1.0, 0
    ckpt_path = ARTIFACTS / "_best_state.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss, _ = run_epoch(model, train_loader, device,
                                  optimizer=optimizer)
        # Evaluate on the training set too: shows whether the model is even
        # fitting the data (useful when the dataset is very small).
        _, train_spans = run_epoch(model, train_eval_loader, device)
        train_metrics = span_metrics(train_spans, examples)
        val_loss, spans = run_epoch(model, val_loader, device)
        metrics = span_metrics(spans, val_examples)
        dt = time.perf_counter() - t0

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_em": round(train_metrics["em"], 4),
            "train_f1": round(train_metrics["f1"], 4),
            "val_em": round(metrics["em"], 4),
            "val_f1": round(metrics["f1"], 4),
            "seconds": round(dt, 2),
        })
        marker = ""
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            bad_epochs = 0
            best_scores = {"epoch": epoch, **history[-1]}
            torch.save({"state_dict": model.state_dict(),
                        "config": config.to_dict()}, ckpt_path)
            marker = "  <- best"
        else:
            bad_epochs += 1

        print(f"[epoch {epoch:3d}] train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  train_EM={train_metrics['em']:.3f}  "
              f"EM={metrics['em']:.3f}  F1={metrics['f1']:.3f}  ({dt:.1f}s){marker}")

        with open(ARTIFACTS / "training_history.json", "w",
                  encoding="utf-8") as f:  # persist incrementally
            json.dump(history, f, indent=2)

        if bad_epochs >= args.patience:
            print(f"[early-stop] no F1 improvement for {args.patience} epochs")
            break

    # ---- 7. save best checkpoint ------------------------------------------
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    save_artifacts(model, vocab, history, best_scores, args, ARTIFACTS)
    ckpt_path.unlink(missing_ok=True)

    print("-" * 64)
    print(f"[done] best epoch={best_scores['epoch']}  EM={best_scores['val_em']:.3f}"
          f"  F1={best_scores['val_f1']:.3f}")
    print(f"[done] artifacts saved to {ARTIFACTS}")
    print("       start the backend with:  python backend/app.py")


if __name__ == "__main__":
    main()
