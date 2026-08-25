"""Extractive Question Answering model.

Pipeline (matches the architecture documented in the README):

    question / context text
        -> Tokenization                (preprocessing package)
        -> Word Embeddings             nn.Embedding, shared for q + context
        -> CNN feature extraction      multi-kernel Conv1d (n-gram detector)
        -> BiLSTM                      bidirectional sequence modelling
        -> Attention feature fusion    BiDAF-style question<->context attention
        -> Start head                  Linear -> logits over context tokens
        -> End head                    Linear -> logits over context tokens
        -> Answer extraction           argmax over valid (start, end) spans

WHY A CNN?  (the Deep-Learning-Techniques requirement)
------------------------------------------------------
The convolutional layers slide learnable filters of size 2/3/4 across the
embedding sequence. Each filter detects a LOCAL pattern of neighbouring
tokens - e.g. "new zealand", "world heritage", "carbon dioxide" - and fires
wherever it occurs, giving the network translation-invariant n-gram features.
Because three kernel sizes run in parallel, the model composes unigrams,
bigrams and trigrams into a richer local feature map BEFORE any sequential
processing happens.

WHY A BiLSTM?
-------------
Convolutions are windowed: they cannot relate "it" back to a noun mentioned
20 tokens earlier. The BiLSTM reads the CNN feature map left-to-right AND
right-to-left, so every position's representation summarises both its past
and its future - exactly what is needed to decide where an answer span
starts or ends.

WHY ATTENTION?
--------------
Not all context words matter equally for a given question. The attention
layer compares every context state with every question state and builds a
question-aware context representation (plus one global "query summary"
vector), so start/end predictions are conditioned on what was actually asked.

Start/end prediction uses two separate output heads over the SAME fused
representation; training minimises cross-entropy on both heads jointly.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QAModelConfig


# --------------------------------------------------------------------------- #
#  CNN block
# --------------------------------------------------------------------------- #
class CNNEncoder(nn.Module):
    """Multi-kernel 1D CNN (TextCNN-style) over embedding vectors.

    For each kernel size k we run Conv1d(embed_dim -> num_filters, width=k)
    with symmetric padding so the output sequence length is PRESERVED - this
    matters because each output position must still correspond to one input
    token for span prediction. Outputs of all kernels are concatenated and
    projected back down to `out_dim`.
    """

    def __init__(self, embed_dim, num_filters, kernel_sizes, out_dim, dropout):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=0,               # padding handled explicitly in forward
                )
                for k in kernel_sizes
            ]
        )
        self.projection = nn.Linear(num_filters * len(kernel_sizes), out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings):
        # embeddings: [batch, seq_len, embed_dim]
        x = embeddings.transpose(1, 2)                 # -> [B, E, L] (channels first)
        conv_outs = []
        for conv in self.convs:
            k = conv.kernel_size[0]
            # Asymmetric "same" padding so ANY kernel size keeps the output
            # length equal to L (symmetric padding breaks this for even k).
            left = (k - 1) // 2
            right = k - 1 - left
            conv_outs.append(F.relu(conv(F.pad(x, (left, right)))))
        x = torch.cat(conv_outs, dim=1)                # -> [B, num_filters*K, L]
        x = x.transpose(1, 2)                          # -> [B, L, num_filters*K]
        x = self.projection(x)                         # -> [B, L, out_dim]
        return self.dropout(self.norm(x))


# --------------------------------------------------------------------------- #
#  Full shared encoder: Embedding -> CNN -> BiLSTM
# --------------------------------------------------------------------------- #
class QAEncoder(nn.Module):
    """Shared encoder applied to BOTH the question and the context."""

    def __init__(self, config: QAModelConfig):
        super().__init__()
        pad_idx = 0
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim,
                                      padding_idx=pad_idx)
        self.cnn = CNNEncoder(config.embed_dim, config.cnn_num_filters,
                              list(config.cnn_kernel_sizes), config.cnn_out_dim,
                              config.dropout)
        self.rnn = nn.LSTM(
            input_size=config.cnn_out_dim,
            hidden_size=config.rnn_hidden_dim,
            num_layers=config.rnn_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.out_dim = 2 * config.rnn_hidden_dim
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, ids):
        emb = self.embedding(ids)                       # [B, L, E]
        feats = self.cnn(emb)                           # [B, L, cnn_out]  local n-grams
        rnn_out, _ = self.rnn(feats)                    # [B, L, 2H]       sequence ctx
        return self.dropout(rnn_out)


# --------------------------------------------------------------------------- #
#  Attention fusion
# --------------------------------------------------------------------------- #
def masked_softmax(scores, mask, dim=-1):
    """Softmax that ignores positions where `mask` is False."""
    scores = scores.masked_fill(~mask, float("-inf"))
    return F.softmax(scores, dim=dim)


class AttentionFusion(nn.Module):
    """BiDAF-style attention between context and question states.

    similarity S[c, q]   = h_c . h_q / sqrt(d)         (scaled dot product)
    c2q[c]               = sum_q softmax_q(S[c]) * h_q  (question-aware context)
    q2c                  = softmax_c(max_q S) @ h_c     (one query summary vector)

    Each context position is finally fused as
        Linear([h_c ; c2q ; h_c*c2q ; h_c*q2c])
    which gives every token both its own sequence state and the parts of the
    question most relevant to it.
    """

    def __init__(self, enc_dim, dropout):
        super().__init__()
        self.scale = 1.0 / math.sqrt(enc_dim)
        self.fuse = nn.Linear(enc_dim * 4, enc_dim)
        self.norm = nn.LayerNorm(enc_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_context, h_question, q_mask):
        # h_context: [B, C, D], h_question: [B, Q, D], q_mask: [B, Q] bool
        scores = torch.bmm(h_context, h_question.transpose(1, 2)) * self.scale
        c2q = masked_softmax(scores, q_mask.unsqueeze(1), dim=2)
        c2q_vec = torch.bmm(c2q, h_question)                     # [B, C, D]

        q2c = masked_softmax(scores.max(dim=2).values.unsqueeze(1),
                             torch.ones(h_context.size(0), 1, h_context.size(1),
                                        dtype=torch.bool,
                                        device=h_context.device), dim=2)
        q2c_vec = torch.bmm(q2c, h_context)                      # [B, 1, D]

        expanded = q2c_vec.expand(-1, h_context.size(1), -1)     # broadcast
        fused = torch.cat(
            [h_context, c2q_vec, h_context * c2q_vec, h_context * expanded],
            dim=2,
        )                                                        # [B, C, 4D]
        fused = self.fuse(fused)
        return self.dropout(torch.tanh(self.norm(fused)))        # [B, C, D]


# --------------------------------------------------------------------------- #
#  Full extractive QA model
# --------------------------------------------------------------------------- #
class ExtractiveQAModel(nn.Module):
    def __init__(self, config: QAModelConfig):
        super().__init__()
        self.config = config
        d = 2 * config.rnn_hidden_dim

        self.encoder = QAEncoder(config)          # shared Embedding+CNN+BiLSTM
        self.attention = AttentionFusion(d, config.dropout)

        # Two SEPARATE heads predict where the answer starts and ends.
        self.start_head = nn.Linear(d, 1)
        self.end_head = nn.Linear(d, 1)
        nn.init.xavier_uniform_(self.start_head.weight)
        nn.init.xavier_uniform_(self.end_head.weight)

    def forward(self, question_ids, question_mask, context_ids, context_mask):
        """
        Returns (start_logits, end_logits), each of shape [B, C].

        Padding positions receive -1e4 logits so they can never be selected
        by the loss (softmax -> ~0 probability) nor by inference decoding.
        """
        h_q = self.encoder(question_ids)              # [B, Q, D]
        h_c = self.encoder(context_ids)               # [B, C, D]

        fused = self.attention(h_c, h_q, question_mask)  # [B, C, D]

        start_logits = self.start_head(fused).squeeze(-1)  # [B, C]
        end_logits = self.end_head(fused).squeeze(-1)      # [B, C]

        neg = torch.tensor(-1e4, device=fused.device)
        start_logits = start_logits.masked_fill(~context_mask, neg)
        end_logits = end_logits.masked_fill(~context_mask, neg)
        return start_logits, end_logits


# --------------------------------------------------------------------------- #
#  Span decoding (shared by validation and backend inference)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def decode_best_span(start_logits, end_logits, max_answer_len=40):
    """Pick the highest-scoring valid span from one example's logits.

    Score(start i, end j) = logit_s[i] + logit_e[j], constrained to
    i <= j <= i + max_answer_len - 1. This mirrors how official SQuAD systems
    decode and prevents nonsensical very long answers.
    """
    n = start_logits.shape[-1]
    best_score, best_span = float("-inf"), (0, min(1, n - 1))
    starts = start_logits.tolist()
    ends = end_logits.tolist()
    for i in range(n):
        if starts[i] <= -1e4:
            continue  # padding position
        upper = min(n, i + max_answer_len)
        for j in range(i, upper):
            if ends[j] <= -1e4:
                break
            score = starts[i] + ends[j]
            if score > best_score:
                best_score, best_span = score, (i, j)
    return best_span


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
