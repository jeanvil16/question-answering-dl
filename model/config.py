"""Model configuration shared by training and inference.

A single dataclass keeps the architecture hyper-parameters in one place. The
config is saved next to the weights so the backend can rebuild the *exact*
model that was trained without duplicating any values in code.
"""

import json
from dataclasses import asdict, dataclass, field


@dataclass
class QAModelConfig:
    # --- vocabulary / embeddings -----------------------------------------
    vocab_size: int = 0            # filled after the vocabulary is built
    embed_dim: int = 96            # size of each word embedding vector

    # --- CNN feature extraction -------------------------------------------
    cnn_kernel_sizes: list = field(default_factory=lambda: [2, 3, 4])
    cnn_num_filters: int = 96      # filters per kernel size
    cnn_out_dim: int = 128         # projection dim of concatenated CNN features

    # --- BiLSTM sequence encoder ------------------------------------------
    rnn_hidden_dim: int = 128      # per direction (output is 2x this)
    rnn_layers: int = 1

    # --- regularization ----------------------------------------------------
    dropout: float = 0.25

    # --- sequence limits (must match preprocessing) ------------------------
    max_question_len: int = 24
    max_context_len: int = 160
    max_answer_len: int = 40       # longest span considered at inference

    # --- bookkeeping -------------------------------------------------------
    dataset_name: str = "sample_squad"

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))
