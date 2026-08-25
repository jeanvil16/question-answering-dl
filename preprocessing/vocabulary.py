"""Word-level vocabulary for the QA model.

Embeddings (item 4 of the pipeline) map every token id to a dense vector.
This class owns the token <-> integer-id mapping that makes those embeddings
possible. It is deliberately tiny so students can read and explain it.

Special ids are fixed so checkpoints stay compatible:
    0 -> <pad>  (masking / batch padding)
    1 -> <unk>  (any word not seen during training)
"""

import json


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_INDEX = 0
UNK_INDEX = 1


class Vocabulary:
    """Bidirectional token <-> id dictionary."""

    def __init__(self):
        self.itos = [PAD_TOKEN, UNK_TOKEN]          # index -> token
        self.stoi = {PAD_TOKEN: PAD_INDEX, UNK_TOKEN: UNK_INDEX}  # token -> index

    # ------------------------------------------------------------------ build
    def add_token(self, token):
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    @classmethod
    def build(cls, token_iterables, min_freq=1, max_size=None):
        """Build a vocabulary from an iterable of token lists.

        Tokens appearing fewer than `min_freq` times are dropped; this keeps
        rare noisy words from consuming embedding rows when training on the
        full SQuAD dataset.
        """
        vocab = cls()
        freq = {}
        for tokens in token_iterables:
            for tok in tokens:
                freq[tok] = freq.get(tok, 0) + 1
        # Deterministic order: by frequency desc, then alphabetically.
        items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        for token, count in items:
            if count < min_freq:
                break
            if max_size is not None and len(vocab.itos) >= max_size:
                break
            vocab.add_token(token)
        return vocab

    # ------------------------------------------------------------------- use
    def encode(self, tokens):
        return [self.stoi.get(tok, UNK_INDEX) for tok in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

    def __len__(self):
        return len(self.itos)

    @property
    def pad_index(self):
        return PAD_INDEX

    @property
    def unk_index(self):
        return UNK_INDEX

    # ------------------------------------------------------------ persistence
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls()
        vocab.itos = list(data["itos"])
        vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}
        # Guarantee the special ids exist even in a hand-edited file.
        if vocab.itos[PAD_INDEX] != PAD_TOKEN or vocab.itos[UNK_INDEX] != UNK_TOKEN:
            raise ValueError("Vocabulary file is missing the <pad>/<unk> specials.")
        return vocab
