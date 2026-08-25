# Question Answering System using Deep Learning

## Abstract

This project implements a **neural extractive Question Answering (QA) system** that reads a user-provided passage and a question, then predicts the answer by extracting the most relevant span directly from the context. Unlike keyword-matching approaches, this system uses a genuine deep-learning architecture consisting of **embeddings, multi-kernel CNN feature extraction, a bidirectional LSTM sequence encoder, and a BiDAF-style attention mechanism** to localise start and end positions of the answer within the passage. The system achieves **88.9% Exact Match** and **88.9% token F1** on the built-in SQuAD-style sample dataset.

---

## Problem Statement

Extractive Question Answering requires a model to read a passage, understand its content, and identify the exact span of text that answers a given question. This is a core NLP task and a key benchmark for measuring how well neural networks can comprehend natural language.

**Challenges addressed:**
- Mapping character-level answer offsets to token positions
- Learning which question words are semantically relevant to which context words
- Predicting *two* positions (start and end) simultaneously
- Running on consumer hardware without GPU requirements

---

## Objectives

1. Build a working end-to-end QA pipeline: text preprocessing → model inference → answer extraction
2. Demonstrate CNN-based local feature extraction for NLP (n-gram detection)
3. Combine CNN with BiLSTM for sequential context modelling
4. Implement attention-based question-context fusion
5. Provide a polished web UI for interactive use
6. Keep the project lightweight enough for a normal laptop

---

## Features

- **Extractive QA**: answers are spans from the original context (no hallucination)
- **CNN feature extraction**: multi-kernel 1D convolutions capture bigram/trigram patterns
- **BiLSTM encoder**: bidirectional sequence modelling for long-range dependencies
- **Attention fusion**: question-aware context representation (BiDAF-style)
- **Confidence scoring**: geometric mean of start/end softmax probabilities
- **Answer highlighting**: the answer span is highlighted in the context
- **Training history charts**: loss curves and accuracy metrics visualised
- **Sample questions**: built-in examples across 8 topic areas
- **Input validation**: helpful error messages for invalid inputs
- **Responsive UI**: works on desktop and mobile

---

## Technologies Used

| Layer | Technology |
|-------|-----------|
| Deep Learning Framework | PyTorch 2.x (CPU) |
| Backend API | Flask + Flask-CORS |
| Frontend | React 18 + Vite 5 |
| Styling | Pure CSS (dark AI theme) |
| Dataset | SQuAD v1.1 format (hand-crafted sample + full download) |
| Language | Python 3.13 / JavaScript (ES modules) |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (React)                      │
│  Context textarea → Question input → "Get Answer" btn   │
│  Answer card with confidence bar + highlighted context   │
│  Architecture diagram, Training charts, Model info       │
└───────────────────────┬─────────────────────────────────┘
                        │ POST /api/predict  (JSON)
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  BACKEND (Flask, port 5000)               │
│  /api/health  /api/predict  /api/model/info  /api/model/history │
│  Input validation → Inference engine → Answer extraction │
└───────────────────────┬─────────────────────────────────┘
                        │ forward pass
                        ▼
┌─────────────────────────────────────────────────────────┐
│             EXTRACTIVE QA MODEL (PyTorch)                 │
│  Embedding → CNN (k=2,3,4) → BiLSTM → Attention → Heads │
│  predicted_start, predicted_end → answer span            │
└─────────────────────────────────────────────────────────┘
```

---

## Deep Learning Architecture

The model follows this pipeline (see `model/qa_model.py` for the full implementation):

```
Question tokens ─┐                 ┌─ Context tokens
                  ▼                 ▼
          ┌──────────────────────────────┐
          │     Shared Embedding Layer   │  nn.Embedding (vocab_size × 96)
          └──────────┬───────────────────┘
                     ▼
          ┌──────────────────────────────┐
          │  CNN Feature Extraction      │  3 parallel Conv1d layers:
          │  (multi-kernel TextCNN)      │    k=2 (bigrams), k=3 (trigrams),
          │                              │    k=4 (4-grams)
          │  Concat → Project → LayerNorm│  96 filters each → 128-dim output
          └──────────┬───────────────────┘
                     ▼
          ┌──────────────────────────────┐
          │  BiLSTM Sequence Encoder     │  1-layer, hidden=128 per direction
          │  (bidirectional LSTM)        │  output: 256-dim per token
          └──────────┬───────────────────┘
                     ▼
          ┌──────────────────────────────┐
          │  Attention Fusion            │  Scaled dot-product attention:
          │  (BiDAF-style)               │    c2q: question-aware context
          │                              │    q2c: context-aware query summary
          │  Linear([h_c; c2q; ⊙; ⊙])  │  → fused 256-dim representation
          └──────┬───────────┬───────────┘
                 ▼           ▼
          ┌──────────┐ ┌──────────┐
          │ Start Head│ │ End Head │  Linear(256 → 1) per position
          └──────────┘ └──────────┘
                 ▼           ▼
          Cross-entropy loss (training)
          Argmax span decoding (inference)
```

### Architecture details

| Component | Purpose | Key hyperparameter |
|-----------|---------|-------------------|
| **Embedding** | Maps token IDs to dense vectors, enabling the network to learn semantic relationships between words | 96-dimensional |
| **CNN (n-gram)** | Detects local patterns (e.g. "carbon dioxide", "white marble") using filters of width 2, 3, and 4. This gives the model translation-invariant n-gram features BEFORE sequential processing | 96 filters × 3 kernel sizes |
| **BiLSTM** | Models long-range sequential context. Every token's representation now contains information from both its left and right neighbours — essential for deciding where answer spans begin and end | 128 hidden × 2 directions |
| **Attention** | Compares every context position with every question position, building a question-aware context representation. This ensures the start/end predictions are conditioned on WHAT was asked, not just what words appear in the passage | Scaled dot-product |
| **Start/End heads** | Two separate linear layers predict log-probabilities over all context positions for the start and end of the answer | Linear(256 → 1) each |

**Total trainable parameters:** 691,682

---

## The Role of CNN

The CNN layer is a critical component of this architecture and is NOT a superficial addition. Specifically:

1. **Local pattern detection**: Conv1d filters of width 2/3/4 slide over the embedding sequence, detecting bigram, trigram, and 4-gram patterns. A filter trained to recognise "carbon dioxide" will fire wherever those two words appear next to each other — regardless of their position in the sentence.

2. **Feature enrichment**: The concatenated CNN output provides the BiLSTM with a richer input than raw embeddings alone. The LSTM then models longer-range dependencies on top of these local features.

3. **Efficiency**: CNNs process the entire sequence in parallel (no sequential dependency between positions), making this step very fast.

4. **Academic relevance**: The TextCNN architecture (Kim, 2014) is a well-known baseline for text classification. Here it is adapted as a feature extractor within a larger QA pipeline, demonstrating how CNN and RNN components can be combined.

---

## The Role of BiLSTM

After the CNN has extracted local features, the BiLSTM reads them sequentially:

- The **forward** LSTM processes left-to-right: at each position, it knows everything that came before
- The **backward** LSTM processes right-to-left: it knows everything that comes after
- The **concatenated** output therefore gives each token a representation of the ENTIRE context, enabling the model to relate, for example, "it" back to a noun mentioned 15 tokens earlier

This is essential for span prediction because the start and end positions of an answer depend on understanding the full context, not just local words.

---

## Dataset

### Built-in sample dataset

The project ships with a hand-crafted SQuAD-format file (`dataset/sample_squad.json`) containing **8 passages and 61 questions** across topics like photosynthesis, the Internet, the Taj Mahal, the solar system, machine learning, the human heart, Mount Everest, and the Pacific Ocean. This dataset is augmented with paraphrased question variants to increase the effective training set size.

### Full SQuAD v1.1

For larger experiments, download the official Stanford Question Answering Dataset:
```bash
python dataset/download_squad.py
```
This fetches `train-v1.1.json` (~30 MB, ~87,000 QA pairs) and `dev-v1.1.json` (~5 MB, ~10,000 QA pairs).

---

## Training Procedure

### Quick demo (sample dataset)

```bash
python training/train.py
```
Trains on the built-in 61-question sample for 40 epochs. Takes under 3 minutes on CPU.

### Full SQuAD training

```bash
python dataset/download_squad.py
python training/train.py \
    --train-file dataset/train-v1.1.json \
    --val-file dataset/dev-v1.1.json \
    --epochs 3 \
    --batch-size 64 \
    --max-context-len 320 \
    --max-question-len 32 \
    --min-token-freq 2
```

### Configurable parameters

| Parameter | Default (sample) | Full SQuAD | Description |
|-----------|-----------------|------------|-------------|
| `--epochs` | 40 | 3 | Number of training epochs |
| `--batch-size` | 32 | 64 | Batch size |
| `--learning-rate` | 1e-3 | 1e-3 | Adam learning rate |
| `--embed-dim` | 96 | 96 | Embedding dimension |
| `--max-context-len` | 160 | 320 | Max context tokens |
| `--max-question-len` | 24 | 32 | Max question tokens |
| `--dropout` | 0.25 | 0.25 | Dropout rate |
| `--patience` | 12 | 12 | Early-stopping patience |

### What gets saved

After training, the `saved_models/` directory contains:
- `qa_model.pt` — model weights
- `model_config.json` — architecture hyper-parameters
- `vocab.json` — token vocabulary
- `training_history.json` — per-epoch loss, EM, F1
- `metrics.json` — best validation scores

---

## API Documentation

### `GET /api/health`

Returns server status and whether a model is loaded.

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "backend_version": "1.0.0"
}
```

### `POST /api/predict`

**Request:**
```json
{
  "context": "The Taj Mahal is a white marble mausoleum in Agra, India.",
  "question": "Where is the Taj Mahal located?"
}
```

**Response:**
```json
{
  "answer": "Agra, India",
  "confidence": 0.9987,
  "start_prob": 0.9993,
  "end_prob": 0.9981,
  "start_char": 47,
  "end_char": 57,
  "start_token": 8,
  "end_token": 9,
  "latency_ms": 46.7,
  "context_tokens": 12,
  "question_tokens": 6,
  "truncated": false
}
```

**Error responses:**
- `400` — missing or invalid input
- `503` — no trained model loaded
- `500` — internal prediction error

### `GET /api/model/info`

Returns architecture configuration, parameter count, and best validation metrics.

### `GET /api/model/history`

Returns the full training history (epoch-by-epoch loss, EM, F1) for the frontend charts.

---

## Installation Steps

### Prerequisites
- Python 3.10+
- Node.js 18+
- 4 GB RAM (for full SQuAD) or 2 GB (for sample dataset)

### Windows

```bash
# 1. Install PyTorch (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install frontend dependencies
cd frontend
npm install
cd ..
```

Or use the convenience scripts:
```bash
# setup_windows.bat — installs everything
setup_windows.bat
```

### macOS / Linux

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

---

## How to Train the Model

```bash
# Quick training on sample data (recommended first time)
python training/train.py

# Full SQuAD training (after downloading)
python dataset/download_squad.py
python training/train.py --train-file dataset/train-v1.1.json --val-file dataset/dev-v1.1.json --epochs 3
```

---

## How to Start the Backend

```bash
python backend/app.py
# Server runs on http://127.0.0.1:5000
```

---

## How to Start the Frontend

In a **second terminal**:
```bash
cd frontend
npm run dev
# Frontend runs on http://localhost:5173
```

The Vite dev server proxies `/api/*` requests to the Flask backend automatically.

---

## Example Input / Output

### Example 1
**Context:** "Photosynthesis is the process used by plants to convert light energy into chemical energy. It takes place inside organelles called chloroplasts."
**Question:** "Where does photosynthesis take place?"
**Answer:** "chloroplasts" (confidence: 99.6%)

### Example 2
**Context:** "The Taj Mahal was commissioned in 1632 by the Mughal emperor Shah Jahan in memory of his favourite wife, Mumtaz Mahal."
**Question:** "Who commissioned the Taj Mahal?"
**Answer:** "Shah Jahan" (confidence: 99.8%)

---

## Project Structure

```
question-answering-system/
├── frontend/                  # React + Vite
│   ├── src/
│   │   ├── components/        # UI components
│   │   │   ├── Header.jsx
│   │   │   ├── QAForm.jsx
│   │   │   ├── AnswerPanel.jsx
│   │   │   ├── Samples.jsx
│   │   │   ├── Architecture.jsx
│   │   │   ├── ChartsCard.jsx
│   │   │   └── ModelInfo.jsx
│   │   ├── api.js             # API client functions
│   │   ├── App.jsx            # Main app component
│   │   ├── main.jsx           # Entry point
│   │   └── styles.css         # Full stylesheet
│   ├── package.json
│   └── vite.config.js
├── backend/                   # Flask API
│   ├── app.py                 # Flask routes
│   ├── inference.py           # QA inference engine
│   └── test_e2e.py            # End-to-end verification
├── model/                     # Deep learning model
│   ├── config.py              # QAModelConfig
│   ├── qa_model.py            # CNN + BiLSTM + Attention model
│   └── __init__.py
├── preprocessing/             # Text processing
│   ├── tokenizer.py           # Regex tokenizer + char offsets
│   ├── vocabulary.py          # Token ↔ ID vocabulary
│   ├── squad_utils.py         # SQuAD parsing + span alignment
│   └── __init__.py
├── training/                  # Training pipeline
│   ├── train.py               # Full training loop
│   ├── metrics.py             # EM / F1 evaluation
│   └── __init__.py
├── dataset/                   # Data files
│   ├── sample_squad.json      # Built-in demo dataset
│   ├── sample_data_builder.py # Generates the sample file
│   └── download_squad.py      # Downloads full SQuAD v1.1
├── saved_models/              # Trained checkpoint + vocab + config
├── notebooks/                 # Jupyter notebook walkthrough
├── requirements.txt           # Python dependencies
├── setup_windows.bat          # One-click setup (Windows)
├── setup_unix.sh              # One-click setup (macOS/Linux)
├── run_backend.bat            # Start backend (Windows)
├── run_frontend.bat           # Start frontend (Windows)
└── README.md                  # This file
```

---

## Limitations

1. **Small sample dataset**: The built-in demo dataset has only 61 questions across 8 passages. Generalisation to unseen topics is limited.
2. **No pretrained embeddings**: Word embeddings are learned from scratch. Using GloVe or BERT embeddings would significantly improve performance.
3. **Extractive only**: The model can only answer by extracting spans from the context; it cannot generate new text or answer "I don't know."
4. **CPU inference**: Without GPU, inference takes ~50ms per question (adequate for interactive use but not batch processing).
5. **Maximum context length**: Contexts longer than 160 tokens (sample mode) or 320 tokens (full mode) are truncated.

---

## Future Enhancements

1. **Pretrained embeddings**: Use GloVe or fastText word vectors for initialisation
2. **Transformer backbone**: Replace CNN+BiLSTM with BERT or RoBERTa for state-of-the-art performance
3. **SQuAD 2.0 support**: Handle unanswerable questions with a "no answer" prediction
4. **Batch inference**: Process multiple questions in parallel for throughput
5. **Question generation**: Add a generative model to create questions from context
6. **Data augmentation**: Paraphrase generation to improve training data diversity
7. **Knowledge graph integration**: Combine with structured knowledge for better reasoning
8. **Multilingual support**: Extend the tokenizer and model to handle non-English text

---
