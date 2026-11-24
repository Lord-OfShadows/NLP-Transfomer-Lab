# NLP Transformer Lab

Three Coursera NLP assignments on attention-based models, written in TensorFlow/Keras. They follow a progression: an RNN with attention, then a Transformer built from scratch, then that Transformer reused as a T5-style model for question answering.

| # | Notebook | Task | Model | Data |
|---|----------|------|-------|------|
| 1 | [Neural Machine Translation.ipynb](Neural%20Machine%20Translation.ipynb) | English → Portuguese translation | Bi-LSTM encoder–decoder with cross-attention | English–Portuguese sentence pairs |
| 2 | [Transformer Summarize.ipynb](Transformer%20Summarize.ipynb) | Abstractive summarization | Encoder–decoder Transformer built from scratch | Document/summary corpus (`data/corpus`) |
| 3 | [Question Answering.ipynb](Question%20Answering.ipynb) | Question answering | T5-style Transformer, pretrained then fine-tuned | C4 (pretraining), SQuAD 2.0 (fine-tuning) |

## Repository structure

```
.
├── Neural Machine Translation.ipynb   # Assignment 1: seq2seq + attention, greedy & MBR decoding
├── Transformer Summarize.ipynb        # Assignment 2: full Transformer from scratch
├── Question Answering.ipynb           # Assignment 3: T5 pretraining (C4) + QA fine-tuning (SQuAD)
└── README.md
```

## How the assignments connect

```mermaid
flowchart LR
    A["1 · Neural Machine Translation<br/>LSTM + cross-attention"] -->|"attention idea"| B["2 · Transformer Summarizer<br/>self-attention, encoder/decoder"]
    B -->|"same Transformer architecture<br/>(transformer_utils)"| C["3 · Question Answering<br/>T5-style pretrain + fine-tune"]
```

---

## 1. Neural Machine Translation

An encoder–decoder RNN with attention that translates English sentences into Portuguese.

**Notebook sections**
1. Data preparation: `TextVectorization` vocabularies (12,000 tokens), `StringLookup` for id↔word mapping, batches of 64
2. Model: `Encoder`, `CrossAttention`, `Decoder`, `Translator`
3. Training: `compile_and_train`
4. Inference: `generate_next_token`, `translate` (greedy / temperature sampling)
5. Minimum Bayes-Risk (MBR) decoding: `generate_samples`, `jaccard_similarity`, `rouge1_similarity`, `average_overlap`, `weighted_avg_overlap`, `mbr_decode`

**Architecture**

```mermaid
flowchart LR
    subgraph Encoder
        E1["English tokens"] --> E2["Embedding<br/>(mask_zero)"] --> E3["Bidirectional LSTM<br/>(256 units)"]
    end
    subgraph Decoder
        D1["Shifted Portuguese tokens"] --> D2["Embedding"] --> D3["LSTM"]
        D3 --> CA["CrossAttention<br/>(1-head MultiHeadAttention<br/>+ residual + LayerNorm)"]
        CA --> D4["LSTM"] --> D5["Dense + log-softmax"]
    end
    E3 -->|"context"| CA
    D5 --> OUT["Next-token log-probs"]
```

**MBR decoding** draws several candidate translations, scores each by its average similarity to the others (Jaccard or ROUGE-1), and returns the candidate that agrees most with the rest:

```mermaid
flowchart LR
    S["Source sentence"] --> G["generate_samples<br/>(n samples, temperature)"]
    G --> C1["candidate 1"] & C2["candidate 2"] & C3["candidate n"]
    C1 & C2 & C3 --> O["Pairwise overlap<br/>(Jaccard / ROUGE-1)"]
    O --> P["Pick highest<br/>average overlap"]
```

---

## 2. Transformer Summarizer

A full Transformer implemented layer by layer and trained to generate summaries.

**Notebook sections**
1–2. Load the dataset (`utils.get_train_test_data`) and preprocess it, adding `[SOS]`/`[EOS]` tokens
3. `positional_encoding`
4. Masks: `create_padding_mask`, `create_look_ahead_mask`
5. `scaled_dot_product_attention`
6. Encoder: `FullyConnected`, `EncoderLayer`, `Encoder`
7. Decoder: `DecoderLayer`, `Decoder`
8. `Transformer` model
9–10. Initialization and training setup: `CustomSchedule` (warmup learning rate), `masked_loss`, `train_step`
11. Inference: `next_word`, `summarize`
12–13. Training and sample summaries

**Architecture**

```mermaid
flowchart TB
    IN["Document tokens"] --> EMB1["Embedding × √d + Positional Encoding"]
    EMB1 --> ENC["Encoder layer × N<br/>Multi-head self-attention → Add & Norm<br/>Feed-forward → Add & Norm"]
    OUTIN["Summary tokens (shifted right)"] --> EMB2["Embedding × √d + Positional Encoding"]
    EMB2 --> DEC["Decoder layer × N<br/>Masked self-attention (look-ahead mask) → Add & Norm<br/>Encoder–decoder attention (padding mask) → Add & Norm<br/>Feed-forward → Add & Norm"]
    ENC -->|"enc_output"| DEC
    DEC --> LIN["Dense → softmax"] --> NEXT["Next word"]
    NEXT -.->|"append & repeat until [EOS]"| OUTIN
```

Configuration used for training: `num_layers=2`, `embedding_dim=128`, `num_heads=2`, `fully_connected_dim=128`, `encoder_maxlen=150`.

---

## 3. Question Answering (T5-style)

The Transformer from assignment 2 (provided as `transformer_utils`) is pretrained with a masked-span objective on C4, then fine-tuned on SQuAD 2.0.

**Notebook sections**
1. Pretraining data
   - The C4 dataset and the masked-language-model objective
   - `get_sentinels`, `pretty_decode`: map sentinel tokens to readable markers
   - `tokenize_and_mask`: randomly drop tokens into sentinel-marked targets
   - `display_input_target_pairs`
2. Pretraining: instantiate `transformer_utils.Transformer`, train with `CustomSchedule`, then load provided pretrained weights
3. Fine-tuning
   - `parse_squad`: build `question: … context: …` → answer pairs
   - Fine-tune the pretrained model
   - `answer_question`: generate answers token by token with `transformer_utils.next_word`

**Pipeline**

```mermaid
flowchart LR
    subgraph Pretraining
        C4["C4 web text"] --> TM["tokenize_and_mask"]
        TM --> X1["Input: text with sentinels<br/>'Younes and Lukasz ‹Z› together in the ‹Y› …'"]
        TM --> Y1["Target: masked spans<br/>'‹Z› were working ‹Y› lab.'"]
        X1 & Y1 --> PT["Train Transformer"]
    end
    subgraph Fine-tuning
        SQ["SQuAD 2.0"] --> PS["parse_squad"]
        PS --> X2["question: … context: …"]
        PS --> Y2["answer: …"]
        X2 & Y2 --> FT["Fine-tune pretrained weights"]
    end
    PT -->|"pretrained weights"| FT
    FT --> AQ["answer_question"]
```

---

## Running the notebooks

The notebooks import helper modules and data that came with the Coursera lab environment and are **not included in this repository**:

| Notebook | Required files |
|----------|----------------|
| Neural Machine Translation | `utils.py`, `w1_unittest.py`, translation data |
| Transformer Summarize | `utils.py`, `w2_unittest.py`, `data/corpus/` |
| Question Answering | `utils.py`, `transformer_utils.py`, `w3_unittest.py`, C4 / SQuAD data, `pretrained_models/` |

To run them outside Coursera, download those files from the lab workspace and put them next to the notebooks. Main dependencies: `tensorflow`, `tensorflow_text`, `numpy`, `pandas`, `matplotlib`, `termcolor`.
