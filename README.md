# Bilingual (AR-EN) LLM

# Bilingual Arabic–English LLM from Scratch

A 42M-parameter GPT-style language model for **Arabic and English**, built end to end in PyTorch: data cleaning, a bilingual BPE tokenizer, pretraining from random weights, and LoRA instruction finetuning into a small chat assistant that answers questions about Egypt.

| Stage | Data | Result |
|---|---|---|
| Pretraining | 48.4M tokens (Arabic + English Wikipedia, Egyptian history) | eval loss 8.2 → **≈4.2** |
| LoRA finetuning | 57,488 question–answer pairs | eval loss 3.03 → **≈2.57** |

---

## What the model can do

After finetuning, the model answers in the language of the question, in one fluent sentence, and stops on its own. Facts are often wrong, which is expected at 42M parameters: the model learned **how to answer** much better than **what is true**.

| Question | Model answer (step 25,500) | Verdict |
|---|---|---|
| ما اسم المعبد الجنائزي الذي شيّدته حتشبسوت، وأين يقع؟ | حتشبسوت هو صاحب المعبد الجنائزي الذي شيد داخل معبد دير البحري… | ✅ Deir el-Bahari is correct |
| مع أي ممثلة شارك عمر الشريف بطولة فيلم «صراع في الوادي»؟ | عمر الشريف شارك في الفيلم المصري الذي اخرجه يوسف شاهين. | ⚠️ True fact (Chahine directed it), wrong question |
| Which major U.S. science honor did Mostafa El-Sayed receive in 2007? | Mostafa El-Sayed received the U.S. National Institute of Technology in 2007. | ⚠️ Close (National Medal of Science) |
| إلى أين نُفي أحمد شوقي، ولماذا؟ | احمد شوقي نفي الي فرنسا… | ❌ He was exiled to Spain |
| Which puppet film trilogy is the artist Wael Shawky known for? | Wael Shawky was known for playing American actor Lance Hodge… | ❌ Hallucinated |

*Arabic answers have no hamza or diacritics because the tokenizer normalises them away (see [Tokenizer](#tokenizer)).*

![Finetuning samples](Trackio%20Dashboard/Samples.png)

---

## Pipeline

```
HF dataset ──► clean corpus ──► train BPE tokenizer ──► pack into 1,024-token windows
                                                              │
                                   pretraining (15 epochs) ◄──┘
                                              │
                         LoRA finetuning on Q&A pairs (100 epochs)
                                              │
                               merged chat model (.pt)
```

| Step | Script |
|---|---|
| 1. Explore the dataset | `scripts/ch02-dataset-exploration/dataset-exploration.ipynb` |
| 2. Clean the corpus | `scripts/ch03-build-tokenizer/01_clean_corpus.py` |
| 3. Train the tokenizer | `scripts/ch03-build-tokenizer/02_build_tokenizer.py` |
| 4. Pack the corpus | `scripts/ch03-build-tokenizer/03_pack_corpus.py` |
| 5. Pretrain | `run_pretrain.py` |
| 6. Finetune (LoRA) | `run_finetune_lora.py` |

---

## Data

Source: [`bakrianoo/jabarti-llm-dataset`](https://huggingface.co/datasets/bakrianoo/jabarti-llm-dataset) on Hugging Face.

| Split | Documents / pairs | Tokens | Packed windows (1,025 tokens) |
|---|---|---|---|
| Pretrain – train | 500,000 documents | 48,400,304 | 47,219 |
| Pretrain – eval | 114,259 documents | 10,990,923 | 10,722 |
| Finetune – train | 57,488 Q&A pairs | – | – |
| Finetune – eval | 6,379 Q&A pairs | – | – |

The pretraining corpus mixes general Arabic and English Wikipedia with a curated Egyptian-history collection. Documents are packed back to back into fixed windows, separated by `[EOS]`, so no compute is spent on padding.

---

## Tokenizer

- **BPE** (SentencePiece-style word boundaries), 32,000 tokens, trained on the cleaned bilingual corpus.
- **Arabic normalisation** (saved inside the tokenizer, so training and inference match):
  - removes diacritics (tashkeel) and tatweel
  - unifies alef forms (أ إ آ → ا) and alef maqsura (ى → ي)
  - collapses repeated spaces, NFC unicode normalisation
- **Special tokens:** `[PAD]` `[UNK]` `[BOS]` `[EOS]` `[SEP]` `[AR]` `[EN]` and the chat tokens `[SYS]` `[USER]` `[ASST]`.

---

## Model

A decoder-only Transformer in the GPT-2 style, written from scratch in `jabarti_llm/model/`.

| Setting | Value |
|---|---|
| Parameters | 42.1M |
| Layers | 8 |
| Hidden size (d_model) | 512 |
| Attention heads | 8 (64 dims each) |
| Feed-forward size | 2,048 (4 × d_model), GELU |
| Context length | 1,024 tokens |
| Positional encoding | learned |
| Normalisation | pre-LayerNorm |
| Dropout | 0.05 |
| Input/output embeddings | tied |
| Generation | KV cache, temperature, top-k, top-p |

---

## Training

Both stages ran on a single **NVIDIA A100** in Google Colab, tracked with [Trackio](https://github.com/gradio-app/trackio).

### 1. Pretraining

| Setting | Value |
|---|---|
| Epochs / steps | 15 / 3,165 |
| Effective batch | 224 sequences (56 × 4 accumulation) ≈ 229k tokens/step |
| Tokens seen | ≈ 725M |
| Optimiser | AdamW, weight decay 0.1, gradient clipping 1.0 |
| Learning rate | 6e-4, 500 warmup steps, cosine decay to 10% |
| Time | ≈ 5 hours |

| Train loss | Eval loss |
|---|---|
| ![Pretrain train loss](Trackio%20Dashboard/pretrain_train_loss.png) | ![Pretrain eval loss](Trackio%20Dashboard/pretrain_eval_loss.png) |
| **Learning rate** | **Gradient norm** |
| ![Pretrain LR](Trackio%20Dashboard/pretrain_learning_rate.png) | ![Pretrain grad norm](Trackio%20Dashboard/pretrain_grad_norm.png) |

Eval loss falls from 8.2 to ≈4.2 and was **still decreasing** at the last step, so 15 passes over the data did not overfit. The gradient norm stays well below the clipping threshold, and the learning rate follows the warmup + cosine schedule.

### 2. LoRA finetuning

LoRA adapters are added to the four attention projections (Q, K, V, output) of all 8 layers; the base model stays frozen. Loss is computed **only on answer tokens**, so the model is graded on its reply, not on predicting the question. After training, the adapters are merged back into the weights.

Chat format:

```
[BOS] [SYS] system prompt [USER] question [ASST] answer [EOS]
```

| Setting | Value |
|---|---|
| LoRA rank / alpha | 256 / 512 (32 adapted projections) |
| Trainable parameters | 8.39M of 50.5M (16.6%) |
| Epochs / steps | 100 / 25,700 |
| Effective batch | 224 pairs (56 × 4 accumulation) |
| Learning rate | 2e-5, 100 warmup steps, cosine decay |
| Time | ≈ 3 hours |

| Train loss | Eval loss |
|---|---|
| ![Finetune train loss](Trackio%20Dashboard/finetune_train_loss.png) | ![Finetune eval loss](Trackio%20Dashboard/finetune_eval_loss.png) |
| **Learning rate** | **Gradient norm** |
| ![Finetune LR](Trackio%20Dashboard/fintune_learning_rate.png) | ![Finetune grad norm](Trackio%20Dashboard/fintune_grad_norm.png) |

Eval loss falls from 3.03 to ≈2.57 and flattens in the second half. The gap to train loss grows from ≈0.3 to ≈0.45, a sign the model starts memorising the training answers, but eval loss never rises, so the final checkpoint is also the best one.

---

## Quick start

```bash
pip install uv
uv pip install -e "."
```

**Pretraining**

```bash
python run_pretrain.py --epochs 15 \
  --batch-size 56 --accumulation-steps 4 \
  --lr 6e-4 --warmup 500 --weight-decay 0.1 --eval-every 100 \
  --print-every 100 --sample-every 100 --shuffle-seed 42 \
  --checkpoint-dir checkpoints/jabarti-512x8-ep15 --save-every 100
```

**LoRA finetuning**

```bash
python run_finetune_lora.py \
  --resume checkpoints/jabarti-512x8-ep15/pretrain_final.pt \
  --epochs 100 --batch-size 56 --accumulation-steps 4 \
  --lr 2e-5 --warmup 100 --lora-r 256 --lora-alpha 512 \
  --print-every 100 --sample-every 500 --shuffle-seed 42 --shuffle \
  --checkpoint-dir checkpoints/jabarti-512x8-lora --save-every 2000
```

**Chat with the model**

```python
from jabarti_llm import (GPT, Tokenizer, GenerationConfig,
                         load_checkpoint, model_config_from_checkpoint, generate)

ckpt = "checkpoints/jabarti-512x8-lora/finetune_chat_lora_final.pt"
model = GPT(model_config_from_checkpoint(ckpt)).to("cuda")
load_checkpoint(ckpt, model)
tokenizer = Tokenizer.from_file()

print(generate(model, tokenizer, "ما هي عاصمة مصر؟", GenerationConfig(), device="cuda", chat=True))
```

Model weights are not stored in this repository.

---

## Repository structure

```
├── jabarti_llm/              # the library
│   ├── model/                # embeddings, attention, block, GPT, LoRA
│   ├── data/                 # packed dataset, chat formatting, collate
│   ├── training/             # trainer, schedule, checkpointing, tracking
│   ├── generation/           # sampling and generation with KV cache
│   ├── tokenizer.py
│   ├── config.py             # model, training and generation configs
│   └── assets/               # tokenizer.json + step-by-step notebooks
├── scripts/                  # data exploration, cleaning, tokenizer, packing
├── Trackio Dashboard/        # training curves and samples
├── run_pretrain.py
├── run_finetune_lora.py
└── pyproject.toml
```

---

## Limitations and next steps

- **Factual accuracy is low.** 42M parameters and 48M unique tokens are not enough to store many facts; the model often hallucinates with confidence.
- **Repeated data.** 15 pretraining epochs and 100 finetuning epochs reuse the same examples; more unique data would help more than more epochs.
- **No benchmark evaluation yet.** Next steps: per-language eval loss (Arabic vs. English), tokenizer fertility analysis (tokens per word in each language), and a small factual QA benchmark.

---

## Author

**Ahmed Ayman Abdelhadi** · [GitHub](https://github.com/ayman23-ds)

## Acknowledgements

- **Abu Bakr Soliman** for the *Build an LLM from Scratch with PyTorch* course, the [reference implementation](https://github.com/bakrianoo/jabarti-llm-from-scratch) and the [Jabarti dataset](https://huggingface.co/datasets/bakrianoo/jabarti-llm-dataset).
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019.
