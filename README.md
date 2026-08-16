# English to Spanish Transformer

A 30M-parameter encoder-decoder transformer, written and trained from
scratch in PyTorch as a learning project. Multi-head attention, positional
encoding and the encoder/decoder stacks are hand-written rather than taken from
`nn.Transformer`, and no pretrained weights are used.

The goal was to understand the architecture from *Attention Is All You Need* by
building it, not to compete with production translation systems.

**[Live demo](https://transformer-translator-harish.streamlit.app)**

## What it can and cannot do

Trained on 93,470 English-Spanish sentence pairs from the `opus_books`
corpus -- 19th-century novels. It handles that register reasonably and struggles
with modern or everyday English.

It gets right: inverted question marks, verb conjugation, article and adjective
agreement, and short declarative sentences.

Known limitations:

- **Narrow domain.** Modern topics are outside everything it has seen, and it
  will sometimes reach for a memorized phrase from the books instead.
- **Small dataset.** 93,470 pairs is tiny for translation; production
  systems train on millions.
- **Undertrained.** Final training loss 3.93, still falling when
  training stopped. Well-trained systems reach roughly half that.
- **Word-level vocabulary.** Words are atomic, so anything outside the
  29,549-word vocabulary is unrepresentable. A subword tokenizer such as
  BPE would split unseen words into familiar pieces.
- **Greedy decoding.** It commits to the highest-probability word at each step.
  Beam search would explore several candidates.

The app detects out-of-vocabulary words as you type and can restrict input to
words the model knows.

## Model

| | |
|---|---|
| Parameters | 30.3M |
| Model dimension | 256 |
| Layers | 4 |
| Attention heads | 8 |
| Feed-forward dimension | 1024 |
| Max sequence length | 160 |
| Vocabulary | 29,549 en / 30,000 es |
| Training | 8 epochs on CPU |
| Final loss | 3.93 |

Weights ship as fp16 (`model.pt`, ~61MB) with the optimizer state stripped, and
are loaded back as fp32 at runtime.

## Repository layout

| File | |
|---|---|
| `app.py` | the Streamlit demo |
| `main_model.py` | the transformer: attention, positional encoding, encoder/decoder |
| `train.py` | training loop, greedy decoding, tokenizer building |
| `dataset.py` | dataset, padding and causal masking |
| `config.py` | hyperparameters |
| `export_for_deploy.py` | packages a checkpoint into a deployable folder |
| `model.pt` | trained weights, fp16 |

## Running the demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reproducing the training

```bash
pip install -r requirements-train.txt
python train.py
```

This downloads `opus_books`, builds the tokenizers, and writes a checkpoint per
epoch to `weights/` with tensorboard logs in `runs/`. Neither directory is
committed. Training took roughly 4.7 hours per epoch on a CPU; a GPU is
much faster.

To package a checkpoint for deployment:

```bash
python export_for_deploy.py        # latest epoch
python export_for_deploy.py 05     # a specific epoch
```
