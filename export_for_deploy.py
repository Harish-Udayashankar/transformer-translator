import json
import shutil
import sys
from pathlib import Path

import torch

from config import get_config, get_weights_file_path
from train import get_model

DEPLOY = Path('deploy')
CURVE_POINTS = 1000


REQUIREMENTS = """\
--extra-index-url https://download.pytorch.org/whl/cpu
streamlit
torch==2.2.2+cpu
numpy<2
tokenizers
pandas
"""

REQUIREMENTS_TRAIN = """\
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.2.2+cpu
numpy<2
tokenizers
datasets
tensorboard
tqdm
"""

TRAINING_FILES = ['train.py', 'dataset.py', 'config.py', 'export_for_deploy.py']

GITIGNORE = """\
__pycache__/
*.pyc
.DS_Store
weights/
runs/
"""

README = """\
# English to Spanish Transformer

A {params:.0f}M-parameter encoder-decoder transformer, written and trained from
scratch in PyTorch as a learning project. Multi-head attention, positional
encoding and the encoder/decoder stacks are hand-written rather than taken from
`nn.Transformer`, and no pretrained weights are used.

The goal was to understand the architecture from *Attention Is All You Need* by
building it, not to compete with production translation systems.

## What it can and cannot do

Trained on {train_pairs:,} English-Spanish sentence pairs from the `opus_books`
corpus -- 19th-century novels. It handles that register reasonably and struggles
with modern or everyday English.

It gets right: inverted question marks, verb conjugation, article and adjective
agreement, and short declarative sentences.

Known limitations:

- **Narrow domain.** Modern topics are outside everything it has seen, and it
  will sometimes reach for a memorized phrase from the books instead.
- **Small dataset.** {train_pairs:,} pairs is tiny for translation; production
  systems train on millions.
- **Undertrained.** Final training loss {final_loss:.2f}, still falling when
  training stopped. Well-trained systems reach roughly half that.
- **Word-level vocabulary.** Words are atomic, so anything outside the
  {vocab_src:,}-word vocabulary is unrepresentable. A subword tokenizer such as
  BPE would split unseen words into familiar pieces.
- **Greedy decoding.** It commits to the highest-probability word at each step.
  Beam search would explore several candidates.

The app detects out-of-vocabulary words as you type and can restrict input to
words the model knows.

## Model

| | |
|---|---|
| Parameters | {params:.1f}M |
| Model dimension | {d_model} |
| Layers | {N} |
| Attention heads | {h} |
| Feed-forward dimension | {d_ff} |
| Max sequence length | {seq_len} |
| Vocabulary | {vocab_src:,} en / {vocab_tgt:,} es |
| Training | {epochs} epochs on CPU |
| Final loss | {final_loss:.2f} |

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
committed. Training took roughly {hours_per_epoch} per epoch on a CPU; a GPU is
much faster.

To package a checkpoint for deployment:

```bash
python export_for_deploy.py        # latest epoch
python export_for_deploy.py 05     # a specific epoch
```
"""


def latest_epoch(config):
    folder = Path(config['model_folder'])
    basename = config['model_basename']
    epochs = sorted(p.stem[len(basename):] for p in folder.glob(f'{basename}*.pt'))
    if not epochs:
        sys.exit(f'No checkpoints found in {folder}/ -- train the model first.')
    return epochs[-1]


def export_loss_curve(config, out_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import pandas as pd

    logdir = Path(config['experiment_name'])
    if not logdir.exists():
        return None

    accumulator = EventAccumulator(str(logdir))
    accumulator.Reload()
    if 'train_loss' not in accumulator.Tags().get('scalars', []):
        return None

    df = pd.DataFrame(
        [{'step': s.step, 'train_loss': s.value}
         for s in accumulator.Scalars('train_loss')]
    ).sort_values('step')

    if len(df) > CURVE_POINTS:
        df = df.iloc[:: len(df) // CURVE_POINTS]
    df.to_csv(out_path, index=False)
    return df


def export_examples(config, tokenizer_src, out_path, n=10):
    import json as _json
    import random

    from datasets import load_dataset

    ds = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}',
                      split='train')
    if config['max_examples']:
        ds = ds.select(range(min(config['max_examples'], len(ds))))

    candidates = []
    for item in ds:
        src = item['translation'][config['lang_src']]
        tgt = item['translation'][config['lang_tgt']]
        words = [w for w, _ in tokenizer_src.pre_tokenizer.pre_tokenize_str(src)]
        if not (5 <= len(words) <= 12):
            continue
        if any(tokenizer_src.token_to_id(w) is None for w in words):
            continue
        candidates.append({'en': src.strip(), 'es': tgt.strip()})

    random.Random(0).shuffle(candidates)
    picked = candidates[:n]
    out_path.write_text(_json.dumps(picked, indent=2, ensure_ascii=False))
    return picked, len(ds)


def main():
    from tokenizers import Tokenizer

    config = get_config()
    epoch = sys.argv[1] if len(sys.argv) > 1 else latest_epoch(config)

    DEPLOY.mkdir(exist_ok=True)
    print(f'Exporting epoch {epoch} -> {DEPLOY}/')

    names = {}
    for key, lang in [('tokenizer_src', config['lang_src']),
                      ('tokenizer_tgt', config['lang_tgt'])]:
        name = config['tokenizer_file'].format(lang)
        shutil.copy(name, DEPLOY / name)
        names[key] = name
    tokenizer_src = Tokenizer.from_file(str(DEPLOY / names['tokenizer_src']))
    tokenizer_tgt = Tokenizer.from_file(str(DEPLOY / names['tokenizer_tgt']))

    checkpoint = torch.load(get_weights_file_path(config, epoch), map_location='cpu')
    weights = {k: v.half() for k, v in checkpoint['model_state_dict'].items()}
    torch.save({'model_state_dict': weights}, DEPLOY / 'model.pt')

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    params_millions = sum(p.numel() for p in model.parameters()) / 1e6

    curve = export_loss_curve(config, DEPLOY / 'loss_curve.csv')
    final_loss = float(curve['train_loss'].iloc[-1]) if curve is not None else 0.0

    picked, train_pairs = export_examples(config, tokenizer_src,
                                          DEPLOY / 'examples.json')
    print(f'  ({len(picked)} example sentences, {train_pairs} training pairs)')

    meta = {
        'd_model': config['d_model'],
        'N': config['N'],
        'h': config['h'],
        'd_ff': config['d_ff'],
        'seq_len': config['seq_len'],
        'vocab_src': tokenizer_src.get_vocab_size(),
        'vocab_tgt': tokenizer_tgt.get_vocab_size(),
        'lang_src': config['lang_src'],
        'lang_tgt': config['lang_tgt'],
        'tokenizer_src': names['tokenizer_src'],
        'tokenizer_tgt': names['tokenizer_tgt'],
        'epochs_trained': int(epoch) + 1,
        'params_millions': params_millions,
        'final_loss': final_loss,
        'corpus': 'the opus_books English-Spanish corpus',
        'train_pairs': train_pairs,
    }
    (DEPLOY / 'model_config.json').write_text(json.dumps(meta, indent=2))

    shutil.copy('deploy_app.py', DEPLOY / 'app.py')
    shutil.copy('main_model.py', DEPLOY / 'main_model.py')
    for name in TRAINING_FILES:
        shutil.copy(name, DEPLOY / name)
    (DEPLOY / 'requirements.txt').write_text(REQUIREMENTS)
    (DEPLOY / 'requirements-train.txt').write_text(REQUIREMENTS_TRAIN)
    (DEPLOY / '.gitignore').write_text(GITIGNORE)
    (DEPLOY / 'README.md').write_text(README.format(
        params=params_millions, train_pairs=train_pairs, final_loss=final_loss,
        d_model=config['d_model'], N=config['N'], h=config['h'],
        d_ff=config['d_ff'], seq_len=config['seq_len'],
        vocab_src=tokenizer_src.get_vocab_size(),
        vocab_tgt=tokenizer_tgt.get_vocab_size(),
        epochs=int(epoch) + 1,
        hours_per_epoch='4.7 hours',
    ))

    print('\nContents:')
    total = 0
    for path in sorted(DEPLOY.iterdir()):
        size = path.stat().st_size
        total += size
        print(f'  {path.name:24s} {size/1e6:8.2f} MB')
    print(f'  {"TOTAL":24s} {total/1e6:8.2f} MB')
    if any(p.stat().st_size > 100e6 for p in DEPLOY.iterdir()):
        print('\nWARNING: a file exceeds GitHub\'s 100MB limit.')


if __name__ == '__main__':
    main()
