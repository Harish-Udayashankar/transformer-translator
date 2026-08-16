import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from main_model import build_transformer

DEVICE = torch.device('cpu')
HERE = Path(__file__).parent

st.set_page_config(page_title='English-Spanish Transformer', layout='centered')


st.markdown("""
<style>
  #MainMenu, header[data-testid="stHeader"], footer {visibility: hidden; height: 0;}
  .block-container {padding-top: 3rem; padding-bottom: 4rem; max-width: 46rem;}
  h1 {font-size: 1.6rem !important; font-weight: 600; letter-spacing: -0.01em;}
  .subtitle {color: #6b7280; font-size: 0.9rem; margin: -0.4rem 0 1.6rem 0;}
  .outbox {font-size: 1.15rem; line-height: 1.5; padding: 0.2rem 0;}
  .meta {color: #6b7280; font-size: 0.82rem;}
  .oov {color: #b45309; font-size: 0.82rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_everything():
    from tokenizers import Tokenizer

    meta = json.loads((HERE / 'model_config.json').read_text())
    tokenizer_src = Tokenizer.from_file(str(HERE / meta['tokenizer_src']))
    tokenizer_tgt = Tokenizer.from_file(str(HERE / meta['tokenizer_tgt']))

    model = build_transformer(
        meta['vocab_src'], meta['vocab_tgt'],
        meta['seq_len'], meta['seq_len'],
        d_model=meta['d_model'], N=meta['N'], h=meta['h'], d_ff=meta['d_ff']
    ).to(DEVICE)

    state = torch.load(HERE / 'model.pt', map_location=DEVICE)
    model.load_state_dict({k: v.float() for k, v in state['model_state_dict'].items()})
    model.eval()
    return meta, tokenizer_src, tokenizer_tgt, model


@st.cache_data(show_spinner=False)
def load_examples():
    path = HERE / 'examples.json'
    return json.loads(path.read_text()) if path.exists() else []


@st.cache_data(show_spinner=False)
def load_loss_curve():
    path = HERE / 'loss_curve.csv'
    return pd.read_csv(path) if path.exists() else None


def unknown_words(text, _tokenizer):
    words = [w for w, _ in _tokenizer.pre_tokenizer.pre_tokenize_str(text)]
    return [w for w in words if _tokenizer.token_to_id(w) is None]


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


@torch.no_grad()
def translate(model, text, tokenizer_src, tokenizer_tgt, seq_len):
    ids = tokenizer_src.encode(text).ids
    sos = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
    eos = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
    pad = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    num_padding = seq_len - len(ids) - 2
    if num_padding < 0:
        return None, len(ids)

    source = torch.cat([sos, torch.tensor(ids, dtype=torch.int64), eos,
                        pad.repeat(num_padding)])
    source_mask = (source != pad).unsqueeze(0).unsqueeze(0).int().unsqueeze(0).to(DEVICE)
    source = source.unsqueeze(0).to(DEVICE)

    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(DEVICE)
    while decoder_input.size(1) < seq_len:
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(DEVICE)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        _, next_word = torch.max(model.project(out[:, -1]), dim=1)
        decoder_input = torch.cat(
            [decoder_input,
             torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(DEVICE)],
            dim=1
        )
        if next_word == eos_idx:
            break

    tokens = decoder_input.squeeze(0).detach().cpu().numpy().tolist()
    return tokenizer_tgt.decode(tokens), len(ids)


meta, tokenizer_src, tokenizer_tgt, model = load_everything()
examples = load_examples()

st.markdown('# English to Spanish')
st.markdown(
    f'<div class="subtitle">A {meta["params_millions"]:.0f}M-parameter transformer '
    f'written and trained from scratch in PyTorch, as a learning project. '
    f'It was trained on {meta["train_pairs"]:,} sentence pairs from 19th-century '
    f'novels, so it works best on that kind of language and struggles with '
    f'modern or everyday English.</div>',
    unsafe_allow_html=True
)

if examples:
    labels = [e['en'] for e in examples]

    def _apply_example():
        st.session_state.src_text = st.session_state.example_pick

    if 'src_text' not in st.session_state:
        st.session_state.src_text = labels[0]

    st.selectbox(
        'Sentences from the training corpus (the model is strongest here)',
        labels, key='example_pick', on_change=_apply_example
    )
else:
    if 'src_text' not in st.session_state:
        st.session_state.src_text = 'The house is small.'

st.text_input('Or write your own, then press Enter', key='src_text')
text = st.session_state.src_text

strict = st.checkbox(
    'Only allow words the model knows',
    value=True,
    help='The model has a fixed vocabulary of %s English words. Anything outside '
         'it becomes [UNK] and cannot be translated.' % f"{meta['vocab_src']:,}"
)

unknown = unknown_words(text, tokenizer_src) if text.strip() else []
if unknown:
    st.markdown(
        '<span class="oov">Not in the model\'s vocabulary: '
        + ', '.join(f'<b>{w}</b>' for w in unknown[:8])
        + ('' if len(unknown) <= 8 else f' (+{len(unknown) - 8} more)')
        + '</span>',
        unsafe_allow_html=True
    )
elif text.strip():
    st.markdown('<span class="meta">Every word is in the vocabulary.</span>',
                unsafe_allow_html=True)

blocked = strict and bool(unknown)

if st.button('Translate', disabled=blocked):
    if strict and unknown_words(text, tokenizer_src):
        st.markdown(
            '<span class="oov">That sentence contains words the model does not '
            'know. Uncheck the box above to try anyway.</span>',
            unsafe_allow_html=True
        )
    elif not text.strip():
        st.markdown('<span class="meta">Enter a sentence first.</span>',
                    unsafe_allow_html=True)
    else:
        with st.spinner(''):
            output, num_tokens = translate(
                model, text, tokenizer_src, tokenizer_tgt, meta['seq_len']
            )
        if output is None:
            st.markdown(
                f'<span class="meta">Too long: {num_tokens} tokens, '
                f'limit is {meta["seq_len"] - 2}.</span>', unsafe_allow_html=True
            )
        elif not output.strip():
            st.markdown('<span class="meta">No output for this input.</span>',
                        unsafe_allow_html=True)
        else:
            with st.container(border=True):
                st.markdown(f'<div class="outbox">{output}</div>',
                            unsafe_allow_html=True)

            reference = next((e['es'] for e in examples if e['en'] == text), None)
            if reference:
                st.markdown(
                    f'<span class="meta">Human translation: {reference}</span>',
                    unsafe_allow_html=True
                )

if blocked:
    st.markdown(
        '<span class="meta">Translation is disabled because the sentence contains '
        'unknown words. Uncheck the box above to try anyway.</span>',
        unsafe_allow_html=True
    )

st.write('')

with st.expander('What this is, and what it cannot do'):
    st.markdown(f"""
**What it is.** An encoder-decoder transformer implemented from scratch --
multi-head attention, positional encoding and the encoder/decoder stacks are
hand-written in PyTorch rather than taken from `nn.Transformer`. It uses no
pretrained weights. The point of the project was to understand the architecture
by building it, not to compete with production translation systems.

**Where it falls short, and why:**

- **Narrow domain.** Training data was {meta['train_pairs']:,} sentence pairs from
  19th-century novels. Modern topics (phones, email, meetings) are outside
  everything it has seen, and it will sometimes reach for a memorized phrase from
  the books instead.
- **Small dataset.** Production translation models train on millions of sentence
  pairs. At this scale a model memorizes more than it generalizes.
- **Undertrained.** Final training loss was {meta['final_loss']:.2f}; the loss was
  still falling when training stopped. Well-trained systems reach roughly half that.
- **Word-level vocabulary.** Words are atomic units, so anything outside the
  {meta['vocab_src']:,}-word vocabulary is unrepresentable. A subword tokenizer
  such as BPE would handle unseen words by splitting them into familiar pieces.
- **Greedy decoding.** It commits to the highest-probability word at each step with
  no ability to reconsider. Beam search would explore several candidates.

**What it does get right:** inverted question marks, verb conjugation, article and
adjective agreement, and short declarative sentences.
    """)

with st.expander('Model details'):
    left, right = st.columns(2)
    left.markdown(f"""
**Architecture**

Model dimension `{meta['d_model']}`
Layers `{meta['N']}`
Attention heads `{meta['h']}`
Feed-forward `{meta['d_ff']}`
Max length `{meta['seq_len']}`
    """)
    right.markdown(f"""
**Training**

Parameters `{meta['params_millions']:.1f}M`
Epochs `{meta['epochs_trained']}`
Final loss `{meta['final_loss']:.2f}`
Vocabulary `{meta['vocab_src']:,} / {meta['vocab_tgt']:,}`
Hardware `CPU`
    """)

curve = load_loss_curve()
if curve is not None:
    st.markdown('<span class="meta">Training loss</span>', unsafe_allow_html=True)
    st.line_chart(curve.set_index('step')[['train_loss']], height=240)
