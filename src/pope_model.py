"""Pope et al.'s transformer, rebuilt for inference: cohorts, token ids, ensemble predictions, encoder features.

Input construction follows reference_code/.../finetune_discriminator/dataset.py:
  - each sample's 512 (id, count) rows are sorted with np.argsort(count)[::-1] (descending, Pope's exact call,
    so tied counts keep Pope's tie order), then [CLS] is prepended -> 513 positions;
  - rows with count 0 become the padding token; attention mask = 0 on padding;
  - tokens are looked up in a 26,729-row table: vocab_embeddings.npy + mask + CLS (zeros) + padding (zeros),
    and fed as `inputs_embeds` (the checkpoints' own word_embeddings matrix is an unused random init).

The model never sees the counts: electra_discriminator.py only uses them for the attention mask. Abundance
reaches the model through token ORDER alone, which is what E4 perturbs.

Known gap: fine-tuning ran with --freeze_opt 2, which leaves the CLS row of the lookup table trainable, and that
table was saved separately (*_embed) and not distributed. CLS is therefore zeros here (its init value).
"""
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import ElectraForSequenceClassification, ElectraModel

from tokens import CLS, PAD, SEQ_LEN

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def path(key):
    return ROOT / CFG["paths"][key]


def device():
    d = CFG["e4"]["device"]
    if d == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(d)


def load_cohort(name):
    """(n, 512, 2) float array of [vocab id, read count] and int labels (1 = IBD)."""
    c = CFG["cohorts"][name]
    arr = np.load(path("cohorts_dir") / c["samples"])
    y = np.load(path("cohorts_dir") / c["labels"]).ravel().astype(int)
    assert len(arr) == len(y), name
    return arr, y


def pope_order(arr):
    """Token ids (n, 513) and counts (n, 513) in Pope's order: [CLS] + taxa by descending count + padding."""
    ids = np.full((len(arr), SEQ_LEN), PAD, dtype=np.int64)
    counts = np.zeros((len(arr), SEQ_LEN))
    for i, s in enumerate(arr):
        s = s[np.argsort(s[:, 1])[::-1]]
        s = s[s[:, 1] > 0]
        ids[i, 0], counts[i, 0] = CLS, 1
        ids[i, 1:1 + len(s)] = s[:, 0].astype(np.int64)
        counts[i, 1:1 + len(s)] = s[:, 1]
    return ids, counts


def pope_test_dropout(ids, rng, p):
    """dataset.py's input dropout (applied at test time too): drop each non-CLS position with prob p,
    shift the survivors left, pad the end."""
    out = np.full_like(ids, PAD)
    for i, row in enumerate(ids):
        keep = rng.random(len(row)) > p
        keep[0] = True
        kept = row[keep & (row != PAD)]
        out[i, :len(kept)] = kept
    return out


def input_table():
    """The 26,729-row lookup table dataset.py builds (mask row = column-wise min, because of its amin/amin)."""
    v = np.load(path("vocab_embeddings"))
    return torch.tensor(np.vstack([v, v.min(0), np.zeros(v.shape[1]), np.zeros(v.shape[1])]), dtype=torch.float32)


def ensemble_dirs():
    return sorted(path("ensemble_dir").glob("run*_epoch0_disc"), key=lambda p: int(p.name[3:].split("_")[0]))


def load_classifier(ckpt_dir):
    return ElectraForSequenceClassification.from_pretrained(ckpt_dir).eval().to(device())


def load_encoder(ckpt_dir=None):
    ckpt_dir = ckpt_dir or path("pretrained_dir") / CFG["e4"]["encoder_checkpoint"]
    return ElectraModel.from_pretrained(ckpt_dir).eval().to(device())


def _batches(ids, table):
    bs = CFG["e4"]["batch_size"]
    for s in range(0, len(ids), bs):
        b = torch.from_numpy(ids[s:s + bs])
        yield table[b].to(device()), (b != PAD).long().to(device())


@torch.no_grad()
def predict(model, ids, table):
    """P(IBD) for each sample from one fine-tuned classifier."""
    out = [torch.softmax(model(inputs_embeds=x, attention_mask=m).logits, 1)[:, 1].cpu() for x, m in _batches(ids, table)]
    return torch.cat(out).numpy()


def predict_ensemble(models, ids, table):
    """(n_models, n) per-model P(IBD); Pope scores the plain mean over models (begin.py, ensembled cross-gen)."""
    return np.stack([predict(m, ids, table) for m in models])


@torch.no_grad()
def all_layer_features(body, ids, table):
    """[CLS] vector and mean over non-padding tokens from EVERY hidden state: (n, n_layers + 1, 200) each.

    `body` is an ElectraModel (a pretrained checkpoint, or `.electra` of a fine-tuned classifier). Index 0 is the
    encoder input (embeddings after embeddings_project), 1..5 the encoder layers. Returned as float32.
    """
    cls, mean = [], []
    for x, m in _batches(ids, table):
        hs = torch.stack(body(inputs_embeds=x, attention_mask=m, output_hidden_states=True).hidden_states, 1)
        w = m[:, None, :, None].float()
        cls.append(hs[:, :, 0].cpu())
        mean.append(((hs * w).sum(2) / w.sum(2)).cpu())
    return torch.cat(cls).numpy(), torch.cat(mean).numpy()


@torch.no_grad()
def encoder_features(encoder, ids, table, layer=-1):
    """CLS vector and mean over non-padding tokens from one hidden layer: (n, 200) each."""
    cls, mean = [], []
    for x, m in _batches(ids, table):
        h = encoder(inputs_embeds=x, attention_mask=m, output_hidden_states=True).hidden_states[layer]
        w = m.unsqueeze(-1).float()
        cls.append(h[:, 0].cpu())
        mean.append(((h * w).sum(1) / w.sum(1)).cpu())
    return torch.cat(cls).numpy(), torch.cat(mean).numpy()
