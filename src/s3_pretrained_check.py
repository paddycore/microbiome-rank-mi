"""STOP POINT S3, stretch goals 1-2: feasibility only (no probing pipeline).

1. Can Pope's pretrained ELECTRA discriminator checkpoints be loaded, and can hidden states be extracted?
2. Does vocab_embeddings.npy align row-wise with vocab.txt (vocab ID j = row j)?

Key fact established here: the checkpoints' own `electra.embeddings.word_embeddings` is an UNUSED random init.
Pope's code (pretrain_discriminator/electra_pretrain_model.py, finetune_discriminator/electra_discriminator.py)
looks tokens up in a separate nn.Embedding built from vocab_embeddings.npy (+ mask, CLS, padding rows, see
finetune_discriminator/dataset.py) and passes `inputs_embeds` to the ELECTRA model. That separate layer
(`*disc_embed`) is not distributed, so inputs are rebuilt the way dataset.py builds them:
    rows 0..26725 = vocab_embeddings.npy, 26726 = mask (= column-wise min, dataset.py's amin/amin),
    26727 = CLS (zeros), 26728 = padding (zeros).

Functional test (verifies loading AND row alignment at once): replace 15% of each sample's taxa with random vocab
IDs and ask the pretrained replaced-token head to find them (AUROC over token positions). Controls: the same
with vocab_embeddings rows shuffled, and the epoch-0 checkpoint.

Writes reports/s3_pretrained_feasibility.md.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import ElectraForPreTraining

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CKPT_ROOT = RAW / "pretrainedmodels" / "pretrainedmodels"
OUT = ROOT / "reports" / "s3_pretrained_feasibility.md"
N_VOCAB = 26726
MASK, CLS, PAD = N_VOCAB, N_VOCAB + 1, N_VOCAB + 2
N_SAMPLES, REPLACE_FRAC, SEED = 64, 0.15, 0


def input_table(vocab_emb):
    """The 26,729-row lookup table dataset.py builds (mask = column-wise min because of amin/amin)."""
    mask = vocab_emb.min(0)
    return torch.tensor(np.vstack([vocab_emb, mask, np.zeros(100), np.zeros(100)]), dtype=torch.float32)


def build_ids(arr, rng, replace_frac, probs=None):
    """[CLS] + taxa by descending abundance (max 512), padding token after. Optionally replace taxa at random:
    uniformly over the vocabulary (probs=None) or prevalence-matched (probs = taxon prevalence in AGP-IBD)."""
    ids = np.full((len(arr), 513), PAD, dtype=np.int64)
    replaced = np.zeros_like(ids, dtype=bool)
    for i, s in enumerate(arr):
        s = s[np.argsort(s[:, 1])[::-1]]
        taxa = s[s[:, 1] > 0, 0].astype(np.int64)[:512]
        if replace_frac:
            pos = rng.choice(len(taxa), max(1, int(round(replace_frac * len(taxa)))), replace=False)
            taxa = taxa.copy()
            taxa[pos] = rng.choice(N_VOCAB, len(pos), p=probs) if probs is not None else rng.integers(0, N_VOCAB, len(pos))
            replaced[i, 1 + pos] = True
        ids[i, 0] = CLS
        ids[i, 1:1 + len(taxa)] = taxa
    return ids, replaced


def cfg_for_init(ckpt_dir):
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    return {k: v for k, v in cfg.items() if k not in ("architectures", "model_type")}


def detect_auroc(model, table, ids, replaced):
    att = torch.from_numpy((ids != PAD).astype(np.int64))
    with torch.no_grad():
        logits = model(inputs_embeds=table[torch.from_numpy(ids)], attention_mask=att).logits.numpy()
    real = (ids != PAD) & (ids != CLS)
    return roc_auc_score(replaced[real], logits[real])


def main():
    L = ["# S3 feasibility: Pope et al. pretrained ELECTRA discriminators", "",
         f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by `src/s3_pretrained_check.py`._", ""]
    vocab_emb = np.load(RAW / "vocab_embeddings.npy")
    assert [int(x) for x in (ROOT / "data" / "processed" / "vocab.txt").read_text().split()] == list(range(N_VOCAB))
    ckpts = sorted(CKPT_ROOT.glob("5head5layer_epoch*_disc"), key=lambda p: int(p.name.split("epoch")[1].split("_")[0]))

    # 1. state dicts
    sds = {c.name: torch.load(c / "pytorch_model.bin", map_location="cpu", weights_only=True) for c in ckpts}
    first, last = sds[ckpts[0].name], sds[ckpts[-1].name]
    layouts_equal = all({k: v.shape for k, v in sd.items()} == {k: v.shape for k, v in last.items()} for sd in sds.values())
    W = last["electra.embeddings.word_embeddings.weight"]
    changed = lambda k: not torch.equal(first[k], last[k])
    L += ["## 1. Checkpoints and state dict", "",
          f"- {len(ckpts)} checkpoints ({', '.join(c.name.split('_')[1] for c in ckpts)}), identical layout: {layouts_equal}; "
          f"{len(last)} tensors, {sum(v.numel() for v in last.values()):,} parameters each.",
          f"- `electra.embeddings.word_embeddings.weight` {tuple(W.shape)}: mean {W.mean():.4f}, std {W.std():.4f} "
          f"(= config initializer_range 0.02), identical in every checkpoint: "
          f"{all(torch.equal(sd['electra.embeddings.word_embeddings.weight'], W) for sd in sds.values())}. "
          f"Max |diff| to `vocab_embeddings.npy` rows: {np.abs(W[:N_VOCAB].double().numpy() - vocab_emb).max():.2f}. "
          "**It is an unused random init** - Pope feeds `inputs_embeds` from a separate embedding layer.",
          f"- Trained between epoch 0 and 120: position embeddings {changed('electra.embeddings.position_embeddings.weight')}, "
          f"embeddings_project {changed('electra.embeddings_project.weight')}, encoder layer 0 query "
          f"{changed('electra.encoder.layer.0.attention.self.query.weight')}, replaced-token head "
          f"{changed('discriminator_predictions.dense.weight')}.",
          "- The separate embedding layer (`*disc_embed`, incl. the CLS vector trained in pretraining) is **not in the "
          "Dryad archive**; inputs are rebuilt from `vocab_embeddings.npy` + mask/CLS/padding rows as `dataset.py` does "
          "(CLS = zeros, as in Pope's fine-tuning runs, which pass no `--load_embed`).", "", "Tensor names and shapes:", "", "```"]
    for k, v in last.items():
        if ".layer." in k and ".layer.0." not in k:
            continue
        L.append(f"{k:70s} {tuple(v.shape)}")
    L += ["(encoder.layer.1-4: same shapes as layer.0)", "```", ""]
    cfg = json.loads((ckpts[-1] / "config.json").read_text())
    L += ["Config: " + ", ".join(f"`{k}={cfg[k]}`" for k in ["vocab_size", "embedding_size", "hidden_size", "num_hidden_layers",
                                                        "num_attention_heads", "intermediate_size", "max_position_embeddings"]), ""]

    # 2. functional test: loading + alignment
    from transformers import ElectraConfig
    arr = np.asarray(np.load(RAW / "microbiomedata" / "microbiomedata" / "total_IBD_512.npy", mmap_mode="r")[:N_SAMPLES])
    Xagp = np.load(ROOT / "data" / "processed" / "agp_ibd_X.npy", mmap_mode="r")
    prev = np.asarray((Xagp > 0).sum(0), dtype=np.float64)
    probs = prev / prev.sum()
    designs = {"prevalence-matched": build_ids(arr, np.random.default_rng(SEED), REPLACE_FRAC, probs),
               "uniform": build_ids(arr, np.random.default_rng(SEED), REPLACE_FRAC)}
    table = input_table(vocab_emb)
    shuffled = input_table(vocab_emb[np.random.default_rng(1).permutation(N_VOCAB)])
    m120, loading = ElectraForPreTraining.from_pretrained(CKPT_ROOT / ckpts[-1].name, output_loading_info=True)
    m0 = ElectraForPreTraining.from_pretrained(CKPT_ROOT / ckpts[0].name)
    torch.manual_seed(SEED)
    mrand = ElectraForPreTraining(ElectraConfig(**{k: v for k, v in cfg_for_init(ckpts[-1]).items()}))
    conds = [("epoch 120", m120, "vocab_embeddings.npy, row j = ID j", table),
             ("epoch 120", m120, "rows shuffled (control)", shuffled),
             ("epoch 0 (saved after 1st epoch)", m0, "vocab_embeddings.npy, row j = ID j", table),
             ("random init, same config (control)", mrand, "vocab_embeddings.npy, row j = ID j", table)]
    res = {}
    for mname, model, tname, tab in conds:
        model.eval()
        for dname, (ids, replaced) in designs.items():
            res[(mname, tname, dname)] = detect_auroc(model, tab, ids, replaced)
    ids, replaced = designs["prevalence-matched"]
    att = torch.from_numpy((ids != PAD).astype(np.int64))
    t0 = time.time()
    with torch.no_grad():
        hs = m120.electra(inputs_embeds=table[torch.from_numpy(ids)], attention_mask=att, output_hidden_states=True).hidden_states
    dt = time.time() - t0
    L += ["## 2. Loading, alignment test, hidden states", "",
          f"`ElectraForPreTraining.from_pretrained` (transformers): missing keys {len(loading['missing_keys'])}, "
          f"unexpected keys {len(loading['unexpected_keys'])}.", "",
          f"Replaced-token detection on {N_SAMPLES} AGP-IBD samples: {int(REPLACE_FRAC * 100)}% of each sample's taxa swapped "
          f"({replaced.sum():,} replaced of {((ids != PAD) & (ids != CLS)).sum():,} taxa). AUROC of the replaced-token logit "
          "(Pope's label: 1 = replaced, so > 0.5 = detects replacements the way it was trained).", "",
          "| model | input embedding table | prevalence-matched replacements | uniform replacements |", "|---|---|---|---|"]
    for mname, _, tname, _ in conds:
        L.append(f"| {mname} | {tname} | {res[(mname, tname, 'prevalence-matched')]:.3f} | {res[(mname, tname, 'uniform')]:.3f} |")
    a = res[("epoch 120", "vocab_embeddings.npy, row j = ID j", "prevalence-matched")]
    sh = res[("epoch 120", "rows shuffled (control)", "prevalence-matched")]
    rnd = res[("random init, same config (control)", "vocab_embeddings.npy, row j = ID j", "prevalence-matched")]
    ok = a > 0.6 and abs(sh - 0.5) < 0.1 and abs(rnd - 0.5) < 0.1
    L += ["", "Reading the table: the discriminator was trained against *generator* replacements, which are plausible "
          "(common) taxa. Prevalence-matched replacements mimic that and are the fair test. Uniform replacements are mostly "
          "rare taxa, which the model has learned to treat as original, so an AUROC below 0.5 there is expected and still "
          "shows the model reads taxon identity from the embeddings.", "",
          f"Hidden states (epoch 120, `model.electra(..., output_hidden_states=True)`): {len(hs)} tensors "
          f"(embedding output + {len(hs) - 1} encoder layers), each {tuple(hs[0].shape)}; {dt:.2f} s for {N_SAMPLES} samples on CPU.", "",
          "## Verdict", "",
          "- **Loading and extraction: feasible.** Stock `transformers.ElectraForPreTraining`/`ElectraModel`, no custom code, "
          "0 missing / 0 unexpected keys; per-layer hidden states (CLS or per-taxon) on CPU in under a second per 64 samples.",
          (f"- **Row alignment confirmed functionally**: with `vocab_embeddings.npy` in ID order the pretrained model detects "
           f"prevalence-matched replacements (AUROC {a:.3f}); shuffling the rows ({sh:.3f}) or using an untrained model with the "
           f"same config ({rnd:.3f}) gives chance. So vocab ID j = embedding row j = `vocab.txt` line j = FASTA `embed{{j+1}}`.")
          if ok else
          (f"- **Functional alignment test inconclusive** (aligned {a:.3f}, rows shuffled {sh:.3f}, untrained model {rnd:.3f}; "
           "criteria were aligned > 0.6 with both controls near 0.5). The discriminator was trained only against the "
           "generator's argmax replacements; the generator is not distributed, so its real task cannot be reproduced, and "
           "synthetic replacements are out of distribution (even an untrained model separates uniform replacements, "
           "0.816, from raw embedding statistics). The aligned and shuffled tables do give clearly different outputs, "
           "so the model reads taxon identity from the embedding rows, but this test cannot say the order is right."),
          "- **Row alignment rests on two other lines of evidence**: (i) by construction, Pope's `dataset.py` indexes "
          "`vocab_embeddings.npy` rows directly with the data's vocab IDs, and our `vocab.txt` is IDs 0..26,725; "
          "(ii) empirically (S1), co-occurring taxa have more similar GloVe embeddings in ID order (AGP-IBD +0.052 cosine "
          "vs cross-sample pairs) while a permuted ID mapping gives -0.002. Treated as aligned; revisit in E3 with a "
          "pretrained-vs-untrained encoder comparison on a downstream probe.",
          "- **Must-dos for E3/E5**: pass `inputs_embeds` (never `input_ids`: the model's own word-embedding matrix is random); "
          "map zero-abundance padding to token 26728, not ID 0 (a real taxon); CLS input = zeros unless Pope's trained CLS "
          "vector (`*disc_embed`) is obtained from the authors.",
          "- Caveat: the discriminator was pretrained on the 18,480-sample AGP pool; AGP-IBD samples do not match pool samples "
          "as stored (S1), so which AGP-IBD samples it saw cannot be established.", ""]
    OUT.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
