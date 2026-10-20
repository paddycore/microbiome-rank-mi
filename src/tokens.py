"""Special token ids of Pope et al.'s 26,729-row lookup table (finetune_discriminator/dataset.py).

Kept free of heavy imports so tests that only need the constants stay fast.
"""
N_VOCAB = 26726
MASK, CLS, PAD = N_VOCAB, N_VOCAB + 1, N_VOCAB + 2
SEQ_LEN = 513  # [CLS] + up to 512 taxa
