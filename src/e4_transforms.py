import numpy as np

from tokens import PAD


def shuffle_tokens(ids, rng):
    """E4a: put each row's taxa in a uniformly random order.

    """
    out = ids.copy()

    for i in range(len(out)):
        n_taxa = (out[i, 1 : ] != PAD).sum()
        out[i,1 : 1 + n_taxa] = rng.permutation(out[i,1 : 1 + n_taxa])

    return out

def shuffle_within_ties(ids, counts, rng):
    """E4a control: shuffle only among taxa with EQUAL counts; every other taxon keeps its position.

    """
    out = ids.copy()
    def process_row(row_ids, row_counts) :
        n_taxa = (row_ids[1:] != PAD).sum()
        end_taxa = 1 + n_taxa

        start = 1
        while(start < end_taxa) :
            end = start

            while end < end_taxa and row_counts[end] == row_counts[start] :
                end+=1

            if end - start > 1 :
                row_ids[start : end] = rng.permutation(row_ids[start : end])

            start = end

    for i in range(len(out)):
        process_row(out[i], counts[i])

    return out


def reverse_order(ids):
    """E4a: least abundant taxon first, most abundant last (CLS still first, PAD still last).

    """
    out = ids.copy()

    for i in range(len(out)):
        n_taxa = (out[i, 1 : ] != PAD).sum()
        out[i,1 : 1 + n_taxa] = out[i,1 : 1 + n_taxa][::-1]

    return out

def top_k(ids, k):
    """E4d: keep CLS and the first k taxa (the k most abundant); every later position becomes PAD.

    """
    out = ids.copy()
    out[... , k +1 : ] = PAD
    return out



def random_k(ids, k, rng):
    """E4d control: keep CLS and k taxa chosen AT RANDOM (not the k most abundant); every other taxon -> PAD.

    """
    out = ids.copy()
    for i in range(len(out)):
        n_taxa = (out[i, 1 : ] != PAD).sum()
        if n_taxa > k:
            chosen_idx = rng.choice(n_taxa, size=k, replace=False)
            chosen_idx = np.sort(chosen_idx)
            kept_idx = out[i,1 : 1 + n_taxa][chosen_idx]
            out[i,1 : 1 + k] = kept_idx
            out[i,1 + k :] = PAD

    return out
