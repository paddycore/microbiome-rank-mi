import numpy as np

from tokens import PAD


def shuffle_tokens(ids, rng):
    """E4a: put each row's taxa in a uniformly random order.

    """
    out = ids.copy()

    if out.ndim == 1:
        n_taxa = (out[1:] != PAD).sum()
        out[1:1 + n_taxa] = rng.permutation(out[1:1 + n_taxa])
        return out

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

    if out.ndim == 1 : 
        process_row(out, counts)

    else :
        for i in range(len(out)):
            process_row(out[i], counts[i])
    
    return out


def reverse_order(ids):
    """E4a: least abundant taxon first, most abundant last (CLS still first, PAD still last).

    """
    out = ids.copy()

    if out.ndim == 1:
        n_taxa = (out[1:] != PAD).sum()
        out[1:1 + n_taxa] = out[1:1 + n_taxa][::-1]
        return out

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

