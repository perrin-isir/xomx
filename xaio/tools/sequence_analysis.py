import numpy as np
from IPython import embed as e

assert e


def sequence_to_onehot(sequence):
    """Maps the given sequence into a one-hot encoded matrix."""
    mapping = {aa: i for i, aa in enumerate("ARNDCQEGHILKMFPSTWYVX")}
    num_entries = max(mapping.values()) + 1
    one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)

    for aa_index, aa_type in enumerate(sequence):
        aa_id = mapping[aa_type]
        one_hot_arr[aa_index, aa_id] = 1

    return one_hot_arr


def sequence_to_onehot_vec(sequence, maxlength):
    """Maps the given sequence into a one-hot encoded matrix."""
    mapping = {aa: i for i, aa in enumerate("ARNDCQEGHILKMFPSTWYVX")}
    num_entries = max(mapping.values()) + 1
    one_hot_arr = np.zeros(maxlength * num_entries, dtype=np.int32)

    for aa_index, aa_type in enumerate(sequence):
        aa_id = mapping[aa_type]
        one_hot_arr[aa_index * num_entries + aa_id] = 1

    return one_hot_arr
