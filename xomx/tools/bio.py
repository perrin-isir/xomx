import numpy as np
from pandas import DataFrame
from scipy.stats import entropy

aminoacids = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]


def onehot(seq: str, max_length: int) -> np.ndarray:
    result = []
    for aa in seq:
        result.append(1 * (np.asarray(aminoacids) == aa))
    result.append(np.zeros((max_length - len(seq)) * len(aminoacids)))
    return np.hstack(result)


def onehot_inverse(vec: np.ndarray, max_length: int) -> str:
    len_aa = len(aminoacids)
    return_string = ""
    v = np.array(vec)
    for k in range(max_length):
        return_string += (
            aminoacids[v[len_aa * k : len_aa * (k + 1)].argmax()]
            if v[len_aa * k : len_aa * (k + 1)].max()
            else ""
        )
    return return_string


def to_float(seq: str) -> float:
    mult = 1
    value = 0
    for char in seq[::-1]:
        value += ((np.array(char) == aminoacids).argmax() + 1) * mult
        mult *= len(aminoacids) + 1
    return 1.0 * value


def to_float_inverse(x: float) -> str:
    base = len(aminoacids) + 1
    n = int(x)
    digit_list = []
    if n == 0:
        digit_list = [0]
    while n:
        digit_list.append(int(n % base))
        n //= base
    dl = digit_list[::-1]
    s = ""
    for digit in dl:
        s += aminoacids[digit - 1]
    return s


def compute_logomaker_df(adata, indices=None, fixed_length: int = None):
    """
    The sample names (adata.obs_names) must be strings made of amino acid characters.
    The list of allowed characters is stored in the variable aminoacids.
    """
    if indices is None:
        indices = np.arange(len(adata.obs_names))
    if fixed_length is None:
        pos_list = np.arange(max([len(adata.obs_names[idx]) for idx in indices]))
        total_size = len(indices)
    else:
        pos_list = np.arange(fixed_length)
        total_size = 0
        for idx in indices:
            if len(adata.obs_names[idx]) == fixed_length:
                total_size += 1
    if total_size == 0:
        raise ValueError("Cannot compute logo on an empty set.")
    probability_matrix = np.zeros((len(pos_list), len(aminoacids)))

    for position in pos_list:
        counts = {}
        for aa in aminoacids:
            counts[aa] = 0
        for idx in indices:
            if fixed_length is None:
                if position < len(adata.obs_names[idx]):
                    counts[adata.obs_names[idx][position]] += 1
            else:
                if len(adata.obs_names[idx]) == fixed_length:
                    counts[adata.obs_names[idx][position]] += 1
        for k in range(len(aminoacids)):
            probability_matrix[position, k] = counts[aminoacids[k]] / total_size
    # from probabilities to bits:
    max_entropy = -np.log2(1 / len(aminoacids))
    for position in pos_list:
        pos_entropy = max_entropy - entropy(
            1e-10 + probability_matrix[position, :], base=2
        )
        probability_matrix[position, :] *= pos_entropy
    dico = {"pos": pos_list}
    for k in range(len(aminoacids)):
        dico[aminoacids[k]] = probability_matrix[:, k]
    df_ = DataFrame(dico)
    df_ = df_.set_index("pos")
    return df_
