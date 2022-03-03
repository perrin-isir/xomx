import numpy as np

aminoacids = ["A", "R", "N", "D", "C", "Q", "E",
              "G", "H", "I", "L", "K", "M", "F",
              "P", "S", "T", "W", "Y", "V"]


def onehot(seq: str, max_length: int) -> np.ndarray:
    result = []
    for aa in seq:
        result.append(1 * (np.asarray(aminoacids) == aa))
    result.append(np.zeros((max_length - len(seq)) * len(aminoacids)))
    return np.hstack(result)


def onehot_inverse(vec: np.ndarray, max_length: int) -> str:
    len_aa = len(aminoacids)
    return_string = ''
    v = np.array(vec)
    for k in range(max_length):
        return_string += aminoacids[v[len_aa * k:len_aa * (k + 1)].argmax()] \
            if v[len_aa * k:len_aa * (k + 1)].max() else ''
    return return_string


def to_float(seq: str) -> float:
    mult = 1
    value = 0
    for char in seq[::-1]:
        value += ((np.array(char) == aminoacids).argmax() + 1) * mult
        mult *= (len(aminoacids) + 1)
    return 1. * value


def to_float_inverse(x: float) -> str:
    base = (len(aminoacids) + 1)
    n = int(x)
    digit_list = []
    if n == 0:
        digit_list = [0]
    while n:
        digit_list.append(int(n % base))
        n //= base
    dl = digit_list[::-1]
    s = ''
    for digit in dl:
        s += aminoacids[digit - 1]
    return s
