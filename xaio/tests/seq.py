# from tools.sequence_analysis import sequence_to_onehot
import pandas as pd
import numpy as np
from IPython import embed as e

assert e


def kmers(seq, n):
    if len(seq) < n:
        return []
    kms = []
    for i in range(len(seq) - n + 1):
        kms.append(seq[i : i + n])
    return kms


def get_reads(filename):
    seq_read = pd.read_table(
        filename,
        sep=",",
        header=0,
        engine="c",
    ).to_numpy()
    trb_indices = np.where(
        [seq_read[i, 4].startswith("TRB") for i in range(seq_read.shape[0])]
    )[0]
    seq_read = np.take(seq_read, trb_indices, axis=0)
    seq_read = np.take(
        seq_read,
        np.where([type(seq_read[i_, 0]) == str for i_ in range(seq_read.shape[0])])[0],
        axis=0,
    )
    seq_read = np.take(seq_read, [0, 9], axis=1)
    return seq_read


def analyze_seq(seq_rd):
    mean_seq_len = 0
    for seq in seq_rd[:, 1]:
        mean_seq_len += len(seq)
    mean_seq_len = mean_seq_len / len(seq_rd)

    kmers_dict = {}
    for seq in seq_rd[:, 1]:
        for km in kmers(seq, 5):
            kmers_dict.setdefault(km, 0)
            kmers_dict[km] += 1

    # sorted_keys = sorted(kmers_dict, key=kmers_dict.get)
    # print(sorted_keys[-20:][::-1])

    seq_dict = {}
    for i in range(seq_rd.shape[0]):
        seq_dict.setdefault(seq_rd[i, 0], [])
        seq_dict[seq_rd[i, 0]].append(seq_rd[i, 1])

    for k in kmers_dict:
        kmers_dict[k] = kmers_dict[k] / len(seq_rd)

    mean_nr_seqs = 0.0
    for k in seq_dict:
        mean_nr_seqs = mean_nr_seqs + len(seq_dict[k])
    mean_nr_seqs = mean_nr_seqs / len(seq_dict)

    # e()
    # quit()

    seq_dict = {}
    for i in range(seq_rd.shape[0]):
        seq_dict.setdefault(seq_rd[i, 1], [])
        seq_dict[seq_rd[i, 1]].append(seq_rd[i, 0])

    e()
    quit()

    return kmers_dict


# seqr = get_reads(
#     "/home/perrin/Desktop/data/"
#     + "MiXCR/TCGA_MiXCR/TCGA_MiXCR_NicolasPerrin/Legacy_fileIDs/"
#     +
#     "tcga_ACC_legacy_file_ids.txt"
#     # "tcga_DLBC_legacy_file_ids.txt"
#     # "tcga_BLCA_legacy_file_ids.txt"
#     # "tcga_LUSC_legacy_file_ids.txt"
#     # "tcga_LUAD_legacy_file_ids.txt"
#     # "tcga_STAD_legacy_file_ids.txt"
#     # "tcga_THCA_legacy_file_ids.txt"
#     # "tcga_OV_legacy_file_ids.txt"
#     # "tcga_GBM_legacy_file_ids.txt"
# )
prefix = (
    "/home/perrin/Desktop/data/"
    + "MiXCR/TCGA_MiXCR/TCGA_MiXCR_NicolasPerrin/Legacy_fileIDs/"
)
km_d1 = analyze_seq(get_reads(prefix + "tcga_STAD_legacy_file_ids.txt"))
km_d2 = analyze_seq(get_reads(prefix + "tcga_DLBC_legacy_file_ids.txt"))

difs = []
keys = []
for key in km_d1:
    if key in km_d2:
        dif = np.abs(km_d1[key] - km_d2[key])
    else:
        dif = km_d1[key]
    difs.append(dif)
    keys.append(key)

print(np.argsort(difs)[::-1][:10])

t = np.argsort(difs)[::-1][:20]

s = np.argsort(list(km_d1.values()))[::-1][:20]
s2 = list(km_d1.keys())
print([s2[i] for i in s])
e()
