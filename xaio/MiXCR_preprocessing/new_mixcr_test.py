# from tools.sequence_analysis import sequence_to_onehot_vec
from xaio.tools.basic_tools import XAIOData

# from xaio.xaio_config import output_dir
from IPython import embed as e
import pandas as pd
import numpy as np
import os
from collections import Counter

# import random
from biotransformers import BioTransformers  # pip install bio-transformers

assert BioTransformers
assert e
# from tools.feature_selection.RFEExtraTrees import RFEExtraTrees


def get_reads(filename):
    seq_read = pd.read_table(
        filename,
        sep=",",
        header=0,
        engine="c",
    ).to_numpy()
    trb_indices = np.where(
        [seq_read[i_, 4].startswith("TRA") for i_ in range(seq_read.shape[0])]
        # [seq_read[i_, 4].startswith(
        #     ("IGL", "TRB", "TRA", "IGK", "IGH")) for i_ in range(seq_read.shape[0])]
    )[0]
    seq_read = np.take(seq_read, trb_indices, axis=0)
    seq_read = np.take(
        seq_read,
        np.where([type(seq_read[i_, 0]) == str for i_ in range(seq_read.shape[0])])[0],
        axis=0,
    )
    seq_read = np.take(seq_read, [0, 9], axis=1)
    return seq_read


def analyze_seq(seq_rd, annotation_, seq_dict=None):
    if seq_dict is None:
        seq_dict = {}
    for i in range(seq_rd.shape[0]):
        # seq_dict.setdefault(seq_rd[i, 1][4:9], {})
        # seq_dict[seq_rd[i, 1][4:9]].setdefault(annotation_, [])
        # if seq_rd[i, 0] not in seq_dict[seq_rd[i, 1][4:9]][annotation_]:
        #     seq_dict[seq_rd[i, 1][4:9]][annotation_].append(seq_rd[i, 0])
        seq_dict.setdefault(seq_rd[i, 1][:], {})
        seq_dict[seq_rd[i, 1][:]].setdefault(annotation_, [])
        if seq_rd[i, 0] not in seq_dict[seq_rd[i, 1][:]][annotation_]:
            seq_dict[seq_rd[i, 1][:]][annotation_].append(seq_rd[i, 0])

    # keys_to_erase = []
    # for key in seq_dict:
    #     totlen = 0
    #     for kk in seq_dict[key]:
    #         totlen += len(seq_dict[key][kk])
    #     if totlen < 2:
    #         keys_to_erase.append(key)
    # for key in keys_to_erase:
    #     del seq_dict[key]


def indiv_rep_seq(seq_rd, annotation_):
    seq_dict = {}
    annot_dict = {}
    nr_seqs = 5
    for elt in seq_rd:
        seq_dict.setdefault(elt[0], []).append(elt[1])
        annot_dict[elt[0]] = annotation_ + str(len(seq_dict[elt[0]]))
        # annot_dict[elt[0]] = annotation_ + str(seq_dict[elt[0]])
        # e()
        # quit()
        # annot_dict[elt[0]] = annotation_
    for key in seq_dict.keys():
        s = np.zeros(100)
        cter = Counter(seq_dict[key]).most_common()[:nr_seqs]
        for i_ in range(len(cter)):
            s += protvec(cter[i_][0])
        s /= nr_seqs
        seq_dict[key] = s
    return seq_dict, annot_dict


if True:
    dico_3mers = {}
    df = pd.read_csv(
        "protVec_100d_3grams.csv", header=None, sep='\\t|"', engine="python"
    ).to_numpy()
    for i in range(df.shape[0]):
        dico_3mers[df[i][1]] = np.array(df[i][2:102], dtype="float32")

    def protvec(aminoseq):
        mers = [aminoseq[i_ : i_ + 3] for i_ in range(len(aminoseq) - 2)]
        s = np.zeros(100)
        for m in mers:
            if m in dico_3mers:
                s += dico_3mers[m]
            else:
                s += dico_3mers["<unk>"]
        return s


data = XAIOData()
data.save_dir = "/home/perrin/Desktop/data/xaio/dataset/MiXCR/subset_new13/"

if not os.path.exists(data.save_dir):
    prefix = (
        "/home/perrin/Desktop/data/"
        + "MiXCR/TCGA_MiXCR/TCGA_MiXCR_NicolasPerrin/Legacy_fileIDs/"
    )
    sdic = {}
    adic = {}

    # for annotation in [
    #     "ACC",
    #     "BLCA",
    #     "BRCA",
    #     "CESC",
    #     "CHOL",
    #     "COAD",
    #     "DLBC",
    #     "ESCA",
    #     "GBM",
    #     "HNSC",
    #     "KICH",
    #     "KIRC",
    #     "KIRP",
    #     "LGG",
    #     "LIHC",
    #     "LUAD",
    #     "LUSC",
    #     "MESO",
    #     "OV",
    #     "PAAD",
    #     "PCPG",
    #     "PRAD",
    #     "READ",
    #     "SARC",
    #     "SKCM",
    #     "STAD",
    #     "TGCT",
    #     "THCA",
    #     "THYM",
    #     "UCEC",
    #     "UCS",
    #     "UVM",
    # ]:
    annot_list = ["STAD", "ACC", "LUAD", "PAAD"]
    # annot_list = ["STAD", "ACC"]
    # annot_list = ["LUAD", "LUSC"]
    for annotation in annot_list:
        # for annotation in ["ACC", "BLCA"]:
        print(annotation)
        gr = get_reads(prefix + "tcga_" + annotation + "_legacy_file_ids.txt")
        sd, ad = indiv_rep_seq(gr, annotation)
        sdic.update(sd)
        adic.update(ad)

    data.sample_ids = np.array(list(sdic.keys()))
    data.sample_annotations = np.empty_like(data.sample_ids)
    for i, s_id in enumerate(data.sample_ids):
        data.sample_annotations[i] = adic[s_id]
    data.compute_sample_indices()
    data.compute_all_annotations()
    data.compute_sample_indices_per_annotation()
    # data.nr_samples = len(data.sample_ids)
    # data.nr_features = 100
    data.feature_names = np.zeros(100)
    darray = np.zeros((data.nr_samples, 100))
    for i in range(data.nr_samples):
        darray[i, :] = sdic[data.sample_ids[i]]
    data.data_array["raw"] = darray
    data.save(["raw"])
    e()
    quit()

if os.path.exists(os.path.join(data.save_dir, "raw_data.bin")):
    data.load(["raw"])
    data.compute_feature_mean_values()
    data.compute_feature_standard_deviations()
    data.compute_normalization("std")
    data.umap_plot("std")
    e()
    quit()
