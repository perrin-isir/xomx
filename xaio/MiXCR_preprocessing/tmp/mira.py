# from tools.sequence_analysis import sequence_to_onehot_vec
from xaio.tools.basic_tools import XAIOData

# from xaio.xaio_config import output_dir
from IPython import embed as e
import pandas as pd
import numpy as np
import os
import re

# from collections import Counter

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
    seq_read = np.take(seq_read, [0, 3], axis=1)
    nr_sples = 1000
    seq_read = np.take(
        seq_read, np.random.choice(len(seq_read), nr_sples, replace=False), axis=0
    )
    for i_ in range(len(seq_read)):
        seq_read[i_][0] = re.search("^[^+]*", seq_read[i_][0])[0]
    return seq_read


if True:
    dico_3mers = {}
    df = pd.read_csv(
        "../protVec_100d_3grams.csv", header=None, sep='\\t|"', engine="python"
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
data.save_dir = "/home/perrin/Desktop/data/xaio/dataset/MiXCR/MIRA/results"

if not os.path.exists(data.save_dir):
    sdic = {}
    adic = {}
    gr = get_reads(
        "/home/perrin/Desktop/data/xaio/dataset/MiXCR/MIRA/"
        + "ImmuneCODE/Release002.1/peptide-detail-ci.csv"
    )

    data.sample_ids = gr[:, 0]
    data.sample_annotations = gr[:, 1]
    data.compute_sample_indices()
    data.compute_all_annotations()
    data.compute_sample_indices_per_annotation()

    if True:
        darray = np.zeros((data.nr_samples, 100))
        for i in range(data.nr_samples):
            darray[i, :] = protvec(data.sample_ids[i])
    else:
        bio_trans = BioTransformers(backend="esm1b_t33_650M_UR50S")
        embeddings = bio_trans.compute_embeddings(gr[:, 0], pool_mode=("cls", "mean"))
        cls_emb = embeddings["cls"]
        # mean_emb = embeddings["mean"]
        darray = cls_emb

    data.data_array["raw"] = darray
    data.save(["raw"])
    e()
    quit()

if os.path.exists(os.path.join(data.save_dir, "raw_data.bin")):
    data.load(["raw"])
    data.compute_feature_mean_values()
    data.compute_feature_standard_deviations()
    data.compute_normalization("std")
    data.umap_plot("raw")
    e()
    quit()
