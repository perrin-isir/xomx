import numpy as np
from tools.basic_tools import RNASeqData
from xaio_config import output_dir


def loadscRNASeq(normalization=""):
    data = RNASeqData()
    data.save_dir = output_dir + "/dataset/scRNASeq/"
    data.nr_features = np.load(
        data.save_dir + "nr_transcripts.npy", allow_pickle=True
    ).item()
    data.nr_samples = np.load(
        data.save_dir + "nr_samples.npy", allow_pickle=True
    ).item()
    data.feature_names = np.load(data.save_dir + "transcripts.npy", allow_pickle=True)
    data.feature_mean_values = np.load(
        data.save_dir + "mean_expressions.npy", allow_pickle=True
    )
    data.feature_standard_deviations = np.load(
        data.save_dir + "std_expressions.npy", allow_pickle=True
    )
    data.feature_shortnames_ref = np.load(
        data.save_dir + "gene_dict.npy", allow_pickle=True
    ).item()
    if normalization == "log":
        data.normalization_type = "log"
        data.data = np.array(
            np.memmap(
                data.save_dir + "lognorm_data.bin",
                dtype="float32",
                mode="r",
                shape=(data.nr_features, data.nr_samples),
            )
        ).transpose()
        data.epsilon_shift = np.load(
            data.save_dir + "epsilon_shift.npy", allow_pickle=True
        ).item()
        data.maxlog = np.load(data.save_dir + "maxlog.npy", allow_pickle=True).item()
    elif normalization == "raw":
        data.normalization_type = "raw"
        data.data = np.array(
            np.memmap(
                data.save_dir + "raw_data.bin",
                dtype="float32",
                mode="r",
                shape=(data.nr_features, data.nr_samples),
            )
        ).transpose()
    else:
        data.normalization_type = "mean_std"
        data.data = np.array(
            np.memmap(
                data.save_dir + "data.bin",
                dtype="float32",
                mode="r",
                shape=(data.nr_features, data.nr_samples),
            )
        ).transpose()
    return data
