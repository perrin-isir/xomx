import os
import numpy as np
from tools.basic_tools import RNASeqData
from xaio_config import output_dir


def loadRNASeq(normalization=""):
    data = RNASeqData()
    data.save_dir = os.path.expanduser(output_dir + "/dataset/RNASeq/")
    data.sample_ids = np.load(data.save_dir + "samples_id.npy", allow_pickle=True)
    data.sample_annotations = np.load(
        data.save_dir + "annot_dict.npy", allow_pickle=True
    ).item()
    data.sample_origins = np.load(
        data.save_dir + "annot_types.npy", allow_pickle=True
    ).item()
    data.sample_origins_per_annotation = np.load(
        data.save_dir + "annot_types_dict.npy", allow_pickle=True
    ).item()
    data.all_annotations = np.load(
        data.save_dir + "annot_values.npy", allow_pickle=True
    )
    data.sample_indices_per_annotation = np.load(
        data.save_dir + "annot_index.npy", allow_pickle=True
    ).item()
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
    data.train_indices_per_annotation = np.load(
        data.save_dir + "annot_index_train.npy", allow_pickle=True
    ).item()
    data.test_indices_per_annotation = np.load(
        data.save_dir + "annot_index_test.npy", allow_pickle=True
    ).item()
    data.feature_shortnames_ref = np.load(
        data.save_dir + "gene_dict.npy", allow_pickle=True
    ).item()
    data.std_values_on_training_sets = np.load(
        data.save_dir + "expressions_on_training_sets.npy", allow_pickle=True
    ).item()
    data.std_values_on_training_sets_argsort = np.load(
        data.save_dir + "expressions_on_training_sets_argsort.npy", allow_pickle=True
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
