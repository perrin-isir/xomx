# from IPython import embed as e
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import umap
from IPython import embed as e

assert e


def identity_func(x):
    return x


class XAIOData:
    """
    Attributes:

        save_dir (str or NoneType):
            directory where data is saved

        sample_ids (np.ndarray or NoneType):
            array of IDs: the i-th sample has ID sample_ids[i] (starting at 0)

        sample_indices (dict or NoneType):
            sample of ID "#" has index sample_indices["#"]

        sample_annotations (np.ndarray or NoneType):
            the i-th sample has annotation sample_annotations[i]

        sample_infos (dict or NoneType):
            sample_infos[i] is a dict containing additional information on the i-th
            sample

        all_annotations (np.ndarray or NoneType):
            list of all annotations

        sample_indices_per_annotation (dict or NoneType):
            sample_indices_per_annotation["#"] is the list of indices of the samples of
            annotation "#"

        feature_names (np.ndarray or NoneType):
            the i-th feature name is feature_names[i]

        feature_indices (dict or NoneType):
            feature_indices["#"] is the index of the feature of name "#"

        feature_mean_values (np.ndarray or NoneType):
            feature_mean_values[i] is the mean value of the i-th feature (in raw data)
            accross all samples

        feature_standard_deviations (np.ndarray or NoneType):
            similar to feature_mean_values, but standard deviation instead of mean

        train_indices (np.ndarray or NoneType):
            the list of indices of the samples belonging to the training set

        test_indices (np.ndarray or NoneType):
            the list of indices of the samples belonging to the test set

        train_indices_per_annotation (dict or NoneType):
            annot_index["#"] is the list of indices of the samples of annotation "#"
            which belong to the training set

        test_indices_per_annotation (dict or NoneType):
            annot_index["#"] is the list of indices of the samples of annotation "#"
            which belong to the test set

        std_values_on_training_sets (dict or NoneType):
            std_values_on_training_sets["#"][j] is the mean value of the j-th feature,
            normalized by mean and std_dev, across all samples of annotation "#"
            belonging to the training set ; it is useful to determine whether a
            feature is up-regulated for a given annotation (positive value), or
            down-regulated (negative value)

        std_values_on_training_sets_argsort (dict or NoneType):
            std_values_on_training_sets_argsort["#"] is the list of feature indices
            sorted by decreasing value in std_values_on_training_sets["#"]

        epsilon_shift (float or NoneType):
            the value of the shift used for log-normalization of the data; this
            parameter is computed during log-normalization

        maxlog (float or NoneType):
            maximum value of the log data; this parameter is computed during
            log-normalization

        data_array (dict):
            - data_array["raw"]:
                data_array["raw"][i, j]: value of the j-th feature of the i-th sample
            - data_array["std"]: data normalized by mean and standard deviation, such
            that:
                data_array["raw"][i, j] ==
                            data_array["std"][i, j] * feature_standard_deviations[j]
                            + feature_mean_values[j]
            - data_array["log"]: log-normalized values, such that:
                data_array["raw"][i, j] == np.exp(
                            data_array["log"][i,j] * (maxlog - np.log(epsilon_shift)
                        ) + np.log(epsilon_shift)) - epsilon_shift

        normalization_type (str or NoneType):
            if normalization_type=="raw", then data = data_array["raw"]
            if normalization_type=="std", then data = data_array["std"]
            if normalization_type=="log", then data = data_array["log"]

        nr_non_zero_features (int or NoneType):
            nr_non_zero_features[i] is, for the i-th sample, the number of features with
            positive values (in raw data)

        nr_non_zero_samples (int or NoneType):
            nr_non_zero_samples[i] is the number of samples with positive values on the
            i-th feature (in raw data)

        total_feature_sums (np.ndarray or NoneType):
            total_feature_sums[i] is, for the i-th sample, the sum of values (in raw
            data) accross all features
    """

    def __init__(self):
        self.save_dir = None
        self.sample_ids = None
        self.sample_indices = None
        self.sample_annotations = None
        self.sample_infos = None
        self.all_annotations = None
        self.sample_indices_per_annotation = None
        self.feature_names = None
        self.feature_indices = None
        self.feature_mean_values = None
        self.feature_standard_deviations = None
        self.data = None
        self.data_array = {}
        self.params = None
        self.train_indices = None
        self.test_indices = None
        self.train_indices_per_annotation = None
        self.test_indices_per_annotation = None
        self.std_values_on_training_sets = None
        self.std_values_on_training_sets_argsort = None
        self.epsilon_shift = None
        self.maxlog = None
        self.normalization_type = None
        self.nr_non_zero_features = None
        self.nr_non_zero_samples = None
        self.total_feature_sums = None

    @property
    def nr_samples(self):
        """
        the total number of samples, or None if self.sample_ids is None
        """
        if self.sample_ids is None and self.data_array is None:
            return None
        elif self.sample_ids is not None:
            return len(self.sample_ids)
        else:
            return self.data_array[list(self.data_array.keys())[0]].shape[0]

    @property
    def nr_features(self):
        """
        the total number of features for each sample, or None if
        self.feature_names is None
        """
        if self.feature_names is None and self.data_array is None:
            return None
        elif self.feature_names is not None:
            return len(self.feature_names)
        else:
            return self.data_array[list(self.data_array.keys())[0]].shape[1]

    def save(self, normalization_types_list=None, save_dir=None):
        if normalization_types_list is None:
            normalization_types_list = []
        if save_dir is not None:
            self.save_dir = save_dir
        assert self.save_dir is not None
        if not (os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir, exist_ok=True)
        if self.nr_samples is not None:
            np.save(os.path.join(self.save_dir, "nr_samples.npy"), self.nr_samples)
            print("Saved: " + os.path.join(self.save_dir, "nr_samples.npy"))
        if self.nr_features is not None:
            np.save(os.path.join(self.save_dir, "nr_features.npy"), self.nr_features)
            print("Saved: " + os.path.join(self.save_dir, "nr_features.npy"))
        if self.sample_ids is not None:
            np.save(os.path.join(self.save_dir, "sample_ids.npy"), self.sample_ids)
            print("Saved: " + os.path.join(self.save_dir, "sample_ids.npy"))
        if self.sample_indices is not None:
            np.save(
                os.path.join(self.save_dir, "sample_indices.npy"), self.sample_indices
            )
            print("Saved: " + os.path.join(self.save_dir, "sample_indices.npy"))
        if self.sample_indices_per_annotation is not None:
            np.save(
                os.path.join(self.save_dir, "sample_indices_per_annotation.npy"),
                self.sample_indices_per_annotation,
            )
            print(
                "Saved: "
                + os.path.join(self.save_dir, "sample_indices_per_annotation.npy")
            )
        if self.sample_infos is not None:
            np.save(
                os.path.join(self.save_dir, "sample_infos.npy"),
                self.sample_infos,
            )
            print("Saved: " + os.path.join(self.save_dir, "sample_infos.npy"))
        if self.sample_annotations is not None:
            np.save(
                os.path.join(self.save_dir, "sample_annotations.npy"),
                self.sample_annotations,
            )
            print("Saved: " + os.path.join(self.save_dir, "sample_annotations.npy"))
        if self.all_annotations is not None:
            np.save(
                os.path.join(self.save_dir, "all_annotations.npy"), self.all_annotations
            )
            print("Saved: " + os.path.join(self.save_dir, "all_annotations.npy"))
        if self.feature_names is not None:
            np.save(
                os.path.join(self.save_dir, "feature_names.npy"), self.feature_names
            )
            print("Saved: " + os.path.join(self.save_dir, "feature_names.npy"))
        if self.feature_mean_values is not None:
            np.save(
                os.path.join(self.save_dir, "feature_mean_values.npy"),
                self.feature_mean_values,
            )
            print(
                "Saved: "
                + os.path.join(self.save_dir, "feature_standard_deviations.npy")
            )
        if self.feature_standard_deviations is not None:
            np.save(
                os.path.join(self.save_dir, "feature_standard_deviations.npy"),
                self.feature_standard_deviations,
            )
            print(
                "Saved: "
                + os.path.join(self.save_dir, "feature_standard_deviations.npy")
            )
        if self.train_indices is not None:
            np.save(
                os.path.join(self.save_dir, "train_indices.npy"),
                self.train_indices,
            )
            print("Saved: " + os.path.join(self.save_dir, "train_indices.npy"))
        if self.test_indices is not None:
            np.save(
                os.path.join(self.save_dir, "test_indices.npy"),
                self.test_indices,
            )
            print("Saved: " + os.path.join(self.save_dir, "test_indices.npy"))
        if self.train_indices_per_annotation is not None:
            np.save(
                os.path.join(self.save_dir, "train_indices_per_annotation.npy"),
                self.train_indices_per_annotation,
            )
            print(
                "Saved: "
                + os.path.join(self.save_dir, "train_indices_per_annotation.npy")
            )
        if self.test_indices_per_annotation is not None:
            np.save(
                os.path.join(self.save_dir, "test_indices_per_annotation.npy"),
                self.test_indices_per_annotation,
            )
            print(
                "Saved: "
                + os.path.join(self.save_dir, "test_indices_per_annotation.npy")
            )
        if self.std_values_on_training_sets is not None:
            np.save(
                os.path.join(self.save_dir, "std_values_on_training_sets.npy"),
                self.std_values_on_training_sets,
            )
            print(
                "Saved: "
                + os.path.join(self.save_dir, "std_values_on_training_sets.npy")
            )
        if self.std_values_on_training_sets_argsort is not None:
            np.save(
                os.path.join(self.save_dir, "std_values_on_training_sets_argsort.npy"),
                self.std_values_on_training_sets_argsort,
            )
            print(
                "Saved: "
                + os.path.join(self.save_dir, "std_values_on_training_sets_argsort.npy")
            )
        if self.epsilon_shift is not None:
            np.save(
                os.path.join(self.save_dir, "epsilon_shift.npy"), self.epsilon_shift
            )
            print("Saved: " + os.path.join(self.save_dir, "epsilon_shift.npy"))
        if self.maxlog is not None:
            np.save(os.path.join(self.save_dir, "maxlog.npy"), self.maxlog)
            print("Saved: " + os.path.join(self.save_dir, "maxlog.npy"))
        if self.feature_indices is not None:
            np.save(
                os.path.join(self.save_dir, "feature_indices.npy"),
                self.feature_indices,
            )
            print("Saved: " + os.path.join(self.save_dir, "feature_indices.npy"))
        if self.nr_non_zero_features is not None:
            np.save(
                os.path.join(self.save_dir, "nr_non_zero_features.npy"),
                self.nr_non_zero_features,
            )
            print("Saved: " + os.path.join(self.save_dir, "nr_non_zero_features.npy"))
        if self.nr_non_zero_samples is not None:
            np.save(
                os.path.join(self.save_dir, "nr_non_zero_samples.npy"),
                self.nr_non_zero_samples,
            )
            print("Saved: " + os.path.join(self.save_dir, "nr_non_zero_samples.npy"))
        if self.total_feature_sums is not None:
            np.save(
                os.path.join(self.save_dir, "total_feature_sums.npy"),
                self.total_feature_sums,
            )
            print("Saved: " + os.path.join(self.save_dir, "total_feature_sums.npy"))
        if self.params is not None:
            np.save(
                os.path.join(self.save_dir, "params.npy"),
                self.params,
            )
        for normtype in self.data_array:
            if (
                self.data_array[normtype] is not None
                and normtype in normalization_types_list
            ):
                fp_data = np.memmap(
                    os.path.join(self.save_dir, normtype + "_data.bin"),
                    dtype="float32",
                    mode="w+",
                    shape=(self.nr_features, self.nr_samples),
                )
                fp_data[:] = self.data_array[normtype].transpose()[:]
                del fp_data
                print("Saved: " + os.path.join(self.save_dir, normtype + "_data.bin"))

    def load(self, normalization_types_list=None, load_dir=None):
        if normalization_types_list is None:
            normalization_types_list = []
        if load_dir is None:
            assert self.save_dir is not None
            ldir = self.save_dir
        else:
            ldir = load_dir
            self.save_dir = load_dir
        nr_samples_ = nr_features_ = 0
        if os.path.exists(os.path.join(ldir, "nr_samples.npy")):
            nr_samples_ = np.load(
                os.path.join(ldir, "nr_samples.npy"), allow_pickle=True
            ).item()
        if os.path.exists(os.path.join(ldir, "nr_features.npy")):
            nr_features_ = np.load(
                os.path.join(ldir, "nr_features.npy"), allow_pickle=True
            ).item()
        if os.path.exists(os.path.join(ldir, "sample_ids.npy")):
            self.sample_ids = np.load(
                os.path.join(ldir, "sample_ids.npy"), allow_pickle=True
            )
        if os.path.exists(os.path.join(ldir, "sample_indices.npy")):
            self.sample_indices = np.load(
                os.path.join(ldir, "sample_indices.npy"), allow_pickle=True
            ).item()
        if os.path.exists(os.path.join(ldir, "sample_indices_per_annotation.npy")):
            self.sample_indices_per_annotation = np.load(
                os.path.join(ldir, "sample_indices_per_annotation.npy"),
                allow_pickle=True,
            ).item()
        if os.path.exists(os.path.join(ldir, "sample_infos.npy")):
            self.sample_infos = np.load(
                os.path.join(ldir, "sample_infos.npy"), allow_pickle=True
            ).item()
        if os.path.exists(os.path.join(ldir, "sample_annotations.npy")):
            self.sample_annotations = np.load(
                os.path.join(ldir, "sample_annotations.npy"), allow_pickle=True
            )
        if os.path.exists(os.path.join(ldir, "all_annotations.npy")):
            self.all_annotations = np.load(
                os.path.join(ldir, "all_annotations.npy"), allow_pickle=True
            )
        if os.path.exists(os.path.join(ldir, "feature_names.npy")):
            self.feature_names = np.load(
                os.path.join(ldir, "feature_names.npy"), allow_pickle=True
            )
        if os.path.exists(os.path.join(ldir, "feature_mean_values.npy")):
            self.feature_mean_values = np.load(
                os.path.join(ldir, "feature_mean_values.npy"), allow_pickle=True
            )
        if os.path.exists(os.path.join(ldir, "feature_standard_deviations.npy")):
            self.feature_standard_deviations = np.load(
                os.path.join(ldir, "feature_standard_deviations.npy"), allow_pickle=True
            )
        if os.path.exists(os.path.join(ldir, "feature_indices.npy")):
            self.feature_indices = np.load(
                os.path.join(ldir, "feature_indices.npy"), allow_pickle=True
            ).item()
        if os.path.exists(os.path.join(ldir, "train_indices.npy")):
            self.train_indices = np.load(
                os.path.join(ldir, "train_indices.npy"), allow_pickle=True
            )
        if os.path.exists(os.path.join(ldir, "test_indices.npy")):
            self.test_indices = np.load(
                os.path.join(ldir, "test_indices.npy"), allow_pickle=True
            )
        if os.path.exists(os.path.join(ldir, "train_indices_per_annotation.npy")):
            self.train_indices_per_annotation = np.load(
                os.path.join(ldir, "train_indices_per_annotation.npy"),
                allow_pickle=True,
            ).item()
        if os.path.exists(os.path.join(ldir, "test_indices_per_annotation.npy")):
            self.test_indices_per_annotation = np.load(
                os.path.join(ldir, "test_indices_per_annotation.npy"), allow_pickle=True
            ).item()
        if os.path.exists(os.path.join(ldir, "std_values_on_training_sets.npy")):
            self.std_values_on_training_sets = np.load(
                os.path.join(ldir, "std_values_on_training_sets.npy"), allow_pickle=True
            ).item()
        if os.path.exists(
            os.path.join(ldir, "std_values_on_training_sets_argsort.npy")
        ):
            self.std_values_on_training_sets_argsort = np.load(
                os.path.join(ldir, "std_values_on_training_sets_argsort.npy"),
                allow_pickle=True,
            ).item()
        if os.path.exists(os.path.join(ldir, "epsilon_shift.npy")):
            self.epsilon_shift = np.load(
                os.path.join(ldir, "epsilon_shift.npy"), allow_pickle=True
            ).item()
        if os.path.exists(os.path.join(ldir, "maxlog.npy")):
            self.maxlog = np.load(
                os.path.join(ldir, "maxlog.npy"), allow_pickle=True
            ).item()
        if os.path.exists(os.path.join(ldir, "nr_non_zero_samples.npy")):
            self.nr_non_zero_samples = np.load(
                os.path.join(ldir, "nr_non_zero_samples.npy"), allow_pickle=True
            )
        if os.path.exists(os.path.join(ldir, "nr_non_zero_features.npy")):
            self.nr_non_zero_features = np.load(
                os.path.join(ldir, "nr_non_zero_features.npy"), allow_pickle=True
            )
        if os.path.exists(os.path.join(ldir, "total_feature_sums.npy")):
            self.total_feature_sums = np.load(
                os.path.join(ldir, "total_feature_sums.npy"), allow_pickle=True
            )
        if os.path.exists(os.path.join(ldir, "params.npy")):
            self.params = np.load(
                os.path.join(ldir, "params.npy"), allow_pickle=True
            ).item()
        for normtype in normalization_types_list:
            assert os.path.exists(os.path.join(ldir, normtype + "_data.bin"))
            assert nr_samples_ > 0
            assert nr_features_ > 0
            self.data_array[normtype] = np.array(
                np.memmap(
                    os.path.join(ldir, normtype + "_data.bin"),
                    dtype="float32",
                    mode="r",
                    shape=(nr_features_, nr_samples_),
                )
            ).transpose()
        if len(normalization_types_list) > 0:
            print("Normalization type: " + normalization_types_list[0])
            self.data = self.data_array[normalization_types_list[0]]
            self.normalization_type = normalization_types_list[0]

    def compute_sample_indices(self):
        assert self.sample_ids is not None
        self.sample_indices = {}
        for i, s_id in enumerate(self.sample_ids):
            self.sample_indices[s_id] = i

    def compute_sample_indices_per_annotation(self):
        assert self.sample_annotations is not None
        self.sample_indices_per_annotation = {}
        for i, annot in enumerate(self.sample_annotations):
            self.sample_indices_per_annotation.setdefault(annot, [])
            self.sample_indices_per_annotation[annot].append(i)

    def compute_all_annotations(self):
        assert self.sample_annotations is not None
        self.all_annotations = np.array(list(dict.fromkeys(self.sample_annotations)))

    def compute_feature_mean_values(self):
        assert self.data_array["raw"] is not None and self.nr_features is not None
        self.feature_mean_values = [
            np.mean(self.data_array["raw"][:, i]) for i in range(self.nr_features)
        ]

    def compute_feature_standard_deviations(self):
        assert self.data_array["raw"] is not None and self.nr_features is not None
        self.feature_standard_deviations = [
            np.std(self.data_array["raw"][:, i]) for i in range(self.nr_features)
        ]

    def compute_normalization(self, normtype):
        if normtype == "std":
            assert (
                self.data_array["raw"] is not None
                and self.feature_mean_values is not None
                and self.feature_standard_deviations is not None
                and self.nr_features is not None
            )
            self.data_array["std"] = np.copy(self.data_array["raw"])
            for i in range(self.nr_features):
                if self.feature_standard_deviations[i] == 0.0:
                    self.data_array["std"][:, i] = 0.0
                else:
                    self.data_array["std"][:, i] = (
                        self.data_array["std"][:, i] - self.feature_mean_values[i]
                    ) / self.feature_standard_deviations[i]
        elif normtype == "log":
            assert self.data_array["raw"] is not None and self.nr_features is not None
            self.data_array["log"] = np.copy(self.data_array["raw"])
            self.epsilon_shift = 1.0
            for i in range(self.nr_features):
                self.data_array["log"][:, i] = np.log(
                    self.data_array["log"][:, i] + self.epsilon_shift
                )
            self.maxlog = np.max(self.data_array["log"])
            for i in range(self.nr_features):
                self.data_array["log"][:, i] = (
                    self.data_array["log"][:, i] - np.log(self.epsilon_shift)
                ) / (self.maxlog - np.log(self.epsilon_shift))

    def compute_train_and_test_indices(self, test_train_ratio=0.25):
        assert self.sample_indices_per_annotation is not None
        self.train_indices_per_annotation = {}
        self.test_indices_per_annotation = {}
        for annot in self.sample_indices_per_annotation:
            idxs = np.random.permutation(self.sample_indices_per_annotation[annot])
            # cut = len(idxs) // 4 + 1
            cut = np.floor(len(idxs) * test_train_ratio).astype("int") + 1
            self.test_indices_per_annotation[annot] = idxs[:cut]
            self.train_indices_per_annotation[annot] = idxs[cut:]
        self.train_indices = np.concatenate(
            list(self.train_indices_per_annotation.values())
        )
        self.test_indices = np.concatenate(
            list(self.test_indices_per_annotation.values())
        )

    def compute_feature_indices(self):
        assert self.feature_names is not None
        self.feature_indices = {}
        for i, elt in enumerate(self.feature_names):
            self.feature_indices[elt] = i

    def compute_std_values_on_training_sets(self):
        assert (
            self.all_annotations is not None
            and self.nr_features is not None
            and self.data_array["std"] is not None
            and self.train_indices_per_annotation is not None
        )
        self.std_values_on_training_sets = {}
        for annot in self.all_annotations:
            self.std_values_on_training_sets[annot] = []
            for i in range(self.nr_features):
                self.std_values_on_training_sets[annot].append(
                    np.mean(
                        [
                            self.data_array["std"][j, i]
                            for j in self.train_indices_per_annotation[annot]
                        ]
                    )
                )

    def compute_std_values_on_training_sets_argsort(self):
        assert (
            self.all_annotations is not None
            and self.std_values_on_training_sets is not None
        )
        self.std_values_on_training_sets_argsort = {}
        for annot in self.all_annotations:
            self.std_values_on_training_sets_argsort[annot] = np.argsort(
                self.std_values_on_training_sets[annot]
            )[::-1]

    def compute_nr_non_zero_features(self):
        assert self.data_array["raw"] is not None and self.nr_samples is not None
        self.nr_non_zero_features = np.empty((self.nr_samples,), dtype=int)
        for i in range(self.nr_samples):
            self.nr_non_zero_features[i] = len(
                np.where(self.data_array["raw"][i, :] > 0.0)[0]
            )

    def compute_nr_non_zero_samples(self):
        assert self.data_array["raw"] is not None and self.nr_features is not None
        self.nr_non_zero_samples = np.empty((self.nr_features,), dtype=int)
        for i in range(self.nr_features):
            self.nr_non_zero_samples[i] = len(
                np.where(self.data_array["raw"][:, i] > 0.0)[0]
            )

    def compute_total_feature_sums(self):
        assert self.data_array["raw"] is not None and self.nr_samples is not None
        self.total_feature_sums = np.empty((self.nr_samples,), dtype=float)
        for i in range(self.nr_samples):
            self.total_feature_sums[i] = np.sum(self.data_array["raw"][i, :])

    def normalize_feature_sums(self, sum_value):
        """
        Normalizes feature values, so that for every sample, the total sum of
        the feature values becomes equal to sum_value.
        normalize_feature_sums() is supposed to be an early normalization of the raw
        data, before any other normalization.
        """
        assert self.data_array["raw"] is not None and self.nr_samples is not None
        self.compute_total_feature_sums()
        for i in range(self.nr_samples):
            self.data_array["raw"][i, :] *= sum_value / self.total_feature_sums[i]
            self.total_feature_sums[i] = sum_value
        if self.feature_mean_values is not None:
            self.compute_feature_mean_values()
        if self.feature_standard_deviations is not None:
            self.compute_feature_standard_deviations()

    def reduce_samples(self, idx_list):
        idx_list_ = np.copy(idx_list)
        for i_, idx in enumerate(idx_list_):
            if type(idx) == str or type(idx) == np.str_:
                idx_list_[i_] = self.sample_indices[idx]
        idx_list_ = idx_list_.astype("int")
        if self.sample_ids is not None:
            self.sample_ids = np.take(self.sample_ids, idx_list_)
        if self.sample_annotations is not None:
            self.sample_annotations = np.take(self.sample_annotations, idx_list_)
        if self.sample_infos is not None:
            self.sample_infos = np.take(self.sample_infos, idx_list_)
        if self.sample_indices is not None:
            self.compute_sample_indices()
        if self.all_annotations is not None:
            self.compute_all_annotations()
        if self.sample_indices_per_annotation is not None:
            self.compute_sample_indices_per_annotation()
        for normtype in self.data_array:
            if self.data_array[normtype] is not None:
                self.data_array[normtype] = np.take(
                    self.data_array[normtype], idx_list_, axis=0
                )
        if self.normalization_type is not None:
            self.data = self.data_array[self.normalization_type]
        if self.feature_mean_values is not None:
            self.compute_feature_mean_values()
        if self.feature_standard_deviations is not None:
            self.compute_feature_standard_deviations()
        if (
            self.train_indices_per_annotation is not None
            or self.test_indices_per_annotation is not None
        ):
            self.compute_train_and_test_indices()
        if self.std_values_on_training_sets is not None:
            self.compute_std_values_on_training_sets()
        if self.std_values_on_training_sets_argsort is not None:
            self.compute_std_values_on_training_sets_argsort()
        if self.nr_non_zero_features is not None:
            self.compute_nr_non_zero_features()
        if self.nr_non_zero_samples is not None:
            self.compute_nr_non_zero_samples()
        if self.total_feature_sums is not None:
            self.compute_total_feature_sums()

    def reduce_features(self, idx_list):
        idx_list_ = np.copy(idx_list)
        for i_, idx in enumerate(idx_list_):
            if type(idx) == str or type(idx) == np.str_:
                idx_list_[i_] = self.feature_indices[idx]
        idx_list_ = idx_list_.astype("int")
        if self.feature_names is not None:
            self.feature_names = np.take(self.feature_names, idx_list_)
        if self.feature_mean_values is not None:
            self.feature_mean_values = np.take(self.feature_mean_values, idx_list_)
        if self.feature_standard_deviations is not None:
            self.feature_standard_deviations = np.take(
                self.feature_standard_deviations, idx_list_
            )
        if self.feature_indices is not None:
            self.compute_feature_indices()
        for normtype in self.data_array:
            if self.data_array[normtype] is not None:
                self.data_array[normtype] = np.take(
                    self.data_array[normtype].transpose(), idx_list_, axis=0
                ).transpose()
        if self.normalization_type is not None:
            self.data = self.data_array[self.normalization_type]
        if (
            self.all_annotations is not None
            and self.std_values_on_training_sets is not None
        ):
            for cat in self.all_annotations:
                self.std_values_on_training_sets[cat] = list(
                    np.take(self.std_values_on_training_sets[cat], idx_list_)
                )
            self.compute_std_values_on_training_sets_argsort()
        if self.nr_non_zero_features is not None:
            self.compute_nr_non_zero_features()
        if self.nr_non_zero_samples is not None:
            self.compute_nr_non_zero_samples()
        if self.total_feature_sums is not None:
            self.compute_total_feature_sums()

    def feature_values_ratio(self, idx_list, sample_idx=None):
        """Computes the sum of values (in raw data), across all samples or for one
        given sample, for features of indices in idx_list, divided by the sum of values
        for all the features"""
        assert self.data_array["raw"] is not None
        if sample_idx:
            return np.sum(self.data_array["raw"][sample_idx, idx_list]) / np.sum(
                self.data_array["raw"][sample_idx, :]
            )
        else:
            return np.sum(self.data_array["raw"][:, idx_list]) / np.sum(
                self.data_array["raw"]
            )

    def regex_search(self, rexpr):
        """Tests for every feature name whether it matches the regular expression
        rexpr; returns the list of indices of the features that do match
        """
        return np.where([re.search(rexpr, s) for s in self.feature_names])[0]

    def feature_func(self, idx, cat_=None, func_=np.mean):
        """Applies the function func_ to the array of values of the feature of index
        idx, across either all samples, or samples with annotation cat_;
        the name of the feature can be given instead of the index idx"""
        if type(idx) == str or type(idx) == np.str_:
            idx = self.feature_indices[idx]
        if not cat_:
            return func_(self.data[:, idx])
        else:
            return func_(
                [self.data[i_, idx] for i_ in self.sample_indices_per_annotation[cat_]]
            )

    def import_pandas(self, df):
        self.sample_ids = df.columns.to_numpy()
        self.compute_sample_indices()
        self.feature_names = df.index.to_numpy()
        self.compute_feature_indices()
        self.data_array["raw"] = df.to_numpy(dtype=np.float64).transpose()

    # def feature_plot(self, idx, cat_=None, v_min=None, v_max=None):
    #     """plots the value of the feature of index idx for all samples;
    #     if cat_ is not None the samples of annotation cat_ have a different color
    #     the short name of the feature can be given instead of the index"""
    #     if type(idx) == str:
    #         idx = self.feature_shortnames_ref[idx]
    #     y = self.data[:, idx]
    #     if v_min is not None and v_max is not None:
    #         y = np.clip(y, v_min, v_max)
    #     x = np.arange(0, self.nr_samples) / self.nr_samples
    #     plt.scatter(x, y, s=1)
    #     if cat_:
    #
    #         y = [self.data[i_, idx] for i_ in
    #              self.sample_indices_per_annotation[cat_]]
    #         if v_min is not None and v_max is not None:
    #             y = np.clip(y, v_min, v_max)
    #         x = np.array(self.sample_indices_per_annotation[cat_]) / self.nr_samples
    #         plt.scatter(x, y, s=1)
    #     plt.show()

    # def function_plot(self, func_=np.identity, cat_=None):
    #     """plots the value of a function on all samples (the function must take sample
    #     indices in input);
    #     if cat_ is not None the samples of annotation cat_ have a different color"""
    #     y = [func_(i) for i in range(self.nr_samples)]
    #     x = np.arange(0, self.nr_samples) / self.nr_samples
    #     fig, ax = plt.subplots()
    #     parts = ax.violinplot(
    #         y,
    #         [0.5],
    #         points=60,
    #         widths=1.0,
    #         showmeans=False,
    #         showextrema=False,
    #         showmedians=False,
    #         bw_method=0.5,
    #     )
    #     for pc in parts["bodies"]:
    #         pc.set_facecolor("#D43F3A")
    #         pc.set_edgecolor("grey")
    #         pc.set_alpha(0.7)
    #     ax.scatter(x, y, s=1)
    #
    #     if cat_:
    #         y = [func_(i_) for i_ in self.sample_indices_per_annotation[cat_]]
    #         x = np.array(self.sample_indices_per_annotation[cat_]) / self.nr_samples
    #         plt.scatter(x, y, s=1)
    #     plt.show()

    def _samples_by_annotations(self, sort_annot=False):
        if sort_annot:
            argsort_annotations = np.argsort(
                [
                    len(self.sample_indices_per_annotation[i])
                    for i in self.all_annotations
                ]
            )[::-1]
        else:
            argsort_annotations = np.arange(len(self.all_annotations))
        list_samples = np.concatenate(
            [
                self.sample_indices_per_annotation[self.all_annotations[i]]
                for i in argsort_annotations
            ]
        )
        boundaries = np.cumsum(
            [0]
            + [
                len(self.sample_indices_per_annotation[self.all_annotations[i]])
                for i in argsort_annotations
            ]
        )
        set_xticks2 = (boundaries[1:] + boundaries[:-1]) // 2
        set_xticks = list(np.sort(np.concatenate((boundaries, set_xticks2))))
        set_xticks_text = ["|"] + list(
            np.concatenate(
                [[str(self.all_annotations[i]), "|"] for i in argsort_annotations]
            )
        )
        return list_samples, set_xticks, set_xticks_text, boundaries

    def function_scatter(
        self,
        func1_=identity_func,
        func2_=identity_func,
        samples_or_features="samples",
        violinplot=False,
        xlog_scale=False,
        ylog_scale=False,
        xlabel="",
        ylabel="",
        function_plot_=False,
    ):
        """Displays a scatter plot, with coordinates computed by applying two
        functions (func1_ and func2_) to every sample or every feature, depending
        on the value of sof_ which must be either "samples" or "features"
        (both functions must take indices in input);
        if sof=="samples" and cat_ is not None the samples of annotation cat_ have
        a different color"""
        assert samples_or_features == "samples" or samples_or_features == "features"
        set_xticks = None
        set_xticks_text = None
        violinplots_done = False
        fig, ax = plt.subplots()
        if samples_or_features == "samples":
            if self.all_annotations is not None and function_plot_:
                (
                    list_samples,
                    set_xticks,
                    set_xticks_text,
                    boundaries,
                ) = self._samples_by_annotations(sort_annot=True)
                y = [func2_(i) for i in list_samples]
                x = [i for i in range(self.nr_samples)]
                if violinplot:
                    for i in range(len(boundaries) - 1):
                        parts = ax.violinplot(
                            y[boundaries[i] : boundaries[i + 1]],
                            [(boundaries[i + 1] + boundaries[i]) / 2.0],
                            points=60,
                            widths=(boundaries[i + 1] - boundaries[i]) * 0.8,
                            showmeans=False,
                            showextrema=False,
                            showmedians=False,
                            bw_method=0.5,
                        )
                        for pc in parts["bodies"]:
                            pc.set_facecolor("#D43F3A")
                            pc.set_edgecolor("grey")
                            pc.set_alpha(0.5)
                violinplots_done = True
            else:
                y = [func2_(i) for i in range(self.nr_samples)]
                x = [func1_(i) for i in range(self.nr_samples)]
        else:
            y = [func2_(i) for i in range(self.nr_features)]
            x = [func1_(i) for i in range(self.nr_features)]
        xmax = np.max(x)
        xmin = np.min(x)
        if violinplot and not violinplots_done:
            parts = ax.violinplot(
                y,
                [(xmax + xmin) / 2.0],
                points=60,
                widths=xmax - xmin,
                showmeans=False,
                showextrema=False,
                showmedians=False,
                bw_method=0.5,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor("#D43F3A")
                pc.set_edgecolor("grey")
                pc.set_alpha(0.5)
        scax = ax.scatter(x, y, s=1)
        ann = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(-100, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        ann.set_visible(False)

        def update_annot(ind, sc):
            pos = sc.get_offsets()[ind["ind"][0]]
            ann.xy = pos
            if samples_or_features == "samples":
                if self.all_annotations is not None and function_plot_:
                    text = "{}".format(self.sample_ids[list_samples[ind["ind"][0]]])
                else:
                    text = "{}".format(self.sample_ids[ind["ind"][0]])

            else:
                text = "{}".format(self.feature_names[ind["ind"][0]])
            ann.set_text(text)

        def hover(event):
            vis = ann.get_visible()
            if event.inaxes == ax:
                cont, ind = scax.contains(event)
                if cont:
                    update_annot(ind, scax)
                    ann.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        ann.set_visible(False)
                        fig.canvas.draw_idle()
            if event.inaxes == ax:
                cont, ind = scax.contains(event)
                if cont:
                    update_annot(ind, scax)
                    ann.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        ann.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

        if set_xticks is not None:
            plt.xticks(set_xticks, set_xticks_text)
        if xlog_scale:
            plt.xscale("log")
        if ylog_scale:
            plt.yscale("log")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def function_plot(
        self,
        func=identity_func,
        samples_or_features="samples",
        violinplot=True,
        ylog_scale=False,
        xlabel="",
        ylabel="",
    ):
        """Plots the value of a function on every sample or every feature, depending
        on the value of sof_ which must be either "samples" or "features"
        (the function must take indices in input)"""
        self.function_scatter(
            identity_func,
            func,
            samples_or_features,
            violinplot,
            xlog_scale=False,
            ylog_scale=ylog_scale,
            xlabel=xlabel,
            ylabel=ylabel,
            function_plot_=True,
        )

    def feature_plot(self, features=None, normalization_type="", ylog_scale=False):
        """ """
        if normalization_type == "":
            data_ = self.data
        else:
            assert normalization_type in self.data_array
            data_ = self.data_array[normalization_type]
        if type(features) == str or type(features) == np.str_ or type(features) == int:
            idx = features
            if type(idx) == str or type(idx) == np.str_:
                assert self.feature_indices is not None
                idx = self.feature_indices[idx]
            self.function_plot(
                lambda i: data_[i, idx], "samples", ylog_scale=ylog_scale
            )
        else:
            xsize = self.nr_samples
            ysize = len(features)
            set_xticks = None
            set_xticks_text = None
            plot_array = np.empty((len(features), xsize))
            feature_indices_list_ = []
            for idx in features:
                if type(idx) == str or type(idx) == np.str_:
                    assert self.feature_indices is not None
                    idx = self.feature_indices[idx]
                feature_indices_list_.append(idx)
            if self.all_annotations is None:
                for k, idx in enumerate(feature_indices_list_):
                    plot_array[k, :] = [data_[i, idx] for i in range(self.nr_samples)]
            else:
                (
                    list_samples,
                    set_xticks,
                    set_xticks_text,
                    boundaries,
                ) = self._samples_by_annotations(sort_annot=False)
                for k, idx in enumerate(feature_indices_list_):
                    plot_array[k, :] = [data_[i, idx] for i in list_samples]

            fig, ax = plt.subplots()
            im = ax.imshow(plot_array, extent=[0, xsize, 0, ysize], aspect="auto")
            if set_xticks is not None:
                plt.xticks(set_xticks, set_xticks_text)
            plt.yticks(
                np.arange(ysize) + 0.5,
                [self.feature_names[i] for i in feature_indices_list_][::-1],
            )
            plt.tick_params(axis=u"both", which=u"both", length=0)
            plt.colorbar(im)
            plt.show()

    def umap_plot(
        self,
        normalization_type="",
        save_dir=None,
        metric="cosine",
        min_dist=0.0,
        n_neighbors=30,
        random_state=None,
    ):
        reducer = umap.UMAP(
            metric=metric,
            min_dist=min_dist,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        print("Starting UMAP reduction...")
        if normalization_type in self.data_array:
            x = self.data_array[normalization_type]
        else:
            x = self.data
        reducer.fit(x)
        embedding = reducer.transform(x)
        print("Done.")

        # all_colors = []
        # def color_function(id_):
        #     label_ = self.sample_annotations[id_]
        #     type_ = data.sample_origins[indices[id_]]
        #     clo = (
        #         np.where(data.all_annotations == label_)[0],
        #         list(data.sample_origins_per_annotation[label_]).index(type_),
        #     )
        #     if clo in all_colors:
        #         return all_colors.index(clo)
        #     else:
        #         all_colors.append(clo)
        #         return len(all_colors) - 1

        def hover_function(id_):
            return "{}".format(
                self.sample_ids[id_] + ": " + str(self.sample_annotations[id_])
            )

        annot_idxs = {}
        for i, annot_ in enumerate(self.all_annotations):
            annot_idxs[annot_] = i

        samples_color = np.empty(self.nr_samples)
        for i in range(self.nr_samples):
            samples_color[i] = annot_idxs[self.sample_annotations[i]]

        fig, ax = plt.subplots()

        sc = plt.scatter(
            embedding[:, 0], embedding[:, 1], c=samples_color, cmap="nipy_spectral", s=5
        )
        plt.gca().set_aspect("equal", "datalim")

        ann = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        ann.set_visible(False)

        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            ann.xy = pos
            text = hover_function(ind["ind"][0])
            ann.set_text(text)

        def hover(event):
            vis = ann.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    ann.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        ann.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "plot.png"), dpi=200)
        else:
            plt.show()


def confusion_matrix(classifier, data_test, target_test):
    nr_neg = len(np.where(target_test == 0)[0])
    nr_pos = len(np.where(target_test == 1)[0])
    result = classifier.predict(data_test) - target_test
    fp = len(np.where(result == 1)[0])
    fn = len(np.where(result == -1)[0])
    tp = nr_pos - fn
    tn = nr_neg - fp
    return np.array([[tp, fp], [fn, tn]])


def matthews_coef(confusion_m):
    tp = confusion_m[0, 0]
    fp = confusion_m[0, 1]
    fn = confusion_m[1, 0]
    tn = confusion_m[1, 1]
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator == 0:
        denominator = 1
    mcc = (tp * tn - fp * fn) / np.sqrt(denominator)
    return mcc


def feature_selection_from_list(
    data,
    annotation,
    feature_indices,
):

    f_indices = np.zeros_like(feature_indices, dtype=np.int64)
    for i in range(len(f_indices)):
        if type(feature_indices[i]) == str or type(feature_indices[i]) == np.str_:
            f_indices[i] = data.feature_indices[feature_indices[i]]
        else:
            f_indices[i] = feature_indices[i]
    data_train = np.take(
        np.take(data.data.transpose(), f_indices, axis=0),
        data.train_indices,
        axis=1,
    ).transpose()
    target_train = np.zeros(data.nr_samples)
    target_train[data.train_indices_per_annotation[annotation]] = 1.0
    target_train = np.take(target_train, data.train_indices, axis=0)
    data_test = np.take(
        np.take(data.data.transpose(), f_indices, axis=0),
        data.test_indices,
        axis=1,
    ).transpose()
    target_test = np.zeros(data.nr_samples)
    target_test[data.test_indices_per_annotation[annotation]] = 1.0
    target_test = np.take(target_test, data.test_indices, axis=0)
    return (
        feature_indices,
        data_train,
        target_train,
        data_test,
        target_test,
    )


def naive_feature_selection(
    data,
    annotation,
    selection_size,
):
    if selection_size >= data.nr_features:
        feature_indices = list(range(data.nr_features))
    else:
        assert data.std_values_on_training_sets_argsort is not None
        feature_indices = np.array(
            data.std_values_on_training_sets_argsort[annotation][
                : (selection_size // 2)
            ].tolist()
            + data.std_values_on_training_sets_argsort[annotation][
                -(selection_size - selection_size // 2) :
            ].tolist()
        )
    return feature_selection_from_list(data, annotation, feature_indices)


def plot_scores(data, scores, score_threshold, indices, annotation=None, save_dir=None):
    annot_colors = {}
    denom = len(data.all_annotations)
    for i, val in enumerate(data.all_annotations):
        if annotation:
            if val == annotation:
                annot_colors[val] = 0.0 / denom
            else:
                annot_colors[val] = (denom + i) / denom
        else:
            annot_colors[val] = i / denom

    samples_color = np.zeros(len(indices))
    for i in range(len(indices)):
        samples_color[i] = annot_colors[data.sample_annotations[indices[i]]]

    fig, ax = plt.subplots()
    if annotation:
        cm = "winter"
    else:
        cm = "nipy_spectral"
    sc = ax.scatter(np.arange(len(indices)), scores, c=samples_color, cmap=cm, s=5)
    ax.axhline(y=score_threshold, xmin=0, xmax=1, lw=1, ls="--", c="red")
    ann = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(-100, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    ann.set_visible(False)

    def update_annot(ind, sc):
        pos = sc.get_offsets()[ind["ind"][0]]
        ann.xy = pos
        text = "{}: {}".format(
            data.sample_ids[indices[ind["ind"][0]]],
            data.sample_annotations[indices[ind["ind"][0]]],
        )
        ann.set_text(text)

    def hover(event):
        vis = ann.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind, sc)
                ann.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    ann.set_visible(False)
                    fig.canvas.draw_idle()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind, sc)
                ann.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    ann.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "plot.png"), dpi=200)
    else:
        plt.show()
