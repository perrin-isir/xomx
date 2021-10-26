import numpy as np
import scanpy as sc


def var_mean_values(adata: sc.AnnData) -> np.ndarray:
    return np.array([np.mean(adata.X[:, i]) for i in range(adata.n_vars)])


def var_standard_deviations(adata: sc.AnnData) -> np.ndarray:
    return np.array([np.std(adata.X[:, i]) for i in range(adata.n_vars)])


def var_indices(adata: sc.AnnData) -> dict:
    vi_dict = {}
    for i, s_id in enumerate(adata.var_names):
        vi_dict[s_id] = i
    return vi_dict


def obs_indices(adata: sc.AnnData) -> dict:
    oi_dict = {}
    for i, s_id in enumerate(adata.obs_names):
        oi_dict[s_id] = i
    return oi_dict


def all_labels(labels) -> np.ndarray:
    return np.array(list(dict.fromkeys(labels)))


def indices_per_label(labels) -> dict:
    i_per_label = {}
    for i, annot in enumerate(labels):
        i_per_label.setdefault(annot, [])
        i_per_label[annot].append(i)
    return i_per_label


def train_and_test_indices(
    adata: sc.AnnData, indices_per_label_key, test_train_ratio=0.25
):
    sc._utils.view_to_actual(adata)
    train_indices_per_label = {}
    test_indices_per_label = {}
    for annot in adata.uns[indices_per_label_key]:
        idxs = np.random.permutation(adata.uns[indices_per_label_key][annot])
        cut = np.floor(len(idxs) * test_train_ratio).astype("int") + 1
        test_indices_per_label[annot] = idxs[:cut]
        train_indices_per_label[annot] = idxs[cut:]
    train_indices = np.concatenate(list(train_indices_per_label.values()))
    test_indices = np.concatenate(list(test_indices_per_label.values()))
    # from IPython import embed as e; e() ; quit()
    adata.uns["train_indices_per_label"] = train_indices_per_label
    adata.uns["test_indices_per_label"] = test_indices_per_label
    adata.uns["train_indices"] = train_indices
    adata.uns["test_indices"] = test_indices
