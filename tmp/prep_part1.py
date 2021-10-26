from xaio_config import output_dir
from RNASeq_preprocessing.config import (
    CSV_RNASeq_data,
    CSV_annotations,
    CSV_annot_types,
)
import os
import pandas as pd
import numpy as np

save_dir = output_dir + "/dataset/RNASeq/"
if not (os.path.exists(save_dir)):
    os.makedirs(save_dir, exist_ok=True)

rnaseq_read = pd.read_table(CSV_RNASeq_data, sep=",", header=0, engine="c", nrows=1)
col_names = rnaseq_read.columns.values[1:]
np.save(save_dir + "samples_id.npy", col_names)
# Load with: np.load(save_dir + 'samples_ID.npy', allow_pickle=True)
print("(1) " + "saved: " + save_dir + "samples_id.npy")

annot_dict = dict(pd.read_csv(CSV_annotations, sep=",").to_numpy())
np.save(save_dir + "annot_dict.npy", annot_dict)
# Load with: np.load(save_dir + 'annot_dict.npy', allow_pickle=True).item()
print("(2) " + "saved: " + save_dir + "annot_dict.npy")

annot_values = np.array(list(dict.fromkeys(list(annot_dict.values()))))
np.save(save_dir + "annot_values.npy", annot_values)
# Load with: np.load(save_dir + 'annot_values.npy', allow_pickle=True)
print("(3) " + "saved: " + save_dir + "annot_values.npy")

annot_index = {}
for num, name in enumerate(col_names):
    k = annot_dict[name]
    annot_index.setdefault(k, [])
    annot_index[k].append(num)
np.save(save_dir + "annot_index.npy", annot_index)
# Load with: np.load(save_dir + 'annot_index.npy', allow_pickle=True).item()
print("(4) " + "saved: " + save_dir + "annot_index.npy")

rnaseq_data = pd.read_table(CSV_RNASeq_data, header=0, engine="c")
nrows = rnaseq_data.shape[0]
np.save(save_dir + "nr_transcripts.npy", nrows)
# Load with: np.load(save_dir + 'nr_transcripts.npy', allow_pickle=True).item()
print("(5) " + "saved: " + save_dir + "nr_transcripts.npy")

rnaseq_array = rnaseq_data.to_numpy()
ncols = len(rnaseq_array[0][0].split(",")) - 1
np.save(save_dir + "nr_samples.npy", ncols)
# Load with: np.load(save_dir + 'nr_samples.npy', allow_pickle=True).item()
print("(6) " + "saved: " + save_dir + "nr_samples.npy")

data_array = np.zeros((nrows, ncols))
original_transcripts = np.empty((nrows,), dtype=object)
for i in range(nrows):
    row_value = rnaseq_array[i][0].split(",")
    original_transcripts[i] = row_value[0]
    data_array[i, :] = row_value[1:]
    if not i % (nrows // 100):
        print(i // (nrows // 100), "%\r", end="")
print()
np.save(save_dir + "transcripts.npy", original_transcripts)
# Load with: np.load(save_dir + 'transcripts.npy', allow_pickle=True)
print("(7) " + "saved: " + save_dir + "transcripts.npy")

mean_expressions = [np.mean(data_array[i, :]) for i in range(nrows)]
np.save(save_dir + "mean_expressions.npy", mean_expressions)
# Load with: np.load(save_dir + 'mean_expressions.npy', allow_pickle=True)
print("(8) " + "saved: " + save_dir + "mean_expressions.npy")

std_expressions = [np.std(data_array[i, :]) for i in range(nrows)]
np.save(save_dir + "std_expressions.npy", std_expressions)
# Load with: np.load(save_dir + 'std_expressions.npy', allow_pickle=True)
print("(9) " + "saved: " + save_dir + "std_expressions.npy")

for i in range(nrows):
    for j in range(ncols):
        if std_expressions[i] == 0.0:
            data_array[i, j] = 0.0
        else:
            data_array[i, j] = (
                data_array[i, j] - mean_expressions[i]
            ) / std_expressions[i]

fp_data = np.memmap(
    save_dir + "data.bin", dtype="float32", mode="w+", shape=(nrows, ncols)
)
fp_data[:] = data_array[:]
del fp_data
# Load with: np.array(np.memmap(save_dir + 'data.bin', dtype='float32', mode='r',
#                     shape=(nr_samples, nr_transcripts)))
print("(10) " + "saved: " + save_dir + "data.bin")

annot_index_train = {}
annot_index_test = {}
for key in annot_index:
    np.random.shuffle(annot_index[key])
    cut = len(annot_index[key]) // 4 + 1
    annot_index_test[key] = annot_index[key][:cut]
    annot_index_train[key] = annot_index[key][cut:]
np.save(save_dir + "annot_index_train.npy", annot_index_train)
# Load with: np.load(save_dir + 'annot_index_train.npy', allow_pickle=True).item()
print("(11) " + "saved: " + save_dir + "annot_index_train.npy")

np.save(save_dir + "annot_index_test.npy", annot_index_test)
# Load with: np.load(save_dir + 'annot_index_test.npy', allow_pickle=True).item()
print("(12) " + "saved: " + save_dir + "annot_index_test.npy")

gene_dict = {}
for i, elt in enumerate(original_transcripts):
    gene_dict[elt.split("|")[1]] = i
np.save(save_dir + "gene_dict.npy", gene_dict)
# Load with: np.load(save_dir + 'gene_dict.npy', allow_pickle=True).item()
print("(13) " + "saved: " + save_dir + "gene_dict.npy")

expressions_on_training_sets = {}
for cat in annot_values:
    expressions_on_training_sets[cat] = []
    for i in range(nrows):
        expressions_on_training_sets[cat].append(
            np.mean([data_array[i, j] for j in annot_index_train[cat]])
        )
np.save(save_dir + "expressions_on_training_sets.npy", expressions_on_training_sets)
# Load with: np.load(save_dir + 'expressions_on_training_sets.npy',
#                    allow_pickle=True).item()
print("(14) " + "saved: " + save_dir + "expressions_on_training_sets.npy")

expressions_on_training_sets_argsort = {}
for cat in annot_values:
    expressions_on_training_sets_argsort[cat] = np.argsort(
        expressions_on_training_sets[cat]
    )[::-1]
np.save(
    save_dir + "expressions_on_training_sets_argsort.npy",
    expressions_on_training_sets_argsort,
)
# Load with: np.load(save_dir + 'expressions_on_training_sets_argsort.npy',
#                    allow_pickle=True).item()
print("(15) " + "saved: " + save_dir + "expressions_on_training_sets_argsort.npy")

annot_types = dict(pd.read_csv(CSV_annot_types, sep=",").to_numpy())
np.save(save_dir + "annot_types.npy", annot_types)
# Load with: np.load(save_dir + 'annot_types.npy', allow_pickle=True).item()
print("(16) " + "saved: " + save_dir + "annot_types.npy")

annot_types_dict = {}
for id_ in col_names:
    label_ = annot_dict[id_]
    annot_types_dict.setdefault(label_, set()).add(annot_types[id_])

np.save(save_dir + "annot_types_dict.npy", annot_types_dict)
# Load with: np.load(save_dir + 'annot_types_dict.npy', allow_pickle=True).item()
print("(17) " + "saved: " + save_dir + "annot_types_dict.npy")
