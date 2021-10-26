from xaio_config import output_dir
from RNASeq_preprocessing.config import (
    CSV_RNASeq_data,
)
import pandas as pd
import numpy as np

save_dir = output_dir + "/dataset/RNASeq/"
nr_transcripts = np.load(save_dir + "nr_transcripts.npy", allow_pickle=True).item()
nr_samples = np.load(save_dir + "nr_samples.npy", allow_pickle=True).item()

epsilon_shift = 1.0
rnaseq_data = pd.read_table(CSV_RNASeq_data, header=0, engine="c")
rnaseq_array = rnaseq_data.to_numpy()
data_array = np.zeros((nr_transcripts, nr_samples))
for i in range(nr_transcripts):
    row_value = rnaseq_array[i][0].split(",")
    data_array[i, :] = row_value[1:]
    data_array[i, :] = np.log(data_array[i, :] + epsilon_shift)

np.save(save_dir + "epsilon_shift.npy", epsilon_shift)
# Load with: np.load(save_dir + 'epsilon_shift.npy', allow_pickle=True).item()
print("(18) " + "saved: " + save_dir + "epsilon_shift.npy")

maxlog = np.max(data_array)
np.save(save_dir + "maxlog.npy", maxlog)
# Load with: np.load(save_dir + 'maxlog.npy', allow_pickle=True).item()
print("(19) " + "saved: " + save_dir + "maxlog.npy")

for i in range(nr_transcripts):
    data_array[i, :] = (data_array[i, :] - np.log(epsilon_shift)) / (
        maxlog - np.log(epsilon_shift)
    )

fp_data = np.memmap(
    save_dir + "lognorm_data.bin",
    dtype="float32",
    mode="w+",
    shape=(nr_transcripts, nr_samples),
)
fp_data[:] = data_array[:]
del fp_data
# Load with: np.array(np.memmap(save_dir + 'lognorm_data.bin', dtype='float32',
#                     mode='r', shape=(nrows, ncols)))
print("(20) " + "saved: " + save_dir + "lognorm_data.bin")
