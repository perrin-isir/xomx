from xaio_config import output_dir
from RNASeq_preprocessing.config import (
    CSV_RNASeq_data,
)
import pandas as pd
import numpy as np

save_dir = output_dir + "/dataset/RNASeq/"
nr_transcripts = np.load(save_dir + "nr_transcripts.npy", allow_pickle=True).item()
nr_samples = np.load(save_dir + "nr_samples.npy", allow_pickle=True).item()

rnaseq_data = pd.read_table(CSV_RNASeq_data, header=0, engine="c")
rnaseq_array = rnaseq_data.to_numpy()
data_array = np.zeros((nr_transcripts, nr_samples))
for i in range(nr_transcripts):
    row_value = rnaseq_array[i][0].split(",")
    data_array[i, :] = row_value[1:]

fp_data = np.memmap(
    save_dir + "raw_data.bin",
    dtype="float32",
    mode="w+",
    shape=(nr_transcripts, nr_samples),
)
fp_data[:] = data_array[:]
del fp_data
# Load with: np.array(np.memmap(save_dir + 'raw_data.bin', dtype='float32',
#                     mode='r', shape=(nrows, ncols)))
print("(21) " + "saved: " + save_dir + "raw_data.bin")
