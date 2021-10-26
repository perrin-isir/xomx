# from xaio.tools.basic_tools import XAIOData
# from xaio.xaio_config import output_dir
from IPython import embed as e
import pandas as pd
import numpy as np

# import random
# import os

assert e

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


e()
quit()
