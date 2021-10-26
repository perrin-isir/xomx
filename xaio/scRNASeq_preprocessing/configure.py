import os
from xaio.xaio_config import root_dir

scRNASeq_data = input("Input path for scRNASeq_data (see description in config.py):")
scRNASeq_features = input(
    "Input path for scRNASeq_features (see description in config.py):"
)
scRNASeq_barcodes = input(
    "Input path for scRNASeq_barcodes (see description in config.py):"
)

with open(root_dir + "/scRNASeq_preprocessing/config.txt", "w") as f:
    f.write(os.path.abspath(os.path.expanduser(scRNASeq_data)) + "\n")
    f.write(os.path.abspath(os.path.expanduser(scRNASeq_features)) + "\n")
    f.write(os.path.abspath(os.path.expanduser(scRNASeq_barcodes)) + "\n")
