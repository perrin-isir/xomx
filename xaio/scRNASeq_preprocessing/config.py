from xaio_config import root_dir

with open(root_dir + "/scRNASeq_preprocessing/config.txt", "r") as f:
    lines = f.readlines()

""" User-defined variables (run configure.py to define them): """

scRNASeq_data = lines[0].rstrip()
"""
10X single-cell RNASeq data format
"""

scRNASeq_features = lines[1].rstrip()
"""
10X single-cell RNASeq data format
"""

scRNASeq_barcodes = lines[2].rstrip()
"""
10X single-cell RNASeq data format
"""
