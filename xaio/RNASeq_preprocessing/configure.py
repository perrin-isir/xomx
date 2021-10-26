import os
from xaio_config import root_dir

CSV_RNASeq_data = input(
    "Input path for CSV_RNASeq_data (see description in config.py):"
)
CSV_annotations = input(
    "Input path for CSV_annotations (see description in config.py):"
)
CSV_annot_types = input(
    "Input path for CSV_annot_types (see description in config.py):"
)

with open(root_dir + "/RNASeq_preprocessing/config.txt", "w") as f:
    f.write(os.path.abspath(os.path.expanduser(CSV_RNASeq_data)) + "\n")
    f.write(os.path.abspath(os.path.expanduser(CSV_annotations)) + "\n")
    f.write(os.path.abspath(os.path.expanduser(CSV_annot_types)) + "\n")
