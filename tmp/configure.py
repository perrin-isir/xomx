import os
from xaio_config import root_dir

output_dir = input("output_dir (see description in xaio_config.py)?")

with open(root_dir + "/xaio_config.txt", "w") as f:
    f.write(os.path.abspath(os.path.expanduser(output_dir)) + "\n")
