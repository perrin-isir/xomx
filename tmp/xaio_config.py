import os
import tempfile
from IPython import embed as e

assert e

rdir = os.path.abspath(__file__)
assert rdir.endswith("xaio_config.py")
root_dir = rdir[:-14]
"""
root_dir is the root directory of the xaio library.
"""

xaioconfig = os.path.join(root_dir, "xaio_config.txt")

""" User-defined variable (run configure.py to set it): """

if os.path.exists(xaioconfig):
    with open(xaioconfig, "r") as f:
        lines = f.readlines()

    if not lines:
        output_dir = tempfile.gettempdir()
    else:
        output_dir = lines[0].rstrip()
    """
    output_dir is the directory in which all outputs will be saved.
    It can be configured by running `python configure.py`, otherwise its
    value is the default temporary directory.
    """
else:
    output_dir = tempfile.gettempdir()
