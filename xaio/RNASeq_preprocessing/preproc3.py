from xaio.xaio_config import output_dir
from xaio.tools.basic_tools import XAIOData

# from IPython import embed as e

data = XAIOData()
data.save_dir = output_dir + "/dataset/RNASeq/"

data.load(["raw"])
data.compute_normalization("log")
data.data_array["raw"] = None
data.save()
