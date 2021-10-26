from xaio.data_importation.gdc import gdc_create_manifest, gdc_create_data_matrix
from xaio.tools.basic_tools import XAIOData, confusion_matrix, matthews_coef
from xaio.tools.feature_selection.RFEExtraTrees import RFEExtraTrees
from xaio.tools.classifiers.multiclass import ScoreBasedMulticlass
from xaio.tools.interfaces.anndata import anndata_interface

assert gdc_create_manifest
assert gdc_create_data_matrix
assert XAIOData
assert confusion_matrix
assert matthews_coef
assert RFEExtraTrees
assert ScoreBasedMulticlass
assert anndata_interface
