from xaio.xaio_config import output_dir, xaio_tag
from xaio.tools.basic_tools import (
    load,
    FeatureTools,
)
from xaio.tools.volcano_plot import VolcanoPlot
from xaio.tools.feature_selection.RFEExtraTrees import RFEExtraTrees

from xaio.tools.feature_selection.RFENet import RFENet

# from tools.classifiers.LinearSGD import LinearSGD
import os

from IPython import embed as e

assert e

_ = RFEExtraTrees, RFENet

data = load()
# data = load()
gt = FeatureTools(data)

# annotation = "Acute myeloid leukemia"
# annotation = "Diffuse large B-cell lymphoma"
# annotation = "Glioblastoma multiforme"
# annotation = "Lung adenocarcinoma"
# annotation = "Lung squamous cell carcinoma"
# annotation = "Pheochromocytoma and paraganglioma"
# annotation = "Small cell lung cancer"
# annotation = "Uveal melanoma"
# annotation = "Skin cutaneous melanoma"
annotation = "Brain lower grade glioma"
# annotation = "TCGA-LGG_Primary Tumor"
# annotation = "Breast invasive carcinoma"


save_dir = os.path.expanduser(
    output_dir + "/results/" + xaio_tag + "/" + annotation.replace(" ", "_")
)

fs1 = RFENet(data, annotation, init_selection_size=4000)
fs2 = RFEExtraTrees(data, annotation, init_selection_size=4000)

# fs1.load(save_dir)
fs2.load(save_dir)


vp = VolcanoPlot(data, annotation)
# vp.init()
vp.init(fs1.current_feature_indices)
vp.plot(fs2.current_feature_indices)
e()
