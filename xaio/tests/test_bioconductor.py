import rpy2.robjects as robjects
from IPython import embed as e

assert e

# Do this only the first time (to install bioconductor packages):
# robjects.r('install.packages("BiocManager", '
#            'repos="http://cran.r-project.org")')
# robjects.r('BiocManager::install("seqLogo")')
robjects.r("library(seqLogo)")

# biocinstaller = importr("BiocInstaller")
# biocinstaller.biocLite("seqLogo")
#
# # load the installed package "seqLogo"
# seqlogo = importr("seqLogo")

e()
