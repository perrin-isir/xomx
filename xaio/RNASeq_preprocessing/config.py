from xaio_config import root_dir

with open(root_dir + "/RNASeq_preprocessing/config.txt", "r") as f:
    lines = f.readlines()

""" User-defined variables (run configure.py to define them): """

CSV_RNASeq_data = lines[0].rstrip()
"""
CSV_RNASeq_data is the path towards a .csv file containing TPM-normalized RNASeq data.
It is a matrix in which each column corresponds to a sample, and each row corresponds
to the levels of expression of a particular transcript accross all samples.
The levels of expression are in TPM (Transcripts Per Million), so for each sample the
sum of all values in the column should be equal to 1,000,000.
The first line of the file must have the structure:
"", "sample1-ID-string", "sample2-ID-string", ...
The following lines must have the structure:
"feature1-IDstring", value_for_sample1, value_for_sample2, value_for_sample3...
Example:
"", "1eRzrg231Te-ID-sample1", "R34ytj9238l-ID-sample2", "5JRbret012l-ID-sample3", ...
"ENSG00000000003.10|TSPAN6",71.6,26.3,49.5,46.7,75.8,180.1,...
"ENSG00000000005.5|TNMD",0.1,0.4,0,0.1,0,0.1,0,0,0.5,0,0,0,...
...
"""

CSV_annotations = lines[1].rstrip()
"""
CSV_annotations is the path towards a .csv file defining labels for all the samples.
The first line of the file should be: "", "Diagnosis".
The following lines must have the structure <sample-ID-string>, <diagnosis string>.
Example:
"", "Diagnosis"
"1eRzrg231Te-ID-sample1", "Adrenocortical carcinoma"
"R34ytj9238l-ID-sample2", "Bladder urothelial carcinoma"
...
"""

CSV_annot_types = lines[2].rstrip()
"""
CSV_annot_types is the path towards a .csv file defining the origin of all the samples.
The first line of the file should be:  "", "Origin".
The following lines must have the structure <sample-ID-string>, <origin string>.
Example:
"", "Origin"
"1eRzrg231Te-ID-sample1", "TCGA-ACC_Primary Tumor"
"R34ytj9238l-ID-sample2", "TCGA-BLCA_Primary Tumor"
...
"""
