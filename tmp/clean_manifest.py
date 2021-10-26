import requests
# import json
import pandas as pd
from io import StringIO
import numpy as np
from IPython import embed as e
assert e


# ./gdc-client download --retry-amount --no-annotations 5 -m manifest.txt

def gdc_create_manifest(disease_type, project_list, nr_of_cases_list):
    df_list = []
    for proj, nr_of_cases in zip(project_list, nr_of_cases_list):
        fields = ["file_name", "md5sum", "file_size", "state",
                  "cases.project.project_id"]
        fields = ",".join(fields)
        files_endpt = "https://api.gdc.cancer.gov/files"
        filters = {
            "op": "and",
            "content":[
                {
                "op": "in",
                "content":{
                    "field": "cases.project.project_id",
                    "value": [proj]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "cases.disease_type",
                        "value": [disease_type]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.data_category",
                        "value": ["Transcriptome Profiling"]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.type",
                        "value": ["gene_expression"]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.analysis.workflow_type",
                        "value": ["HTSeq - Counts"]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.data_format",
                        "value": ["TXT"]
                    }
                },
            ]
        }
        params = {
            "filters": filters,
            "fields": fields,
            "format": "TSV",
            "size": str(nr_of_cases)
            }
        response = requests.post(
            files_endpt, headers={"Content-Type": "application/json"}, json=params
        )
        df = pd.read_table(StringIO(response.content.decode("utf-8")))
        df = df.rename(columns={"file_name": "filename", "file_size": "size",
                                "md5sum": "md5",
                                "cases.0.project.project_id": "annotation"})
        df = df[["id", "filename", "md5", "size", "state", "annotation"]]
        df_list.append(df)
    return df_list

#
# def gdc_legacy_create_manifest(project_list, nr_of_cases_list):
#     df_list = []
#     for proj, nr_of_cases in zip(project_list, nr_of_cases_list):
#         fields = ["file_name", "md5sum",
#                   "cases.project.project_id", "cases.disease_type"]
#         fields = ",".join(fields)
#         files_endpt = "https://api.gdc.cancer.gov/legacy/files"
#         # files_endpt = "https://api.gdc.cancer.gov/files"
#         filters = {
#             "op": "and",
#             "content":[
#                 {
#                 "op": "in",
#                 "content":{
#                     "field": "cases.project.project_id",
#                     "value": [proj]
#                     }
#                 },
#                 {
#                     "op": "in",
#                     "content": {
#                         "field": "cases.samples.sample_type",
#                         "value": ["Primary Tumor"]
#                     }
#                 },
#                 {
#                 "op": "in",
#                 "content":{
#                     "field": "files.data_category",
#                     "value": ["Gene expression"]
#                     }
#                 },
#                 {
#                     "op": "in",
#                     "content": {
#                         "field": "files.data_type",
#                         "value": ["Gene expression quantification"]
#                     }
#                 },
#                 {
#                     "op": "in",
#                     "content": {
#                         "field": "files.tags",
#                         "value": ["normalized"]
#                     }
#                 },
#                 {
#                 "op": "in",
#                 "content":{
#                     "field": "files.experimental_strategy",
#                     "value": ["RNA-Seq"]
#                     }
#                 },
#                 {
#                 "op": "in",
#                 "content":{
#                     "field": "files.data_format",
#                     "value": ["TXT"]
#                     }
#                 }
#             ]
#         }
#         params = {
#             "filters": filters,
#             "fields": fields,
#             "format": "TSV",
#             "size": str(nr_of_cases)
#             }
#         response = requests.post(
#             files_endpt, headers={"Content-Type": "application/json"}, json=params
#         )
#         df = pd.read_table(StringIO(response.content.decode("utf-8")))
#         df = df.rename(columns={"file_name": "filename", "md5sum": "md5",
#                                 "cases.0.disease_type": "annotation",
#                                 "cases.0.project.project_id": "project"})
#         df = df[["id", "filename", "md5", "annotation", "project"]]
#         # df.to_csv(r'manifest.txt', header=True,
#         #           index=None, sep='\t', mode='w')
#         df_list.append(df)
#     return df_list


# df_list = gdc_create_manifest(["TCGA-BRCA", "TCGA-GBM", "TCGA-OV", "TCGA-LUAD", "TCGA-LUSC", "TCGA-], [5] * 2)
df_list = gdc_create_manifest("Adenomas and Adenocarcinomas",
                              ["TCGA-KIRC", "TCGA-THCA", "TCGA-PRAD", "TCGA-LUAD",
                               "TCGA-UCEC", "TCGA-COAD", "TCGA-LIHC", "TCGA-STAD",
                               "TCGA-KIRP", "TCGA-READ"], [5] * 10)

df_final = pd.concat(df_list)
df_final.to_csv(r'manifest.txt', header=True, index=False, sep='\t', mode='w')
e()