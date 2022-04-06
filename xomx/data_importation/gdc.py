import requests
import os
from io import StringIO
from pandas import read_table


def gdc_create_manifest(disease_type, project_list, nr_of_cases_list):
    df_list = []
    for proj, nr_of_cases in zip(project_list, nr_of_cases_list):
        fields = [
            "file_name",
            "md5sum",
            "file_size",
            "state",
            "cases.project.project_id",
        ]
        fields = ",".join(fields)
        files_endpt = "https://api.gdc.cancer.gov/files"
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {"field": "cases.project.project_id", "value": [proj]},
                },
                {
                    "op": "in",
                    "content": {"field": "cases.disease_type", "value": [disease_type]},
                },
                {
                    "op": "in",
                    "content": {
                        "field": "cases.samples.sample_type",
                        "value": ["Primary Tumor"],
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.data_category",
                        "value": ["Transcriptome Profiling"],
                    },
                },
                {
                    "op": "in",
                    "content": {"field": "files.type", "value": ["gene_expression"]},
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.analysis.workflow_type",
                        "value": ["STAR - Counts"],
                    },
                },
                {
                    "op": "in",
                    "content": {"field": "files.data_format", "value": ["TSV"]},
                },
            ],
        }
        max_number_of_cases = 1_000_000
        params = {
            "filters": filters,
            "fields": fields,
            "format": "TSV",
            "size": str(max_number_of_cases),
        }
        response = requests.post(
            files_endpt, headers={"Content-Type": "application/json"}, json=params
        )
        df = read_table(StringIO(response.content.decode("utf-8")))
        df = df.rename(
            columns={
                "file_name": "filename",
                "file_size": "size",
                "md5sum": "md5",
                "cases.0.project.project_id": "annotation",
            }
        )
        df = df[~df["filename"].str.endswith("tsv.gz")]
        df = df[["id", "filename", "md5", "size", "state", "annotation"]]
        df_list.append(df.head(nr_of_cases))
    return df_list


def gdc_create_data_matrix(dir_path, manifest_path):
    manifest = read_table(manifest_path)
    df_list = []
    nr_of_samples = manifest.shape[0]
    for i in range(nr_of_samples):
        if not i % 10:
            print("  " + str(i) + "/" + str(nr_of_samples), end="\r")
        if os.path.exists(
            os.path.join(dir_path, manifest["id"][i], manifest["filename"][i])
        ):
            df_list.append(
                read_table(
                    os.path.join(dir_path, manifest["id"][i], manifest["filename"][i]),
                    header=None,
                    delimiter="\t",
                    skiprows=6,
                )[[0, 3]]
                .rename(columns={3: manifest["id"][i]})
                .set_index(0)
            )

    df_total = df_list[0].join(df_list[1:])
    df_total.index.name = None
    return df_total
