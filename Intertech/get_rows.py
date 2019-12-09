from azureml.core import Workspace, Datastore
from azureml.core.authentication import ServicePrincipalAuthentication
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description="Start a propensity model serving")
parser.add_argument('--pipeline_dir', dest="pipeline_dir", required=True)
parser.add_argument('--threshold', dest="threshold", required=True)
parser.add_argument('--output_rows', dest="output_rows", required=True)

args = parser.parse_args()

path = os.path.join(args.pipeline_dir,"dataset.csv")
df = pd.read_csv(path)
df = df.loc[df.predictions > int(args.threshold)]

os.makedirs(args.output_rows, exist_ok=True)
output_path = os.path.join(args.output_rows,"rows.csv")
df.to_csv(path_or_buf = output_path,index=False)



