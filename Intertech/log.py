from sklearn import metrics
import pandas as pd
import os
import argparse
import numpy as np 

parser = argparse.ArgumentParser(description="Start a propensity model serving")
parser.add_argument('--pipeline_dir', dest="pipeline_dir", required=True)
parser.add_argument('--output_dir', dest="output_dir", required=True)
args = parser.parse_args()


path = os.path.join(args.pipeline_dir,"dataset.csv")
df = pd.read_csv(path)

path2 = os.path.join(args.pipeline_dir,"target.csv")
target = pd.read_csv(path2)

target.loc[target["y"]=="yes","y"] = 1
target.loc[target["y"]=="no","y"] = 0
preds = df.iloc[:,-1]

fpr, tpr, thresholds = metrics.roc_curve(target,preds)
auc = pd.DataFrame([metrics.auc(fpr, tpr)],columns=["auc"])


os.makedirs(args.output_dir, exist_ok=True)
auc_path = os.path.join(args.output_dir,"auc.csv")
auc.to_csv(path_or_buf = auc_path,index=False)

