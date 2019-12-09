import os
import argparse
import numpy as np
import shutil
from azureml.core.model import Model
from azure.storage.blob import BlockBlobService
import pandas as pd 
from io import StringIO
from sklearn.preprocessing import StandardScaler
from azureml.core import Workspace, Datastore
from azureml.core.authentication import ServicePrincipalAuthentication
from sklearn.externals import joblib

parser = argparse.ArgumentParser(description="Start a propensity model serving")
parser.add_argument('--model_name', dest="model_name", required=True)
parser.add_argument('--dataset_path', dest="dataset_path", required=True)
parser.add_argument('--pipeline_dir', dest="pipeline_dir", required=True)
args = parser.parse_args()




spa=ServicePrincipalAuthentication("9d396536-b01d-424e-ace8-2a339bc8e502", "e9311799-43b1-4a27-9e08-266cc96a2c05","K_?XRu6bSX/DV3.WYsuS3cwxjgOg4DzO" , _enable_caching=False)
ws = Workspace.get(name='acc_ws',auth=spa, subscription_id='f30951ea-6926-40a7-9c19-adeba1c67ec4', resource_group='acc_ws_rg')



ds = ws.get_default_datastore()



model = joblib.load(filename=args.model_name)

df = pd.read_csv(os.path.join(args.dataset_path,"bank-additional-full.csv"),sep=";")


df.drop("duration",axis=1,inplace=True)

df["is_contacted"] = [1 if x != 999  else 0  for x in df.pdays] # Is the customer contacted for another campaign previously?
df["contact_ratio"] = (df["campaign"]) / (df["previous"]+0.001) # Customer's contact ratio for this campaign and the campaigns before.

scaler = StandardScaler()
df["cons.price.idx"] = scaler.fit_transform(np.array(df["cons.price.idx"]).reshape(-1,1))

target = df["y"]
df.drop("y",axis=1,inplace=True)



col_names = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=col_names,drop_first=True)


preds = model.predict_proba(df)
preds = preds[:,-1]
preds = pd.DataFrame.from_records(np.reshape(preds,(-1,1)))
preds.columns=["predictions"]

df = pd.concat([df.reset_index(drop=True),preds],ignore_index=False,axis=1,sort=False)


os.makedirs(args.pipeline_dir, exist_ok=True)
path = os.path.join(args.pipeline_dir,"dataset.csv")
path2 = os.path.join(args.pipeline_dir,"target.csv")

df.to_csv(path_or_buf = path,index=False)
target.to_csv(path_or_buf = path2,index=False,header="target")



