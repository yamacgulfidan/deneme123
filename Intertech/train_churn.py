#!/usr/bin/env python
# coding: utf-8

# In[17]:


from azure.storage.blob import BlockBlobService
import pandas as  pd 
from io import StringIO
from azureml.core.model import Model
from azureml.core import Experiment
from azureml.core import Workspace, Experiment, Run
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from azureml.core.authentication import ServicePrincipalAuthentication

## Data Lake GEN2 account'umuza bağlanıyoruz
block_blob_service = BlockBlobService(account_name = 'lakestorage', account_key = 'RVg8vNu4rddB86M6G38qnpi2MzTBv9JGxbAgixEQHUSJtrlBtItQraAN7OVzAlm0EbqWQEsv9eO3k4BdZ34flA==')
# Container name intertech deneme , blob name bank-additional-full.csv
textresult = block_blob_service.get_blob_to_text("intertechdeneme","bank-additional-full.csv")
df = pd.read_csv(StringIO(textresult.content),sep=";")

df.drop("duration",axis=1,inplace=True)


df["is_contacted"] = [1 if x != 999  else 0  for x in df.pdays] # Is the customer contacted for another campaign previously?
df["contact_ratio"] = (df["campaign"]) / (df["previous"]+0.001) # Customer's contact ratio for this campaign and the campaigns before.




scaler = StandardScaler()
df["cons.price.idx"] = scaler.fit_transform(np.array(df["cons.price.idx"]).reshape(-1,1))



target = df["y"]
df.drop("y",axis=1,inplace=True)
col_names = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=col_names,drop_first=True)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df,target, test_size=0.2, random_state=0)
data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}


spa=ServicePrincipalAuthentication("9d396536-b01d-424e-ace8-2a339bc8e502", "e9311799-43b1-4a27-9e08-266cc96a2c05","K_?XRu6bSX/DV3.WYsuS3cwxjgOg4DzO" , _enable_caching=False)
ws = Workspace.get(name='acc_ws',auth=spa, subscription_id='f30951ea-6926-40a7-9c19-adeba1c67ec4', resource_group='acc_ws_rg')
exp = Experiment(workspace=ws, name="exp")
run = exp.start_logging()


clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(data["train"]["X"], data["train"]["y"])
preds = clf.predict(data["test"]["X"])

run.log("accuracy", accuracy_score(data["test"]["y"],preds))




filename = 'outputs/propensity_model.pkl'
joblib.dump(clf, filename)



# Registering model with "Model" class.
model = Model.register(model_name='propensity_model', model_path='outputs/propensity_model.pkl',  workspace = ws)




# Registering model with run object
# model = run.register_model(model_name='churn_model', model_path='outputs/churn_model.pkl')


run.complete()





