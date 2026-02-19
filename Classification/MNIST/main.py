import numpy as np 
import pandas as pd 
import random 
random.seed(42)
np.random.seed(42)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline 

df_train = pd.read_csv("mnist_train.csv")

X = df_train.drop("label", axis=1)
y = df_train["label"]

num_cols = X.columns.tolist()

num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipe, num_cols)
], remainder="drop")

best_model = Pipeline(steps=[
    ("process", preprocessor), 
    ("model", RandomForestClassifier(n_estimators=200, max_depth=20, max_leaf_nodes=None, max_features="sqrt", n_jobs=-1, random_state=42))
])

df_test = pd.read_csv("mnist_test.csv")
X_test = df_test.drop("label", axis=1)

best_model.fit(X,y)
y_pred_test = best_model.predict(X_test) 

submission = pd.DataFrame({
    "index" : df_test["label"],
    "value" : y_pred_test
})

submission.to_csv("submission.csv", index=False)