import numpy as np 
import pandas as pd 
import random 
random.seed(42)
np.random.seed(42)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor


df_train = pd.read_csv("train_dataset.csv")

X = df_train.drop(columns=["target"])
y = df_train["target"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

ord_cols = [col for col in cat_cols if col.startswith("ord_")]
nom_cols = [col for col in cat_cols if col.startswith("cat_")]


num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())        
])

ord_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder())
])

nom_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder())
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_cols),
    ("ord", ord_pipeline, ord_cols),
    ("nom", nom_pipeline, nom_cols)
])

final_model = Pipeline(steps=[
    ("preprocessor", preprocessor), 
    ("regressor", RandomForestRegressor(max_depth= None, max_features= None, min_samples_split= 2, n_estimators= 300, n_jobs= -1, random_state=42))
])

final_model.fit(X, y)

df_test = pd.read_csv("test_dataset.csv")

y_pred_test = final_model.predict(df_test)

submission = pd.DataFrame({
    "index" : range(len(y_pred_test)),
    "value" : y_pred_test
})
submission.to_csv("submission.csv", index=False)