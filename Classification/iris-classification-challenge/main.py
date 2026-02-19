import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random 
random.seed(32)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score

df_train = pd.read_csv("train.csv")

X = df_train.drop(columns=["species"])
y = df_train["species"]
le = LabelEncoder()
y = le.fit_transform(y)

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
# cat_cols = X.select_dtypes(include=["object"]).columns

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler" , MinMaxScaler())
])



preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, num_cols),
], remainder="drop")


prams = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "max_features": [None,"sqrt", "log2"],
    "n_jobs": [-1]
}

scores = []
for config in ParameterGrid(prams):
    model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42, **config))
])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average="weighted")
    print(config, "F1 Score:", score)
    scores.append((score))


best_score = max(scores)
best_index = scores.index(best_score)
best_config = list(ParameterGrid(prams))[best_index]

final_model = Pipeline(steps=[
    ("preprocessor", preprocessor), 
    ("model", RandomForestClassifier(random_state=42, **best_config))
])

final_model.fit(X,y)
df_test = pd.read_csv("test.csv")
df_test = df_test.rename(columns={"seepal_width": "sepal_width"})

X_test_submit = df_test.drop(columns=["id"])
y_pred_test = final_model.predict(X_test_submit)
submission = pd.DataFrame({
    "id": df_test["id"],
    "species": le.inverse_transform(y_pred_test)
})
submission.to_csv("submission.csv", index=False)