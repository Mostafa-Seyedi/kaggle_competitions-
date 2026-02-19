import numpy as np 
import pandas as pd 
import random 
random.seed(42)
np.random.seed(42)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer   
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

df_train = pd.read_csv("development.csv")

X = df_train.drop(columns=["target"])
y = df_train["target"]

num_cols = [f"var_{i}" for i in range(0, 10)]
ord_cols = [f"var_{i}" for i in range(10, 15)]
cat_cols = [f"var_{i}" for i in range(15, 21)]


# As most of values in var_20 are missing (which is categorical), we can remove this column
cat_cols.remove("var_20")

num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

ord_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder())
])

cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),   
    ("encoder", OneHotEncoder())
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipe, num_cols),
    ("ord", ord_pipe, ord_cols),
    ("cat", cat_pipe, cat_cols)
])

final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    # got from hyperparameter tuning in explore.ipynb
    ("regressor", RandomForestRegressor(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200, n_jobs=-1, random_state=42))
])

# train with full training data
final_model.fit(X, y)

df_test = pd.read_csv("evaluation.csv")
y_pred_test = final_model.predict(df_test)

submission = pd.DataFrame({ 
    "index": range(len(y_pred_test)),
    "value": y_pred_test
})
submission.to_csv("submission.csv", index=False)