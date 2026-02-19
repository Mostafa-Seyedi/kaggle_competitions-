import numpy as np
import pandas as pd 
import random 
random.seed(42)
np.random.seed(42)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


df_train = pd.read_csv('train.csv')
X = df_train.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = df_train['Survived']

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")), 
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")), 
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_cols), 
    ("cat", cat_pipeline, cat_cols)
], remainder="drop")


best_model = Pipeline(steps=[
    ("process", preprocessor),
    # got from the hyperparameter tunning in the ipynb file 
    ("model", RandomForestClassifier(n_estimators=100, max_features=None,max_leaf_nodes=8, max_depth=3, n_jobs=-1 , random_state=42))
])

df_test = pd.read_csv("test.csv")
X_test = df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
best_model.fit(X,y)

y_pred_test = best_model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId" : df_test['PassengerId'], 
    "survived" : y_pred_test 
})

submission.to_csv("submission.csv", index=False)

