"""
Boosted trees classifier with 128 input features and 2 output classes with keras and sklearn
"""
import os
import random
import typing as t
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.pipeline import Pipeline
import sigopt

# fix random seed for reproducibility
SEED = 33
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)  # type: ignore

print("loading dataset")
train = t.cast(pd.DataFrame, pd.read_csv("../input/train.csv.zip"))
test = t.cast(pd.DataFrame, pd.read_csv("../input/test.csv.zip"))

print("feature augmentation")
train["n_missing"] = train.isna().sum(axis=1)
test["n_missing"] = test.isna().sum(axis=1)
train["claim"] = train.loc[:, "claim"]

features = [col for col in train.columns if col not in ["claim", "id"]]
X_train, y_train = train[features], train["claim"]

print("training model...")
input_shape = t.cast(np.ndarray, X_train).shape[1]

pipe = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(strategy="constant", missing_values=np.nan, fill_value=-999),
        ),
        (
            "scaler",
            QuantileTransformer(
                n_quantiles=sigopt.get_parameter("n_quantiles", 128),
                output_distribution="uniform",
            ),
        ),
        (
            "model",
            xgb.XGBClassifier(
                **{
                    "n_estimators": sigopt.get_parameter("n_estimators", default=200),
                    "reg_lambda": sigopt.get_parameter("reg_lambda", default=1),
                    "reg_alpha": sigopt.get_parameter("reg_alpha", default=1),
                    "subsample": sigopt.get_parameter("subsample", default=1),
                    "colsample_bytree": sigopt.get_parameter(
                        "colsample_bytree", default=0.6
                    ),
                    "max_depth": sigopt.get_parameter("max_depth", default=3),
                    "min_child_weight": sigopt.get_parameter(
                        "min_child_weight", default=5
                    ),
                    "gamma": sigopt.get_parameter("gamma", default=1),
                    "learning_rate": sigopt.get_parameter(
                        "learning_rate", default=0.0001
                    ),
                    "tree_method": "hist",
                    "booster": "gbtree",
                }
            ),
        ),
    ]
)

pipe.fit(X_train, y_train)

print("Generating predictions...")

claim_prob = pipe.predict_proba(test[features])
claim = [1 if i > 0.5 else 0 for i in claim_prob[:, 1]]
submission = pd.DataFrame({"id": test["id"], "claim": claim})
submission.to_csv("./output/submission.csv", index=False)

print("Done!")
