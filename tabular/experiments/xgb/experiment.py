"""
Boosted trees classifier with 128 input features and 2 output classes with keras and sklearn
"""
import os
import random
import typing as t
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import xgboost as xgb
from sklearn.pipeline import Pipeline
import sigopt

sigopt.log_model("XGBClassifier")

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

print("split, split, split...")
X_train, X_test, y_train, y_test = train_test_split(
    train[features], train["claim"], test_size=0.1, random_state=42
)

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

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

results = cross_val_score(
    pipe,
    X_train,
    y_train,
    cv=skfold,
    scoring="roc_auc",
    verbose=1,
    n_jobs=1,
)

print("Baseline: %.2f%% %.2f%%)" % (results.mean() * 100, results.std() * 100))
sigopt.log_metric("cv_score", results.mean(), stddev=results.std())

pipe.fit(X_train, y_train)

y_prob = pipe.predict_proba(X_test)
y_pred = [1 if i > 0.5 else 0 for i in y_prob[:, 1]]

roc = roc_auc_score(y_test, y_pred)
print(f"roc: {roc}")
sigopt.log_metric("roc", roc)

accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy}")
sigopt.log_metric("accuracy", accuracy)

precision = precision_score(y_test, y_pred)
print(f"precision: {precision}")
sigopt.log_metric("precision", precision)

recall = recall_score(y_test, y_pred)
print(f"recall: {recall}")
sigopt.log_metric("recall", recall)

f1 = f1_score(y_test, y_pred)
print(f"f1: {f1}")
sigopt.log_metric("f1", f1)

# confusion = confusion_matrix(y_test, y_pred)
# print(f"confusion: {confusion}")
# sigopt.log_metric("confusion", confusion)

# print("Generating predictions...")
#
# claim_prob = pipe.predict_proba(test[features])
# claim = [1 if i > 0.5 else 0 for i in claim_prob[:, 1]]
# submission = pd.DataFrame({"id": test["id"], "claim": claim})
# submission.to_csv("./output/submission.csv", index=False)

print("Done!")
