"""
Boosted trees classifier with 128 input features and 2 output classes with keras and sklearn
"""
import typing as t
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.metrics import AUC
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
import sigopt

sigopt.log_model("DNN")

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

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
    train[features], train["claim"], test_size=0.33, random_state=42
)

print("training model...")
input_shape = t.cast(np.ndarray, X_train).shape[1]

# model params
n_layers = sigopt.get_parameter("n_layers", default=3)
n_hidden = sigopt.get_parameter("n_hidden", default=8)
dropout = sigopt.get_parameter("dropout", default=0.2)
activation = sigopt.get_parameter("activation", default="relu")
n_quantiles = sigopt.get_parameter("n_quantiles", default=512)
quantiles_output_distribution = sigopt.get_parameter(
    "quantiles_output_distribution", default="uniform"
)


def baseline_model():
    model = Sequential()
    model.add(Input(input_shape))
    for _ in range(n_layers):
        model.add(Dense(n_hidden, activation=activation))
        model.add(Dropout(dropout))
    model.add(Dense(2, activation="softmax"))

    auc = AUC(name="aucroc")

    model.compile(
        optimizer=Adam(learning_rate=0.1),
        loss="categorical_crossentropy",
        metrics=["accuracy", auc],
    )

    return model


print(baseline_model().summary())

estimator = KerasClassifier(
    build_fn=baseline_model,
    epochs=100,
    batch_size=512,
    callbacks=[
        ReduceLROnPlateau(monitor="aucroc", factor=0.2, patience=3, min_lr=0.0001),
        EarlyStopping(monitor="loss", patience=5, min_delta=0.005),
        TerminateOnNaN(),
    ],
    verbose=0,
)


pipe = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(strategy="constant", missing_values=np.nan, fill_value=-999),
        ),
        (
            "scaler",
            QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution=quantiles_output_distribution,
            ),
        ),
        ("model", estimator),
    ]
)

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

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

confusion = confusion_matrix(y_test, y_pred)
print(f"confusion: {confusion}")

# print("Generating predictions...")
#
# claim_prob = pipe.predict_proba(test[features])
# claim = [1 if i > 0.5 else 0 for i in claim_prob[:, 1]]
# submission = pd.DataFrame({"id": test["id"], "claim": claim})
# submission.to_csv("./output/submission.csv", index=False)

print("Done!")
