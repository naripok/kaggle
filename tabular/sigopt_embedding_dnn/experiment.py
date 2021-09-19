"""
Embedding DRNN classifier with 128 input features and 2 output classes with keras and sklearn
"""
import os
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Add
from tensorflow.keras.metrics import AUC
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from sigopt import Connection as SigOpt
import logging

logging.basicConfig(level=logging.INFO)


# fix random seed for reproducibility
SEED = 7
np.random.seed(SEED)  # type: ignore


def prepare_data():
    logging.info("loading dataset")
    train = t.cast(pd.DataFrame, pd.read_csv("../input/train.csv.zip"))
    test = t.cast(pd.DataFrame, pd.read_csv("../input/test.csv.zip"))

    logging.info("feature augmentation")
    train["n_missing"] = train.isna().sum(axis=1)
    test["n_missing"] = test.isna().sum(axis=1)
    train["claim"] = train.loc[:, "claim"]

    features = [col for col in train.columns if col not in ["claim", "id"]]

    logging.info("split, split, split...")
    X_train, X_test, y_train, y_test = train_test_split(
        train[features], train["claim"], test_size=0.33, random_state=42
    )

    return (X_train, y_train, X_test, y_test), (test["id"], test[features])


def create_pipeline(input_shape, assignments):
    n_layers = assignments["n_layers"]
    n_hidden = assignments["n_hidden"]
    dropout = assignments["dropout"]
    activation = assignments["activation"]
    n_quantiles = assignments["n_quantiles"]
    quantile_output_distribution = assignments["quantile_output_distribution"]

    def baseline_model():
        input = Input((input_shape,))
        proj = Dropout(dropout)(
            BatchNormalization()(Dense(n_hidden, activation=activation)(input))
        )
        hidden = Dropout(dropout)(
            BatchNormalization()(Dense(n_hidden, activation=activation)(proj))
        )
        hidden = Add()([proj, hidden])
        for _ in range(n_layers):
            hidden = Dropout(dropout)(
                BatchNormalization()(Dense(n_hidden, activation=activation)(hidden))
            )
            hidden = Add()([proj, hidden])

        output = Dense(2, activation="softmax")(hidden)

        model = Model(input, output)

        model.compile(
            optimizer=Adam(learning_rate=0.1),
            loss="categorical_crossentropy",
            metrics=["accuracy", AUC(name="aucroc")],
        )

        return model

    logging.info(baseline_model().summary())

    estimator = KerasClassifier(
        build_fn=baseline_model,
        epochs=100,
        batch_size=512,
        callbacks=[
            ReduceLROnPlateau(monitor="aucroc", factor=0.2, patience=3, min_lr=0.0001),
            EarlyStopping(monitor="aucroc", patience=5, min_delta=0.0005),
            TerminateOnNaN(),
        ],
        verbose=1,
    )

    pipe = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant", missing_values=np.nan, fill_value=-999
                ),
            ),
            (
                "scaler",
                QuantileTransformer(
                    n_quantiles=n_quantiles,
                    output_distribution=quantile_output_distribution,
                ),
            ),
            ("model", estimator),
        ]
    )

    return pipe


def evaluate_model(X_train, y_train, assignments):
    logging.info("training model...")
    input_shape = t.cast(np.ndarray, X_train).shape[1]

    pipe = create_pipeline(input_shape, assignments)

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

    logging.info(
        "Baseline: %.2f%% %.2f%%)" % (results.mean() * 100, results.std() * 100)
    )

    return results.mean()


def test_pipeline(X_train, y_train, X_test, y_test, assignments):
    input_shape = t.cast(np.ndarray, X_train).shape[1]
    pipe = create_pipeline(input_shape, assignments)

    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)
    y_pred = [1 if i > 0.5 else 0 for i in y_prob[:, 1]]

    roc = roc_auc_score(y_test, y_pred)
    logging.info(f"roc: {roc}")

    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"accuracy: {accuracy}")

    precision = precision_score(y_test, y_pred)
    logging.info(f"precision: {precision}")

    recall = recall_score(y_test, y_pred)
    logging.info(f"recall: {recall}")

    f1 = f1_score(y_test, y_pred)
    logging.info(f"f1: {f1}")

    confusion = confusion_matrix(y_test, y_pred)
    logging.info(f"confusion: {confusion}")


def make_predictions(X_train, y_train, pred_index, X_pred, assignments):
    logging.info("Generating predictions...")

    input_shape = t.cast(np.ndarray, X_train).shape[1]
    pipe = create_pipeline(input_shape, assignments)

    pipe.fit(X_train, y_train)

    claim_prob = pipe.predict_proba(X_pred)
    claim = [1 if i > 0.5 else 0 for i in claim_prob[:, 1]]
    submission = pd.DataFrame({"id": pred_index, "claim": claim})
    submission.to_csv("./output/submission.csv", index=False)

    logging.info("Done!")


def main():
    sigopt = SigOpt(client_token=os.environ.get("SIGOPT_API_TOKEN"))

    experiment = sigopt.experiments().create(
        **{
            "name": "Deep Residual Neural Network",
            "project": "tabularplaygroundseries-sep2021",
            "observation_budget": 10,
            "metrics": [{"name": "cv_scores", "objective": "maximize"}],
            "parameters": [
                {"name": "n_layers", "type": "int", "bounds": {"min": 1, "max": 6}},
                {
                    "name": "n_hidden",
                    "type": "int",
                    "bounds": {"min": 8, "max": 256},
                },
                {"name": "dropout", "type": "double", "bounds": {"min": 0, "max": 0.6}},
                {
                    "name": "n_quantiles",
                    "type": "int",
                    "bounds": {"min": 32, "max": 1024},
                },
                {
                    "name": "activation",
                    "type": "categorical",
                    "categorical_values": [
                        {
                            "enum_index": 1,
                            "name": "relu",
                            "object": "categorical_value",
                        },
                        {
                            "enum_index": 2,
                            "name": "swish",
                            "object": "categorical_value",
                        },
                    ],
                },
                {
                    "name": "quantile_output_distribution",
                    "type": "categorical",
                    "categorical_values": [
                        {
                            "enum_index": 1,
                            "name": "uniform",
                            "object": "categorical_value",
                        },
                        {
                            "enum_index": 2,
                            "name": "normal",
                            "object": "categorical_value",
                        },
                    ],
                },
            ],
        }
    )

    train, test = prepare_data()
    X_train, y_train, X_test, y_test = train
    pred_index, X_pred = test

    for _ in range(experiment.observation_budget):  # type: ignore
        # Receive a suggestion from SigOpt
        suggestion = sigopt.experiments(experiment.id).suggestions().create()  # type: ignore
        assignments = suggestion.assignments  # type: ignore

        # Evaluate the model using suggestion
        try:
            value = evaluate_model(train[0], train[1], assignments)

            # Update the experiment
            sigopt.experiments(experiment.id).observations().create(  # type: ignore
                suggestion=suggestion.id, value=value  # type: ignore
            )
        except Exception as e:
            logging.exception(e)
            sigopt.experiments(experiment.id).observations().create(  # type: ignore
                failed=True, suggestion=suggestion.id  # type: ignore
            )

    # Get the best parameters
    assignments = (
        sigopt.experiments(experiment.id).best_assignments().fetch().data[0].assignments  # type: ignore
    )

    logging.info(assignments)

    # This is a SigOpt-tuned model
    test_pipeline(X_train, y_train, X_test, y_test, assignments)

    make_predictions(
        np.concatenate(X_train, X_test),
        np.concatenate(y_train, y_test),
        pred_index,
        X_pred,
        assignments,
    )

    logging.info("Explore your experiment: https://app.sigopt.com/experiment/" + experiment.id + "/analysis")  # type: ignore


if __name__ == "__main__":
    main()