"""
Embedding DRNN classifier with 128 input features and 2 output classes with keras and sklearn
"""
import os
import random
import typing as t
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Input,
    Dropout,
    BatchNormalization,
    Add,
    Embedding,
    Flatten,
)
from tensorflow.keras.metrics import AUC
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.backend import clear_session
from sigopt import Connection as SigOpt
import logging

logging.basicConfig(level=logging.INFO)

# fix random seed for reproducibility
SEED = 33
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)  # type: ignore

SIGOPT_API_TOKEN = os.environ.get("SIGOPT_API_TOKEN")

PIPELINE_NAME = "Embedded Deep Residual Neural Network"
PROJECT = "tabularplaygroundseries-sep2021"

TRAIN_INPUT_PATH = "../input/train.csv.zip"
TEST_INPUT_PATH = "../input/test.csv.zip"
OUTPUT_PATH = "./output/submission.csv"

N_JOBS = 1
N_CV_SPLITS = 5
OBSERVATION_BUDGET = 20
REDUCE_LR_PATIENCE = 2
REDUCE_LR_MIN_DELTA = 0.005
MAX_LR = 0.1
MIN_LR = 0.00001
MAX_TRAIN_EPOCHS = 25

METADATA = {
    "description": "https://www.kaggle.com/c/tabular-playground-series-sep-2021",
    "dataset": "https://www.kaggle.com/c/tabular-playground-series-sep-2021/data",
}

PARAMETERS = [
    {
        "name": "early_stop_patience",
        "type": "int",
        "bounds": {"min": 1, "max": 6},
    },
    {
        "name": "early_stop_min_delta",
        "type": "double",
        "bounds": {"min": 0.001, "max": 0.1},
    },
    {"name": "n_k_best_params", "type": "int", "bounds": {"min": 2, "max": 118}},
    {"name": "n_layers", "type": "int", "bounds": {"min": 0, "max": 8}},
    {
        "name": "n_hidden",
        "type": "int",
        "bounds": {"min": 16, "max": 128},
    },
    {"name": "dropout", "type": "double", "bounds": {"min": 0.01, "max": 0.5}},
    {
        "name": "n_quantiles",
        "type": "int",
        "bounds": {"min": 64, "max": 2048},
    },
    {
        "name": "n_bins",
        "type": "int",
        "bounds": {"min": 32, "max": 512},
    },
    {
        "name": "embedding_output_dim",
        "type": "int",
        "bounds": {"min": 4, "max": 64},
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
]


def prepare_data():
    logging.info("loading dataset")
    train = t.cast(pd.DataFrame, pd.read_csv(TRAIN_INPUT_PATH))
    test = t.cast(pd.DataFrame, pd.read_csv(TEST_INPUT_PATH))

    logging.info("feature augmentation")
    train["n_missing"] = train.isna().sum(axis=1)
    test["n_missing"] = test.isna().sum(axis=1)
    train["claim"] = train.loc[:, "claim"]

    features = [col for col in train.columns if col not in ["claim", "id"]]

    logging.info("split, split, split...")
    X_train, X_test, y_train, y_test = train_test_split(
        train[features], train["claim"], test_size=0.1, random_state=SEED
    )

    return (X_train, y_train, X_test, y_test), (test["id"], test[features])


def create_pipeline(assignments):
    clear_session()

    logging.info("Creating pipeline...")
    logging.info(assignments)

    n_layers = assignments["n_layers"]
    n_hidden = assignments["n_hidden"]
    dropout = assignments["dropout"]
    activation = assignments["activation"]
    n_quantiles = assignments["n_quantiles"]
    quantile_output_distribution = assignments["quantile_output_distribution"]
    n_bins = assignments["n_bins"]
    embedding_output_dim = assignments["embedding_output_dim"]
    early_stop_patience = assignments["early_stop_patience"]
    early_stop_min_delta = assignments["early_stop_min_delta"]
    n_k_best_params = assignments["n_k_best_params"]

    def baseline_model():
        def make_proj(input):
            proj = Embedding(
                input_dim=n_bins,
                output_dim=embedding_output_dim,
                embeddings_initializer="glorot_uniform",
            )(input)
            proj = Flatten()(proj)
            proj = Dense(n_hidden, activation=activation)(proj)
            proj = Dropout(dropout)(proj)
            return proj

        def make_hidden(input):
            hidden = Dense(n_hidden, activation=activation)(input)
            hidden = Dropout(dropout)(hidden)
            hidden = Add()([proj, hidden])
            hidden = BatchNormalization()(hidden)
            return hidden

        input = Input((n_k_best_params,))
        proj = make_proj(input)
        hidden = make_hidden(proj)
        for _ in range(n_layers):
            hidden = make_hidden(hidden)

        output = Dense(2, activation="softmax")(hidden)

        model = Model(input, output)

        model.compile(
            optimizer=Adam(learning_rate=MAX_LR),
            loss="categorical_crossentropy",
            metrics=[AUC(name="aucroc")],
        )

        return model

    baseline_model().summary()

    estimator = KerasClassifier(
        build_fn=baseline_model,
        epochs=MAX_TRAIN_EPOCHS,
        batch_size=1024,
        callbacks=[
            ReduceLROnPlateau(
                monitor="aucroc",
                factor=0.2,
                patience=REDUCE_LR_PATIENCE,
                min_lr=MIN_LR,
                min_delta=REDUCE_LR_MIN_DELTA,
                mode="max",
            ),
            EarlyStopping(
                monitor="aucroc",
                patience=early_stop_patience,
                min_delta=early_stop_min_delta,
                mode="max",
                restore_best_weights=True,
            ),
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
            (
                "feature_selection",
                SelectKBest(f_classif, k=n_k_best_params),
            ),
            (
                "binning",
                KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform"),
            ),
            ("model", estimator),
        ]
    )

    return pipe


def evaluate_model(X_train, y_train, assignments):
    pipe = create_pipeline(assignments)

    skfold = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=SEED)

    results = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=skfold,
        scoring="roc_auc",
        verbose=1,
        n_jobs=N_JOBS,
    )

    logging.info(
        "Baseline: %.2f%% %.2f%%)" % (results.mean() * 100, results.std() * 100)
    )

    return results.mean()


def test_pipeline(X_train, y_train, X_test, y_test, assignments):
    pipe = create_pipeline(assignments)

    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)
    y_pred = [1 if i > 0.5 else 0 for i in y_prob[:, 1]]

    results = {
        "roc": roc_auc_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    confusion = confusion_matrix(y_test, y_pred)
    logging.info(f"confusion: {confusion}")

    report = classification_report(y_test, y_pred)
    logging.info(report)

    return results


def make_predictions(X_train, y_train, pred_index, X_pred, assignments):
    pipe = create_pipeline(assignments)

    pipe.fit(X_train, y_train)

    claim_prob = pipe.predict_proba(X_pred)
    claim = [1 if i > 0.5 else 0 for i in claim_prob[:, 1]]
    submission = pd.DataFrame({"id": pred_index, "claim": claim})
    submission.to_csv(OUTPUT_PATH, index=False)

    logging.info("Done!")


def main():
    sigopt = SigOpt(client_token=SIGOPT_API_TOKEN)

    logging.info("Creating experiment")

    experiment = sigopt.experiments().create(
        **{
            "name": PIPELINE_NAME,
            "project": PROJECT,
            "observation_budget": OBSERVATION_BUDGET,
            "metrics": [{"name": "cv_scores", "objective": "maximize"}],
            "parameters": PARAMETERS,
            "metadata": METADATA,
        }
    )

    logging.info("Preparing dataset")

    train, test = prepare_data()
    X_train, y_train, X_test, y_test = train
    pred_index, X_pred = test

    logging.info("Optimizing hyper-parameters")

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

    logging.info("Testing best model")

    # Get the best parameters
    assignments = (
        sigopt.experiments(experiment.id).best_assignments().fetch().data[0].assignments  # type: ignore
    )

    logging.info(assignments)

    # This is a SigOpt-tuned model
    test_results = test_pipeline(X_train, y_train, X_test, y_test, assignments)

    sigopt.experiments(experiment.id).update(metadata={**METADATA, **test_results})  # type: ignore

    logging.info("Producing predictions for target dataset")

    make_predictions(
        pd.concat([X_train, X_test]),  # type: ignore
        pd.concat([y_train, y_test]),  # type: ignore
        pred_index,
        X_pred,
        assignments,
    )

    logging.info("Done! Explore your experiment: https://app.sigopt.com/experiment/" + experiment.id + "/analysis")  # type: ignore


if __name__ == "__main__":
    main()
