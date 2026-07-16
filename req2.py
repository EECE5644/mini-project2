from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, PolynomialFeatures

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    import wandb

    if not hasattr(wandb, "init"):
        wandb = None
except ImportError:
    wandb = None
    print("wandb is not installed. Skipping wandb logging.")


# ==================== Constants ====================

RANDOM_SEED = 818
TRAIN_SIZE = 0.8

BINARY_COLS = ["sex", "smoker"]
ONEHOT_COLS = ["region"]
SCALER_COLS = ["age", "bmi", "children"]
BEST_SINGLE_FEATURE = ["smoker"]


# ==================== Data Loading ====================

data = pd.read_csv(r"./insurance.csv")
# data.info()

features = data.drop(columns=["charges"])
target = data["charges"]

X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=TRAIN_SIZE, random_state=RANDOM_SEED)

target_scaler = StandardScaler()
y_train = target_scaler.fit_transform(y_train.to_frame()).ravel()
y_test = target_scaler.transform(y_test.to_frame()).ravel()


# ==================== Preprocessors ====================

preprocessor = ColumnTransformer(
    transformers=[
        ("binary", OrdinalEncoder(categories=[["female", "male"], ["no", "yes"]]), BINARY_COLS),
        ("onehot", OneHotEncoder(drop="first", dtype="int64", sparse_output=False), ONEHOT_COLS),
        ("scaler", StandardScaler(), SCALER_COLS),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)

preprocessor_single = ColumnTransformer(
    transformers=[("binary", OrdinalEncoder(categories=[["no", "yes"]]), BEST_SINGLE_FEATURE)],
    remainder="drop",
    verbose_feature_names_out=False,
)

# preprocessor.set_output(transform="pandas")
# res = preprocessor.fit_transform(features)
# res.info()
# print(res)


# ==================== Model Configs ====================
@dataclass
class ModelConfig:
    pipeline: Pipeline
    param_grid: dict[str, list] | None = None


configs = {
    "Simple Linear": ModelConfig(
        Pipeline(
            [
                ("preprocessor", preprocessor_single),
                ("model", LinearRegression()),
            ]
        )
    ),
    "Multiple Linear": ModelConfig(
        Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        )
    ),
    "Polynomial": ModelConfig(
        Pipeline(
            [
                ("preprocessor", preprocessor),
                ("poly", PolynomialFeatures(include_bias=False)),
                ("model", LinearRegression()),
            ]
        ),
        {"poly__degree": [2, 3, 4]},
    ),
    "Ridge": ModelConfig(
        Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", Ridge()),
            ]
        ),
        {"model__alpha": [0.01, 0.1, 1, 10, 100]},
    ),
    "Lasso": ModelConfig(
        Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", Lasso(max_iter=10000)),
            ]
        ),
        {"model__alpha": [0.01, 0.1, 1, 10, 100]},
    ),
    "SVR": ModelConfig(
        Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", SVR()),
            ]
        ),
        {
            "model__kernel": ["linear", "rbf"],
            "model__C": [0.1, 1, 10],
        },
    ),
    "Decision Tree": ModelConfig(
        Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", DecisionTreeRegressor(random_state=RANDOM_SEED)),
            ]
        ),
        {"model__max_depth": [2, 3, 4, 5, 6, None]},
    ),
}


# ==================== Training & Evaluation ====================
def grid_search(pipeline, param_grid, X_train, y_train):
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


for name, config in configs.items():
    pipeline = config.pipeline

    run = None
    if wandb is not None:
        run = wandb.init(project="mini-project2-insurance", name=name, group="model-comparison", dir="./wandb_logs")

    if config.param_grid:
        pipeline, best_params, best_score = grid_search(pipeline, config.param_grid, X_train, y_train)
        print(f"{name}: Best Params: {best_params}, Best CV R2 Score: {best_score}")
        if run is not None:
            run.config.update(best_params)
            run.log({"best_cv_r2": best_score})
    else:
        pipeline.fit(X_train, y_train)

    score = pipeline.score(X_test, y_test)
    print(f"{name}: R2 Score on test set: {score}")

    if run is not None:
        run.log({"test_r2": score})
        run.finish()
