from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    PolynomialFeatures,
)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)

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

X_train, X_test, y_train, y_test = train_test_split(
    features, target, train_size=TRAIN_SIZE, random_state=RANDOM_SEED
)

target_scaler = StandardScaler()
y_train = target_scaler.fit_transform(y_train.to_frame()).ravel()
y_test = target_scaler.transform(y_test.to_frame()).ravel()


# ==================== Preprocessors ====================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "binary",
            OrdinalEncoder(categories=[["female", "male"], ["no", "yes"]]),
            BINARY_COLS,
        ),
        (
            "onehot",
            OneHotEncoder(drop="first", dtype="int64", sparse_output=False),
            ONEHOT_COLS,
        ),
        ("scaler", StandardScaler(), SCALER_COLS),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)

preprocessor_single = ColumnTransformer(
    transformers=[
        ("binary", OrdinalEncoder(categories=[["no", "yes"]]), BEST_SINGLE_FEATURE)
    ],
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

fitted_models = {}


def grid_search(pipeline, param_grid, X_train, y_train):
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
    )


for name, config in configs.items():
    pipeline = config.pipeline

    # wandb = None
    run = None
    if wandb is not None:
        run = wandb.init(
            project="mini-project2-insurance",
            name=name,
            group="model-comparison",
            dir="./wandb_logs",
        )

    if config.param_grid:
        pipeline, best_params, best_score = grid_search(
            pipeline, config.param_grid, X_train, y_train
        )
        print(f"{name}: Best Params: {best_params}, Best CV R2 Score: {best_score}")
        if run is not None:
            run.config.update(best_params)
            run.log({"best_cv_r2": best_score})
    else:
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"{name}: R2={r2:.4f}, MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}")

    fitted_models[name] = pipeline

    if run is not None:
        run.log({"test_r2": r2, "test_mae": mae, "test_mse": mse, "test_rmse": rmse})
        run.finish()


# ==================== Questions ====================

# ---------- Polynomial Regression
# The larger degree of the polynomial regression model may lead to overfitting,
# as it can capture more complex relationships in the training data.
# This can also result in poor generalization to unseen data,
# leading to lower performance on the test set.

# ---------- Which features did Lasso eliminate?
lasso_pipeline: Pipeline = fitted_models["Lasso"]
lasso_model: Lasso = lasso_pipeline.named_steps["model"]
lasso_preprocessor: ColumnTransformer = lasso_pipeline.named_steps["preprocessor"]

lasso_feature_names = lasso_preprocessor.get_feature_names_out()
lasso_coefs = lasso_model.coef_

for feature_name, coef in zip(lasso_feature_names, lasso_coefs):
    print(f"{feature_name}: {coef:.4f}")

eliminated_features = lasso_feature_names[lasso_coefs == 0]
print("Features eliminated by Lasso:", list(eliminated_features))

# ---------- Visualize the Decision Tree
dt_pipeline: Pipeline = fitted_models["Decision Tree"]
dt_model: DecisionTreeRegressor = dt_pipeline.named_steps["model"]
dt_preprocessor: ColumnTransformer = dt_pipeline.named_steps["preprocessor"]
dt_feature_names = dt_preprocessor.get_feature_names_out()

plt.figure(figsize=(20, 10))
plot_tree(
    dt_model,
    feature_names=dt_feature_names.tolist(),
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.savefig("decision_tree.png", dpi=150, bbox_inches="tight")
