# Workflow:
## Data Processing
- for the binary columns `sex` and `smoker`, we will convert them into numerical values (0 and 1) for easier analysis.
- for the categorical columns `region`, we will use one-hot encoding to create separate binary columns for each category.

## Exploratory Data Analysis (EDA)
- we use a *heatmap* to visualize the correlation between different numerical features in the dataset.
  - `smoker`, `age`, and `bmi` are the most correlated features with the target variable `charges`.
  - when smoking status is considered, the fatter and older individuals tend to have higher medical costs.

## Model Development
- we split the dataset into training and testing sets (80% training, 20% testing).
- we used seven different regression models with cross-validation to fine-tune hyperparameters and evaluate their performance.

## Model Recommendation
- based on R2_score metric, we recommend the *Decision Tree* model as it performs the best.
