import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ==================== Data Preprocessing ====================

data = pd.read_csv(r"./insurance.csv")
# data.info()

# ---------- Binary Encoding
data["sex"] = data["sex"].map({"male": 1, "female": 0})
data["smoker"] = data["smoker"].map({"yes": 1, "no": 0})
# data.info()

# ---------- One-Hot Encoding
data = pd.get_dummies(data, columns=["region"], drop_first=True)
bool_columns = data.select_dtypes(["boolean"]).columns
data[bool_columns] = data[bool_columns].astype("int64")
# data.info()

# ---------- Save Cleaned Data
# data.to_csv(r"cleaned_data.csv", index=False)


# ==================== Requirement 1 ====================

# ----------- Correlation Matrix
correlated_matrix = data.corr()
# print(correlated_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlated_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title("Correlation Heatmap of Insurance Features", fontsize=14)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
# plt.show()

# ----------- Features and target relationship
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.boxplot(data, x="smoker", y="charges", hue="smoker", ax=axes[0, 0], legend=False)
axes[0, 0].set_title("Smoking Status vs Charges")
axes[0, 0].set_xticklabels(["Non-Smoker", "Smoker"])

sns.scatterplot(data, x="age", y="charges", hue="smoker", ax=axes[0, 1], alpha=0.6)
axes[0, 1].set_title("Age vs Charges")

sns.scatterplot(data, x="bmi", y="charges", hue="smoker", ax=axes[1, 0], alpha=0.6)
axes[1, 0].set_title("BMI vs Charges")

sns.histplot(data, x="charges", kde=True, bins=30, ax=axes[1, 1])
axes[1, 1].set_title("Distribution of Charges")

plt.tight_layout()
plt.savefig("features_vs_charges.png", dpi=150)
# plt.show()

# Insight: The biggest factor is smoking status, followed by age and BMI.
# When smoking status is considered, the fatter and older individuals tend to have higher medical costs.
