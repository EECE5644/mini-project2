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
data.to_csv(r"cleaned_data.csv", index=False)

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
