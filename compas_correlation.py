import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
from HSIC import hsic_gam  # Ensure HSIC.py is in your working directory

# Set random seed
random_state = 2025

# --- Load and preprocess data ---
dataset = pd.read_csv('./data/compas/recid.data', delimiter=' ', header=None)
dataset.columns = ["age", "race", "sex", "priors_count", "length_of_stay", "c_charge_degree", "two_year_recid"]

# Feature groups
binary = ['sex', 'c_charge_degree']
other = ['age', 'race', 'priors_count', 'length_of_stay']

# Scale continuous features to [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset[other] = scaler.fit_transform(dataset[other])

# Convert binary categorical values to 0/1
dataset[binary] = dataset[binary] - 1

# Train/test split
data_train, data_test = train_test_split(dataset, train_size=0.8, test_size=0.2, random_state=random_state)
X_train, y_train = data_train[other + binary], data_train['two_year_recid']

# List of feature names
feature_names = X_train.columns.tolist()

# --- Compute Pearson Correlation ---
print("=== Pearson Correlation ===")
pearson_scores = []
for col in feature_names:
    corr, _ = pearsonr(X_train[col], y_train)
    pearson_scores.append(corr)
    print(f"{col}: {corr:.4f}")

# --- Compute Mutual Information ---
print("\n=== Mutual Information ===")
mi_scores = mutual_info_classif(
    X_train, 
    y_train, 
    discrete_features=[feature_names.index(col) for col in binary], 
    random_state=random_state
)
for col, score in zip(feature_names, mi_scores):
    print(f"{col}: {score:.4f}")

# --- Compute HSIC ---
print("\n=== HSIC (Hilbert-Schmidt Independence Criterion) ===")
hsic_scores = []
for col in feature_names:
    X_col = X_train[col].values.reshape(-1, 1)
    y_vals = y_train.values.reshape(-1, 1)
    testStat, _ = hsic_gam(X_col, y_vals, alph=0.05)
    hsic_scores.append(testStat)
    print(f"{col}: {testStat:.6f}")

# --- Normalize Scores to [-1, 1] ---
def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return 2 * (arr - min_val) / (max_val - min_val) - 1

normalized_mi = min_max_normalize(mi_scores)
normalized_hsic = min_max_normalize(hsic_scores)

# --- Display Normalized Scores ---
print("\n=== Normalized Scores (Range: [-1, 1]) ===")
print(f"{'Feature':<20} {'Pearson':>10} {'MI':>10} {'HSIC':>10}")
for i, col in enumerate(feature_names):
    print(f"{col:<20} {pearson_scores[i]:>10.4f} {normalized_mi[i]:>10.4f} {normalized_hsic[i]:>10.4f}")



# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.feature_selection import mutual_info_regression
# from scipy.stats import pearsonr
# from HSIC import hsic_gam  # ensure HSIC.py is in the working directory

# # --- Load and preprocess COMPAS data ---
# dataset = pd.read_csv('./data/compas/recid.data', delimiter=' ', header=None)
# dataset.columns = ["age", "race", "sex", "priors_count", "length_of_stay", "c_charge_degree", "two_year_recid"]

# binary = ['sex', 'c_charge_degree']
# other = ['age', 'race', 'priors_count', 'length_of_stay']
# features = other + binary

# scaler = MinMaxScaler(feature_range=(-1, 1))
# dataset[other] = scaler.fit_transform(dataset[other])
# dataset[binary] = dataset[binary] - 1  # make binary {0, 1}

# data_train, _ = train_test_split(dataset, train_size=0.8, random_state=2025)
# X_train = data_train[features]

# # --- Pearson Correlation ---
# print("=== Pearson Correlation Between Features ===")
# corr_matrix = X_train.corr(method='pearson').round(4)
# print(corr_matrix)

# pearson_pairs = []
# for i in range(len(features)):
#     for j in range(i + 1, len(features)):
#         f1, f2 = features[i], features[j]
#         corr = corr_matrix.loc[f1, f2]
#         pearson_pairs.append((f1, f2, corr))

# # --- Mutual Information (MI) ---
# print("\n=== Mutual Information Between Features ===")
# mi_pairs = []
# for i in range(len(features)):
#     for j in range(i + 1, len(features)):
#         f1, f2 = features[i], features[j]
#         X_feat = X_train[f1].values.reshape(-1, 1)
#         Y_feat = X_train[f2].values
#         mi_score = mutual_info_regression(X_feat, Y_feat, random_state=2025)[0]
#         mi_pairs.append((f1, f2, mi_score))
#         print(f"{f1} ↔ {f2}: {mi_score:.4f}")

# # --- HSIC ---
# print("\n=== HSIC Between Features ===")
# hsic_pairs = []
# for i in range(len(features)):
#     for j in range(i + 1, len(features)):
#         f1, f2 = features[i], features[j]
#         x = X_train[f1].values.reshape(-1, 1)
#         y = X_train[f2].values.reshape(-1, 1)
#         score, _ = hsic_gam(x, y, alph=0.05)
#         hsic_pairs.append((f1, f2, score))
#         print(f"{f1} ↔ {f2}: {score:.4f}")


# --------------------


# # --- LaTeX Table Formatter ---
# def print_latex_table(pairs, caption, value_label):
#     print("\n\\begin{table}[h]")
#     print("\\centering")
#     print(f"\\caption{{{caption}}}")
#     print("\\begin{tabular}{l l r}")
#     print("\\toprule")
#     print("Feature 1 & Feature 2 & " + value_label + " \\\\")
#     print("\\midrule")
#     for f1, f2, val in sorted(pairs, key=lambda x: -abs(x[2])):
#         print(f"{f1} & {f2} & {val:.4f} \\\\")
#     print("\\bottomrule")
#     print("\\end{tabular}")
#     print("\\end{table}")

# # --- Output LaTeX Tables ---
# print_latex_table(pearson_pairs, "Pearson Correlation Between Features", "Correlation")
# print_latex_table(mi_pairs, "Mutual Information Between Features", "MI Score")
# print_latex_table(hsic_pairs, "HSIC Scores Between Features", "HSIC Score")

# import matplotlib.pyplot as plt
# import seaborn as sns

# # Create square matrices from flat pair lists
# def build_matrix(pairs, features):
#     mat = pd.DataFrame(np.zeros((len(features), len(features))), index=features, columns=features)
#     for f1, f2, score in pairs:
#         mat.loc[f1, f2] = score
#         mat.loc[f2, f1] = score  # symmetric
#     np.fill_diagonal(mat.values, 1.0 if mat.values.max() <= 1 else 0.0)
#     return mat

# pearson_mat = build_matrix(pearson_pairs, features)
# mi_mat = build_matrix(mi_pairs, features)
# hsic_mat = build_matrix(hsic_pairs, features)

# # Plot heatmaps

# def plot_heatmap(mat, title, filename, center=None):
#     # Create a masked copy without the diagonal for color scaling
#     mat_for_vmin_vmax = mat.copy()
#     np.fill_diagonal(mat_for_vmin_vmax.values, np.nan)  # Exclude diagonal from scale

#     vmin = np.nanmin(mat_for_vmin_vmax.values)
#     vmax = np.nanmax(mat_for_vmin_vmax.values)

#     plt.figure(figsize=(8, 6))
#     sns.heatmap(
#         mat, annot=True, fmt=".2f", cmap="coolwarm",
#         center=center, vmin=vmin, vmax=vmax
#     )
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(f'./figures/compas_{filename}.png')
#     plt.show()


# def plot_heatmap(mat, title, filename, center=None, log_scale=False, mask_diagonal=False):
#     # Copy matrix and optionally apply log scale
#     mat_plot = mat.copy()
#     if log_scale:
#         # Shift to avoid log(0)
#         mat_plot = np.log10(mat_plot + 1e-8)

#     # Create mask for diagonal
#     mask = None
#     if mask_diagonal:
#         mask = np.eye(len(mat_plot), dtype=bool)

#     # Compute color limits ignoring diagonal
#     mat_for_vmin_vmax = mat_plot.copy()
#     np.fill_diagonal(mat_for_vmin_vmax.values, np.nan)
#     vmin = np.nanmin(mat_for_vmin_vmax.values)
#     vmax = np.nanmax(mat_for_vmin_vmax.values)

#     plt.figure(figsize=(8, 6))
#     sns.heatmap(
#         mat_plot,
#         annot=True,
#         fmt=".2f",
#         cmap="coolwarm",
#         center=center,
#         vmin=vmin,
#         vmax=vmax,
#         mask=mask,
#         cbar_kws={'label': 'log(HSIC)' if log_scale else ''}
#     )
    
#     # Draw black boxes over diagonal cells
#     if mask_diagonal:
#         for i in range(len(mat_plot)):
#             plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='black', lw=0))

#     plt.title(title)
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.savefig(f'./figures/compas_{filename}.png')
#     plt.show()


# # plot_heatmap(pearson_mat, "Feature–Feature Pearson Correlation", 'corr', center=0)
# # plot_heatmap(mi_mat, "Feature–Feature Mutual Information", 'mi', center=None)
# # plot_heatmap(hsic_mat, "Feature–Feature HSIC", 'hsic', center=None)


# plot_heatmap(pearson_mat, "Feature–Feature Pearson Correlation", 'corr', center=0, mask_diagonal=True)
# plot_heatmap(mi_mat, "Feature–Feature Mutual Information", 'mi', center=None, mask_diagonal=True)
# plot_heatmap(hsic_mat, "Feature–Feature HSIC (log scale)", 'hsic', center=None, log_scale=True, mask_diagonal=True)
