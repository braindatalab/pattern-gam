import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from nam.wrapper import NAMClassifier
import torch
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def compute_feature_pattern(model, X_train):
    """
    Compute feature-space activation pattern, including main and interaction effects.
    Args:
        model: trained NAMClassifier model
        X_train: original (unscaled) training data
    Returns:
        pattern: array of shape (num_features + num_interactions,)
    """
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()

    Z_all = []

    for learner in model.models:
        learner.eval()
        with torch.no_grad():
            _, _, main_effects_out, interaction_effects_out = learner.forward(torch.tensor(X_train, dtype=torch.float32))
            if interaction_effects_out.shape[1] > 0:
                Z = torch.cat([main_effects_out, interaction_effects_out], dim=1)  # (n_samples, total_units)
            else:
                Z = main_effects_out  # (n_samples, num_features)
            Z_all.append(Z.cpu().numpy())

    Z_all = np.stack(Z_all, axis=0)  # (num_learners, n_samples, total_units)
    Z_mean = np.mean(Z_all, axis=0)  # (n_samples, total_units)

    cov_Z = np.cov(Z_mean, rowvar=False)  # (total_units, total_units)

    linreg = LogisticRegression(penalty=None, fit_intercept=False, max_iter=1000, random_state=2025)
    linreg.fit(Z_mean, y_train)

    w_Z = linreg.coef_[0]  # (total_units,)

    pattern = cov_Z @ w_Z  # (total_units,)

    return pattern

# # Categorical mappings
# categorical_features = {
#     "sex": ["Female", "Male"],
#     "c_charge_degree": ["Misdemeanor", "Felony"],
#     "race": ["African American", "Asian", "Caucasian", "Hispanic", "Native American", "Other"]
# }

# # Define manual x-axis limits to match the ranges seen in Agarwal et al.'s NAM paper
# plot_xlims = {
#     "age": (18, 45),
#     "priors_count": (0, 5),
#     "length_of_stay": (0, 10),
# }

# feature_names = ["age", "race", "sex", "priors_count", "length_of_stay", "c_charge_degree"]

# def shade_by_density_blocks_nam(X_orig, feature_index, ax, is_categorical, n_blocks=5, color=[0.9, 0.5, 0.5], min_y=None, max_y=None):
#     if isinstance(X_orig, pd.DataFrame):
#         X_orig = X_orig.to_numpy()
#     feature_data = X_orig[:, feature_index]

#     if min_y is None or max_y is None:
#         min_y, max_y = ax.get_ylim()

#     # Detect "integer-like" features (e.g., priors_count)
#     is_integer_discrete = np.allclose(feature_data, np.round(feature_data))

#     if not is_categorical and not is_integer_discrete:
#         # Continuous feature
#         if feature_names and feature_names[feature_index] in plot_xlims:
#             min_x, max_x = plot_xlims[feature_names[feature_index]]
#         else:
#             min_x, max_x = np.min(feature_data), np.max(feature_data)

#         segments = np.linspace(min_x, max_x, n_blocks + 1)
#         density, _ = np.histogram(feature_data, bins=segments)
#         normed_density = density / np.max(density) if np.max(density) != 0 else density

#         for i in range(n_blocks):
#             start_x = segments[i]
#             end_x = segments[i + 1]
#             alpha = min(1.0, 0.01 + normed_density[i])

#             rect = patches.Rectangle(
#                 (start_x, min_y - 1),
#                 end_x - start_x,
#                 max_y - min_y + 1,
#                 linewidth=0,
#                 edgecolor=color,
#                 facecolor=color,
#                 alpha=alpha
#             )
#             ax.add_patch(rect)

#     else:
#         # Categorical or integer-discrete feature
#         categories = np.unique(feature_data)
#         counts = np.array([(feature_data == cat).sum() for cat in categories])
#         normed_counts = counts / np.max(counts) if np.max(counts) != 0 else counts

#         for i, cat in enumerate(categories):
#             width = 1.0
#             alpha = min(1.0, 0.01 + normed_counts[i])

#             rect = patches.Rectangle(
#                 (cat - 0.5, min_y - 1),
#                 width,
#                 max_y - min_y + 1,
#                 linewidth=0,
#                 edgecolor=color,
#                 facecolor=color,
#                 alpha=alpha
#             )
#             ax.add_patch(rect)




# def plot_nam_ensemble_feature_shapes(
#     model, X_orig, feature_names=None, color='black', n_blocks=5,
#     figsize=(20, 12), sharey=False, pattern=None, out_path=''
# ):
#     """Plot NAM ensemble feature shapes (main effects) with per-learner curves and density shading."""
#     if isinstance(X_orig, pd.DataFrame):
#         X_orig = X_orig.to_numpy()

#     X_real_min, X_real_max = np.min(X_orig, axis=0), np.max(X_orig, axis=0)
#     num_features, num_learners = X_orig.shape[1], len(model.models)

#     fig, axs = plt.subplots(
#         nrows=(num_features + 2) // 3, ncols=3,
#         figsize=figsize, squeeze=False, sharey=sharey
#     )
#     axs = axs.flatten()

#     global_min, global_max = np.inf, -np.inf
#     feature_outputs = {}

#     for feature_idx in range(num_features):
#         feature_name = feature_names[feature_idx] if feature_names else f"Feature {feature_idx}"
#         is_categorical = feature_name in categorical_features

#         if is_categorical:
#             x_vals_real = np.unique(X_orig[:, feature_idx])
#             X_query = np.zeros((len(x_vals_real), num_features))
#             X_query[:, feature_idx] = x_vals_real
#         else:
#             x_vals_norm = np.linspace(-1, 1, 500)
#             x_vals_real = ((x_vals_norm + 1) / 2) * (X_real_max[feature_idx] - X_real_min[feature_idx]) + X_real_min[feature_idx]
#             X_query = np.zeros((len(x_vals_norm), num_features))
#             X_query[:, feature_idx] = x_vals_norm

#         preds_all = []
#         for learner in model.models:
#             learner.eval()
#             with torch.no_grad():
#                 _, _, main_effects_out, _ = learner.forward(torch.tensor(X_query, dtype=torch.float32))
#                 preds = main_effects_out[:, feature_idx].cpu().numpy()
#                 preds_all.append(preds)

#         preds_all = np.stack(preds_all, axis=0)
#         mean_preds = np.mean(preds_all, axis=0)

#         if pattern is not None:
#             mean_preds *= np.abs(pattern[feature_idx])
#             preds_all *= np.abs(pattern[feature_idx])

#         feature_outputs[feature_idx] = (x_vals_real, mean_preds, is_categorical, preds_all)

#         global_min = min(global_min, np.min(mean_preds))
#         global_max = max(global_max, np.max(mean_preds))

#     for feature_idx in range(num_features):
#         ax = axs[feature_idx]
#         x_vals_real, mean_preds, is_categorical, preds_all = feature_outputs[feature_idx]

#         if is_categorical:
#             x_ext = np.concatenate([[x_vals_real[0] - 0.5], x_vals_real + 0.5])
#             # plot each learner
#             for preds in preds_all:
#                 preds_ext = np.concatenate([[preds[0]], preds])
#                 ax.step(x_ext, preds_ext, color=color, alpha=0.05, linewidth=1, where='post')
#             # plot mean
#             mean_preds_ext = np.concatenate([[mean_preds[0]], mean_preds])
#             ax.step(x_ext, mean_preds_ext, color=color, alpha=1.0, linewidth=2, where='post')

#             shade_by_density_blocks_nam(X_orig, feature_idx, ax, is_categorical=True, n_blocks=n_blocks, min_y=global_min, max_y=global_max)

#             labels = categorical_features.get(feature_names[feature_idx], None)
#             if labels is not None and len(labels) == len(x_vals_real):
#                 ax.set_xticks(x_vals_real)
#                 ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
#             else:
#                 ax.set_xticks(x_vals_real)
#                 ax.set_xticklabels([str(x) for x in x_vals_real], rotation=30, ha='right', fontsize=10)

#             if feature_names and feature_names[feature_idx] in plot_xlims:
#                 xmin, xmax = plot_xlims[feature_names[feature_idx]]
#                 ax.set_xlim(xmin, xmax)
#             else:
#                 # default behavior for unknown features
#                 ax.set_xlim(np.min(x_vals_real) - 0.5, np.max(x_vals_real) + 0.5)

#         else:
#             # plot each learner
#             for preds in preds_all:
#                 ax.plot(x_vals_real, preds, color=color, alpha=0.05, linewidth=1)
#             # plot mean
#             ax.plot(x_vals_real, mean_preds, color=color, alpha=1.0, linewidth=2)

#             shade_by_density_blocks_nam(X_orig, feature_idx, ax, is_categorical=False, n_blocks=n_blocks, min_y=global_min, max_y=global_max)

#             if feature_names and feature_names[feature_idx] in plot_xlims:
#                 xmin, xmax = plot_xlims[feature_names[feature_idx]]
#                 ax.set_xlim(xmin, xmax)
#             else:
#                 # default behavior for unknown features
#                 ax.set_xlim(np.min(x_vals_real), np.max(x_vals_real))

#         ax.set_title(feature_names[feature_idx], fontsize=12) if feature_names else ax.set_title(f"Feature {feature_idx}")
#         ax.tick_params(labelsize=10)

#     if sharey:
#         for ax in axs:
#             ax.set_ylim(global_min, global_max)

#     # for i in range(num_features, len(axs)):
#     #     axs[i].axis('off')

#     plt.tight_layout()

#     if out_path != '':
#         plt.savefig(f'{out_path}.png', bbox_inches='tight')
#         plt.savefig(f'{out_path}_hires.png', dpi=300, bbox_inches='tight')
#     else:
#         pat_str = ''
#         if pattern != None:
#             pat_str = '_pattern_scaled'

#         plt.savefig(f'./figures/recid{pat_str}.png', bbox_inches='tight')
#         plt.savefig(f'./figures/recid{pat_str}_hires.png', dpi=300, bbox_inches='tight')

#     plt.show()


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as sk_metrics

from interpret.utils import measure_interactions

random_state = 2025

# categorical_label_maps = {
#     'race': ['African\nAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Native\nAmerican', 'Other'],
#     'sex': ['Female', 'Male'],
#     'c_charge_degree': ['Misdemeanor', 'Felony']
# }


# # --- Load and preprocess data ---
# dataset = pd.read_csv('./data/compas/recid.data', delimiter=' ', header=None)
# dataset.columns = ["age", "race", "sex", "priors_count", "length_of_stay", "c_charge_degree", "two_year_recid"]

# binary = ['sex', 'c_charge_degree']
# other = ['age', 'race', 'priors_count', 'length_of_stay']

# scaler = MinMaxScaler((-1, 1))
# dataset[other] = scaler.fit_transform(dataset[other])
# dataset[binary] = dataset[binary] - 1

# data_train, data_test = train_test_split(dataset, train_size=0.8, test_size=0.2, random_state=random_state)
# X_train, y_train = data_train[other + binary], data_train['two_year_recid']
# X_test, y_test = data_test[other + binary], data_test['two_year_recid']

# feature_names = X_train.columns.tolist()
# d = X_train.shape[1]
# # Rebuild real-world X for plotting
# X_train_continuous_real = scaler.inverse_transform(X_train[other])
# X_train_binary_real = X_train[binary].to_numpy()
# X_train_real = np.concatenate([X_train_continuous_real, X_train_binary_real], axis=1)


# --- Helper for finding interaction pairs ---
def FAST(X_train, y_train, n_interactions, init_score=None, feature_names=None, feature_types=None):
    top_interactions = measure_interactions(
        X_train, y_train,
        interactions=n_interactions,
        init_score=init_score,
        feature_names=feature_names,
        feature_types=feature_types
    )
    return [(i, j) for (i, j), _ in top_interactions]

# --- Directory for figures ---
OUT_DIR = './figures'
os.makedirs(OUT_DIR, exist_ok=True)




# df = pd.read_csv('./data/nls/nls_3i0s7u_0_0.2.csv', delimiter=',')
df = pd.read_csv('./data/nls/nls_3i0s7u_9_0.5.csv', delimiter=',')
feature_columns = [col for col in df.columns if col != 'label']
X = df[feature_columns].values
y = df['label'].values

# Train/val/test split
num_samples = X.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)

# First split: Train+Val and Test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2025, stratify=y
)

# Then split: Train and Val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=2025, stratify=y_temp
)

d = X_train.shape[1]

# # --- Train and plot for different n_interactions ---
n_interactions_list = [0,32,'full']
num_learners = 1


for n_interactions in n_interactions_list:

    if n_interactions == 0:
        interaction_pairs = []
    elif n_interactions == 'full':
        interaction_pairs = [(i, j) for i in range(d) for j in range(i+1, d) if i < j]
    else:
        interaction_pairs = FAST(X_train, y_train, n_interactions=n_interactions)

    model = NAMClassifier(
        num_epochs=100,
        num_learners=num_learners,
        metric='auroc',
        interaction_pairs=interaction_pairs,
        early_stop_mode='max',
        monitor_loss=False,
        n_jobs=10,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    pred = model.predict_proba(X_test)
    auroc = sk_metrics.roc_auc_score(y_test, pred)
    print(f"n_interactions={n_interactions}, AUROC={auroc:.3f}")

    

    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    # Assuming X_train and model are already available from your training loop

    feature_names = [f'x_{i}' for i in range(d)]

    # --- Precompute global min/max for scaling plots
    global_preds_raw = []
    global_preds_pattern = []

    pattern = compute_feature_pattern(model, X_train)  # Assuming you want pattern scaling too

    for feature_idx in range(d):
        x_vals_real = np.unique(X_train[:, feature_idx])

        X_query = np.zeros((len(x_vals_real), d))
        X_query[:, feature_idx] = 2 * (x_vals_real - np.min(X_train[:, feature_idx])) / (np.max(X_train[:, feature_idx]) - np.min(X_train[:, feature_idx])) - 1

        preds_all = []
        for learner in model.models:
            learner.eval()
            with torch.no_grad():
                _, _, main_effects_out, _ = learner.forward(torch.tensor(X_query, dtype=torch.float32))
                preds = main_effects_out[:, feature_idx].cpu().numpy().flatten()
                preds_all.append(preds)

        preds_all = np.stack(preds_all, axis=0)
        mean_preds = np.mean(preds_all, axis=0)

        global_preds_raw.append(mean_preds)
        global_preds_pattern.append(mean_preds * np.abs(pattern[feature_idx]))

    global_min_raw = np.min([np.min(m) for m in global_preds_raw])
    global_max_raw = np.max([np.max(m) for m in global_preds_raw])

    global_min_pattern = np.min([np.min(m) for m in global_preds_pattern])
    global_max_pattern = np.max([np.max(m) for m in global_preds_pattern])

    # --- Plotting ---
    fig, axs = plt.subplots(2, d, figsize=(4*d, 8), sharey='row')

    for row_idx, scale_pattern in enumerate([False, True]):
        for feature_idx in range(d):
            ax = axs[row_idx, feature_idx]
            x_vals_real = np.unique(X_train[:, feature_idx])

            X_query = np.zeros((len(x_vals_real), d))
            X_query[:, feature_idx] = 2 * (x_vals_real - np.min(X_train[:, feature_idx])) / (np.max(X_train[:, feature_idx]) - np.min(X_train[:, feature_idx])) - 1

            preds_all = []
            for learner in model.models:
                learner.eval()
                with torch.no_grad():
                    _, _, main_effects_out, _ = learner.forward(torch.tensor(X_query, dtype=torch.float32))
                    preds = main_effects_out[:, feature_idx].cpu().numpy().flatten()
                    preds_all.append(preds)

            preds_all = np.stack(preds_all, axis=0)
            mean_preds = np.mean(preds_all, axis=0)

            if scale_pattern:
                mean_preds *= np.abs(pattern[feature_idx])
                preds_all *= np.abs(pattern[feature_idx])

            for preds in preds_all:
                ax.plot(x_vals_real, preds, color='blue', alpha=0.05, linewidth=1)

            ax.plot(x_vals_real, mean_preds, color='blue', alpha=1.0, linewidth=2)

            ax.set_xlim(np.min(x_vals_real), np.max(x_vals_real))

            if row_idx == 0:
                ax.set_ylim(global_min_raw, global_max_raw)
                ax.set_title(feature_names[feature_idx], fontsize=12)
            else:
                ax.set_ylim(global_min_pattern, global_max_pattern)

            if feature_idx == 0:
                ylabel = 'NAM Prediction' if row_idx == 0 else 'Pattern Scaled Prediction'
                ax.set_ylabel(ylabel, fontsize=14)

            ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.show()


    # pattern = compute_feature_pattern(model, X_train)

    # # ax = axs[row_idx, feature_idx]

    # # --- PLOT BOTH normal and pattern-scaled together on one figure ---
    # fig, axs = plt.subplots(2, len(feature_names), figsize=(24, 8), sharey='row')

    # # --- First: precompute global min/max across all features
    # # --- Precompute global min/max separately for NAM and PatternNAM

    # global_preds_raw = []
    # global_preds_pattern = []

    # for feature_idx in range(len(feature_names)):
    #     feature_name = feature_names[feature_idx]
    #     is_categorical = feature_name in categorical_features

    #     if is_categorical:
    #         x_vals_real = np.unique(X_train_real[:, feature_idx])
    #     else:
    #         x_vals_real = np.unique(X_train_real[:, feature_idx])

    #     X_query = np.zeros((len(x_vals_real), X_train.shape[1]))
    #     if is_categorical:
    #         X_query[:, feature_idx] = x_vals_real
    #     else:
    #         X_query[:, feature_idx] = 2 * (x_vals_real - np.min(X_train_real[:, feature_idx])) / (np.max(X_train_real[:, feature_idx]) - np.min(X_train_real[:, feature_idx])) - 1

    #     preds_all = []
    #     for learner in model.models:
    #         learner.eval()
    #         with torch.no_grad():
    #             _, _, main_effects_out, _ = learner.forward(torch.tensor(X_query, dtype=torch.float32))
    #             preds = main_effects_out[:, feature_idx].cpu().numpy().flatten()
    #             preds_all.append(preds)

    #     preds_all = np.stack(preds_all, axis=0)
    #     mean_preds = np.mean(preds_all, axis=0)

    #     global_preds_raw.append(mean_preds)
    #     global_preds_pattern.append(mean_preds * np.abs(pattern[feature_idx]))

    # # Compute separate global min/max
    # global_min_raw = np.min([np.min(m) for m in global_preds_raw])
    # global_max_raw = np.max([np.max(m) for m in global_preds_raw])

    # global_min_pattern = np.min([np.min(m) for m in global_preds_pattern])
    # global_max_pattern = np.max([np.max(m) for m in global_preds_pattern])


    # # --- Now actual plotting loop ---
    # for row_idx, scale_pattern in enumerate([False, True]):
    #     for feature_idx, feature_name in enumerate(feature_names):
    #         ax = axs[row_idx, feature_idx]
    #         is_categorical = feature_name in categorical_features

    #         if is_categorical:
    #             x_vals_real = np.unique(X_train_real[:, feature_idx])
    #             X_query = np.zeros((len(x_vals_real), X_train.shape[1]))
    #             X_query[:, feature_idx] = x_vals_real
    #         else:
    #             x_vals_real = np.unique(X_train_real[:, feature_idx])
    #             X_query = np.zeros((len(x_vals_real), X_train.shape[1]))
    #             X_query[:, feature_idx] = 2 * (x_vals_real - np.min(X_train_real[:, feature_idx])) / (np.max(X_train_real[:, feature_idx]) - np.min(X_train_real[:, feature_idx])) - 1

    #         preds_all = []
    #         for learner in model.models:
    #             learner.eval()
    #             with torch.no_grad():
    #                 _, _, main_effects_out, _ = learner.forward(torch.tensor(X_query, dtype=torch.float32))
    #                 preds = main_effects_out[:, feature_idx].cpu().numpy().flatten()
    #                 preds_all.append(preds)

    #         preds_all = np.stack(preds_all, axis=0)
    #         mean_preds = np.mean(preds_all, axis=0)

    #         if scale_pattern:
    #             mean_preds *= np.abs(pattern[feature_idx])
    #             preds_all *= np.abs(pattern[feature_idx])

    #         # --- Shade density BEFORE plotting lines ---

    #         if row_idx == 0:
    #             shade_by_density_blocks_nam(
    #                 X_train_real, feature_idx, ax,
    #                 is_categorical=is_categorical,
    #                 n_blocks=10,
    #                 min_y=global_min_raw,
    #                 max_y=global_max_raw
    #             )
    #         else:
    #             shade_by_density_blocks_nam(
    #                 X_train_real, feature_idx, ax,
    #                 is_categorical=is_categorical,
    #                 n_blocks=10,
    #                 min_y=global_min_pattern,
    #                 max_y=global_max_pattern
    #             )
            
    #         if is_categorical:
    #             centers = x_vals_real  # e.g., [0, 1] for sex

    #             if len(centers) == 1:
    #                 # single class, weird corner case
    #                 left_edges = centers - 0.5
    #                 right_edges = centers + 0.5
    #             else:
    #                 midpoints = (centers[:-1] + centers[1:]) / 2
    #                 leftmost = centers[0] - (midpoints[0] - centers[0])
    #                 rightmost = centers[-1] + (centers[-1] - midpoints[-1])
    #                 edges = np.concatenate(([leftmost], midpoints, [rightmost]))

    #             for preds in preds_all:
    #                 x_steps = np.empty(2 * len(centers))
    #                 y_steps = np.empty(2 * len(centers))

    #                 x_steps[0::2] = edges[:-1]
    #                 x_steps[1::2] = edges[1:]
    #                 y_steps[0::2] = preds
    #                 y_steps[1::2] = preds

    #                 ax.plot(x_steps, y_steps, color='blue', alpha=0.05, linewidth=1)

    #             # Mean
    #             x_steps = np.empty(2 * len(centers))
    #             y_steps = np.empty(2 * len(centers))
    #             x_steps[0::2] = edges[:-1]
    #             x_steps[1::2] = edges[1:]
    #             y_steps[0::2] = mean_preds
    #             y_steps[1::2] = mean_preds

    #             ax.plot(x_steps, y_steps, color='blue', alpha=1.0, linewidth=3)

    #             # xticks should be at bin centers
    #             # xticks should be at bin centers
    #             labels = categorical_label_maps.get(feature_name, None)
    #             bin_centers = (edges[:-1] + edges[1:]) / 2
    #             if labels is not None:
    #                 ax.set_xticks(bin_centers)
    #                 ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    #             else:
    #                 ax.set_xticks(bin_centers)
    #                 ax.set_xticklabels([str(x) for x in bin_centers], rotation=30, ha='right', fontsize=10)


    #             ax.set_xlim(edges[0], edges[-1])
    #         else:
    #             for preds in preds_all:
    #                 ax.plot(x_vals_real, preds, color='blue', alpha=0.05, linewidth=1)
    #             ax.plot(x_vals_real, mean_preds, color='blue', alpha=1.0, linewidth=2)

    #         # --- Axis settings ---
    #         if feature_name in plot_xlims:
    #             ax.set_xlim(*plot_xlims[feature_name])
    #         else:
    #             ax.set_xlim(np.min(x_vals_real), np.max(x_vals_real))

    #         if row_idx == 0:
    #             ax.set_ylim(global_min_raw, global_max_raw)
    #         else:
    #             ax.set_ylim(global_min_pattern, global_max_pattern)


    #         if row_idx == 0 and feature_idx == 0:
    #             ax.set_ylabel('NAM\nRecidivism Risk', fontsize=14)
    #         if row_idx == 1 and feature_idx == 0:
    #             ax.set_ylabel('PatternNAM\nRecidivism Risk', fontsize=14)

    #         if row_idx == 0:
    #             ax.set_title(feature_name, fontsize=12)

    #         ax.tick_params(labelsize=10)

    # plt.tight_layout()
    # plt.savefig(f"{OUT_DIR}/recid_combined_{n_interactions}ints_{num_learners}models.png", dpi=300)
    # plt.close()

