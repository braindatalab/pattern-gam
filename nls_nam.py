import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
from sklearn.model_selection import train_test_split
from nam.wrapper import NAMClassifier
import torch
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os
from interpret.utils import measure_interactions

random_state = 2025
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_pattern(model, X_train):
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()

    Z_all = []

    for learner in model.models:
        learner.eval()
        learner.to(device)  # Move learner to GPU
        with torch.no_grad():
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
            _, _, main_effects_out, interaction_effects_out = learner.forward(X_train_tensor)
            if interaction_effects_out.shape[1] > 0:
                Z = torch.cat([main_effects_out, interaction_effects_out], dim=1)  # (n_samples, total_units)
            else:
                Z = main_effects_out  # (n_samples, num_features)
            Z_all.append(Z.cpu().numpy())  # move to cpu to convert to numpy

    Z_all = np.stack(Z_all, axis=0)  # (num_learners, n_samples, total_units)
    Z_mean = np.mean(Z_all, axis=0)  # (n_samples, total_units)

    cov_Z = np.cov(Z_mean, rowvar=False)  # (total_units, total_units)

    linreg = LogisticRegression(penalty=None, fit_intercept=False, max_iter=1000, random_state=random_state)
    linreg.fit(Z_mean, y_train)

    w_Z = linreg.coef_[0]  # (total_units,)

    pattern = cov_Z @ w_Z  # (total_units,)

    return pattern


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

results = [] 


from glob import glob

for data_path in glob('./data/nls/*.csv'):
    scen_name = data_path.split('/')[2].replace('.csv','')
    df = pd.read_csv(data_path, delimiter=',')
    feature_columns = [col for col in df.columns if col != 'label']
    X = df[feature_columns].values
    y = df['label'].values

    # Train/val/test split
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # First split: Train+Val and Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Then split: Train and Val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=random_state, stratify=y_temp
    )

    d = X_train.shape[1]

    # # --- Train and plot for different n_interactions ---
    n_interactions_list = [0,16,32,'full']
    num_learners = 10

    SAVE_DIR = './saved_models'
    os.makedirs(SAVE_DIR, exist_ok=True)

    for n_interactions in n_interactions_list:

        if n_interactions == 0:
            interaction_pairs = []
        elif n_interactions == 'full':
            interaction_pairs = [(i, j) for i in range(d) for j in range(i+1, d) if i < j]
        else:
            interaction_pairs = FAST(X_train, y_train, n_interactions=n_interactions)

        model = NAMClassifier(
            num_epochs=500,
            num_learners=num_learners,
            metric='auroc',
            interaction_pairs=interaction_pairs,
            early_stop_mode='max',
            device=device_str,
            monitor_loss=False,
            n_jobs=10,
            random_state=random_state
        )

        model.fit(X_train, y_train)

        # --- Save full NAM model after training ---
        
        save_path = os.path.join(SAVE_DIR, f'{scen_name}_{n_interactions}ints.pth')

        torch.save(
            {'model_state_dicts': [
                {k: v.cpu() for k, v in learner.state_dict().items()}
                for learner in model.models
            ]},
            save_path
        )

        pred = model.predict_proba(X_test)
        auroc = sk_metrics.roc_auc_score(y_test, pred)
        print(f"{scen_name} n_interactions={n_interactions}, AUROC={auroc:.3f}")

        # Save into results
        results.append({
            'scenario': scen_name,
            'n_interactions': n_interactions,
            'auroc': auroc
        })

        feature_names = [f'x_{i}' for i in range(d)]

        # --- Precompute global min/max for scaling plots
        global_preds_raw = []
        global_preds_pattern = []

        pattern = compute_pattern(model, X_train)  # Assuming you want pattern scaling too

        for feature_idx in range(d):
            x_vals_real = np.unique(X_train[:, feature_idx])

            X_query = np.zeros((len(x_vals_real), d))
            X_query[:, feature_idx] = 2 * (x_vals_real - np.min(X_train[:, feature_idx])) / (np.max(X_train[:, feature_idx]) - np.min(X_train[:, feature_idx])) - 1

            for learner in model.models:
                learner.eval()
                learner.to(device)  # move learner to GPU
                with torch.no_grad():
                    X_query_tensor = torch.tensor(X_query, dtype=torch.float32, device=device)
                    _, _, main_effects_out, _ = learner.forward(X_query_tensor)
                    preds = main_effects_out[:, feature_idx].cpu().numpy().flatten()  # back to CPU for numpy
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
                    learner.to(device)  # move learner to GPU
                    with torch.no_grad():
                        X_query_tensor = torch.tensor(X_query, dtype=torch.float32, device=device)
                        _, _, main_effects_out, _ = learner.forward(X_query_tensor)
                        preds = main_effects_out[:, feature_idx].cpu().numpy().flatten()  # back to CPU for numpy
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

        plt.savefig(f'{OUT_DIR}/{scen_name}_{n_interactions}ints.png', bbox_inches='tight')
        plt.savefig(f'{OUT_DIR}/{scen_name}_{n_interactions}ints_hires.png', bbox_inches='tight', dpi=300)
        plt.close()
