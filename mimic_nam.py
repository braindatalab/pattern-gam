import pandas as pd
import numpy as np
import os
import warnings
import pickle
import torch
import seaborn as sns # For density plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # <<< Added for layout
import matplotlib.patches as mpatches
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression # For pattern calculation linear probe
from sklearn.preprocessing import StandardScaler
from sklearn import metrics as sk_metrics # For AUROC

from scipy.stats import pearsonr
from nam.wrapper import NAMClassifier
from interpret.utils import measure_interactions

random_state = 2025

# --- Helper for finding interaction pairs ---
# (Keep the FAST function as is)
def FAST(X_train, y_train, n_interactions, init_score=None, feature_names=None, feature_types=None):
    top_interactions = measure_interactions(
        X_train, y_train, interactions=n_interactions, init_score=init_score,
        feature_names=feature_names, feature_types=feature_types
    )
    return [(i, j) for (i, j), _ in top_interactions]

# Suppress warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Ignore specific seaborn warning about discrete variables for KDE
warnings.filterwarnings("ignore", message="When grouping with a length-1 list-like")


# --- Device Setup ---
if torch.cuda.is_available(): device_str = "cuda"; print("CUDA available.")
else: device_str = "cpu"; print("CUDA not available.")
device = torch.device(device_str)

# --- Configuration ---
output_path = './data/mimiciv'; figure_path = './figures'
window_hours = 24
os.makedirs(output_path, exist_ok=True); os.makedirs(figure_path, exist_ok=True)

def compute_patterns(
    model,
    X_train_tensor_scaled,
    y_train,
    standardize_Z: bool = False,
):
    """
    Compute PatternGAM parameters for plotting:
      Blue:  \tilde f_j(x) = w_j * f_j(x)
      Red:   f^{PGM}_j(x) = b_j * \tilde f_j(x) + d_j

    Steps per learner:
      1) Get Z = main_effects_out(X_train)               # (n_samples, D), columns are f_j(X_j)
      2) Fit multivariate LLR on Z (optionally on standardized Z) -> w
         - If standardize_Z=True, convert coef back to original Z scale
      3) For each feature j:
           tilde_f_j = w[j] * Z[:, j]
           Fit univariate LLR y ~ tilde_f_j with intercept=True -> (b_j, d_j)

    Returns:
      w_list : list[len=models] of (D,)   multivariate LLR weights per learner (on ORIGINAL Z scale)
      b_list : list[len=models] of (D,)   univariate slopes per learner (fit_intercept=True)
      d_list : list[len=models] of (D,)   univariate intercepts per learner (fit_intercept=True)
      avg_b  : (D,)                       average b across learners
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    # y -> numpy (float64 for sklearn)
    if isinstance(y_train, torch.Tensor):
        y_np = y_train.detach().cpu().numpy().astype(np.float64)
    else:
        y_np = np.asarray(y_train, dtype=np.float64)

    num_features = X_train_tensor_scaled.shape[1]
    w_list, b_list, d_list = [], [], []

    # Helper for baseline log-odds when predictor is (near-)constant
    def _baseline_logit(y):
        p = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        return float(np.log(p / (1.0 - p)))

    for learner in model.models:
        learner.eval(); learner.to(device)
        with torch.no_grad():
            _, _, main_effects_out, _ = learner.forward(X_train_tensor_scaled)
            Z_np = main_effects_out.detach().cpu().numpy().astype(np.float64)  # (n, D), columns f_j(X_j)

        # --- Step 2: multivariate LLR on Z -> w (optionally standardize Z) ---
        try:
            if standardize_Z:
                mu = Z_np.mean(axis=0)
                sigma = Z_np.std(axis=0, ddof=0)
                sigma_safe = np.where(sigma > 1e-12, sigma, 1.0)
                Z_std = (Z_np - mu) / sigma_safe

                lr_multi = LogisticRegression(
                    penalty=None, fit_intercept=True, max_iter=2000, random_state=random_state
                )
                lr_multi.fit(Z_std, y_np)
                w_std = lr_multi.coef_[0].astype(np.float64)         # coef wrt standardized Z
                # Convert to original Z scale: w = w_std / sigma
                w = (w_std / sigma_safe).astype(np.float64)
            else:
                lr_multi = LogisticRegression(
                    penalty=None, fit_intercept=True, max_iter=2000, random_state=random_state
                )
                lr_multi.fit(Z_np, y_np)
                w = lr_multi.coef_[0].astype(np.float64)              # coef wrt original Z
        except Exception:
            w = np.ones(num_features, dtype=np.float64)

        # --- Step 3: univariate LLRs on tilde_f_j with intercept=True -> (b_j, d_j) ---
        b = np.zeros(num_features, dtype=np.float64)
        d0 = np.zeros(num_features, dtype=np.float64)

        for j in range(num_features):
            tilde_f_j = (w[j] * Z_np[:, j]).reshape(-1, 1)

            # Degenerate predictor -> zero slope, baseline intercept
            if not np.isfinite(tilde_f_j).all() or np.std(tilde_f_j) < 1e-10:
                b[j]  = 0.0
                # d0[j] = _baseline_logit(y_np)
                d0[j] = 0.0

                continue

            try:
                lr_uni = LogisticRegression(
                    penalty=None, fit_intercept=True, max_iter=2000, random_state=random_state + j
                )
                lr_uni.fit(tilde_f_j, y_np)
                b[j]  = float(lr_uni.coef_[0, 0])
                d0[j] = float(lr_uni.intercept_[0])
            except Exception:
                b[j]  = 0.0
                # d0[j] = _baseline_logit(y_np)
                d0[j] = 0.0


        w_list.append(w)
        b_list.append(b)
        d_list.append(d0)

    avg_b = np.mean(np.stack(b_list, axis=0), axis=0) if b_list else np.zeros(num_features, dtype=np.float64)
    return w_list, b_list, d_list, avg_b


def compute_patterns_over_Zmean(
    model,
    X_train_tensor_scaled,
    y_train,
    *,
    device=torch.device("cpu"),
    include_interactions=False,   # keep False to match your shape-function usage
    standardize_Z=False,          # if True: fit on z-scored Z_mean, then map coef back
    random_state=2025,
):
    """
    Returns single (D,) vectors:
      w_vec: multivariate LLR weights fit on Z_mean (ensemble-average main effects)
      b_vec, d_vec: univariate LLR slope & intercept fit on \tilde z_j = w_j * Z_mean[:, j]
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    # 1) Build Z_mean over training samples
    X_tensor = X_train_tensor_scaled.to(dtype=torch.float32, device=device)
    Z_list = []
    with torch.no_grad():
        for learner in model.models:
            learner.eval(); learner.to(device)
            _, _, me_out, int_out = learner.forward(X_tensor)
            Z = me_out
            if include_interactions and (int_out is not None) and (int_out.shape[1] > 0):
                Z = torch.cat([me_out, int_out], dim=1)
            Z_list.append(Z.detach().cpu().numpy())
    Z_all  = np.stack(Z_list, axis=0)          # (L, N, D)
    Z_mean = np.mean(Z_all, axis=0).astype(np.float64)  # (N, D)

    # 2) Prepare y
    if isinstance(y_train, torch.Tensor):
        y_np = y_train.detach().cpu().numpy().astype(int).ravel()
    else:
        y_np = np.asarray(y_train, int).ravel()

    D = Z_mean.shape[1]

    # 3) Multivariate LLR on Z_mean -> w_vec
    if standardize_Z:
        mu = Z_mean.mean(axis=0)
        sd = Z_mean.std(axis=0, ddof=0)
        sd_safe = np.where(sd > 0, sd, 1.0)
        Z_fit = (Z_mean - mu) / sd_safe
    else:
        sd_safe = None
        Z_fit = Z_mean

    lr_multi = LogisticRegression(
        penalty=None, solver="lbfgs", fit_intercept=True,
        max_iter=10000, random_state=random_state
    )
    lr_multi.fit(Z_fit, y_np)
    w_fit = lr_multi.coef_[0].astype(np.float64)
    w_vec = (w_fit / sd_safe) if standardize_Z else w_fit   # map back if standardized

    # 4) Univariate LLRs on \tilde z_j = w_j * Z_mean[:, j]  -> b_vec, d_vec
    b_vec = np.zeros(D, dtype=np.float64)
    d_vec = np.zeros(D, dtype=np.float64)

    def _baseline_logit(y):
        p = np.clip(np.mean(y), 1e-6, 1-1e-6)
        return float(np.log(p/(1-p)))

    for j in range(D):
        tilde_z_j = (w_vec[j] * Z_mean[:, j]).reshape(-1, 1)
        if not np.isfinite(tilde_z_j).all() or np.std(tilde_z_j) < 1e-10:
            b_vec[j] = 0.0
            d_vec[j] = _baseline_logit(y_np)
            continue
        try:
            lr_uni = LogisticRegression(
                penalty=None, solver="lbfgs", fit_intercept=True,
                max_iter=10000, random_state=random_state + j
            )
            lr_uni.fit(tilde_z_j, y_np)
            b_vec[j] = float(lr_uni.coef_[0, 0])
            d_vec[j] = float(lr_uni.intercept_[0])
        except Exception:
            b_vec[j] = 0.0
            d_vec[j] = _baseline_logit(y_np)

    return w_vec, b_vec, d_vec




def get_outlier_mask_iqr(df, columns, k=3.0):
    """
    Returns a boolean mask for rows to KEEP (not outliers),
    using both upper and lower IQR thresholds.
    """
    keep_mask = np.ones(len(df), dtype=bool)

    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR

        keep_mask &= df[col].between(lower_bound, upper_bound)

    return keep_mask

def get_symmetric_ticks(min_val, max_val):
    abs_max = max(abs(min_val), abs(max_val))
    if abs_max == 0: return [0.0] # Handle case where min/max are both 0
    return np.round([-abs_max, 0.0, abs_max], 2)


def plot_shape_functions(
    model,
    X_train_tensor_scaled,
    X_train_original,
    patterns,  # UNUSED; kept for API compatibility
    feature_names,
    scaler,
    continuous_feature_indices,
    categorical_feature_indices,
    categorical_mappings=None,
    w_list=None,               # per-learner w vectors (for \tilde f)
    b_list=None,               # per-learner b vectors
    d_list=None,               # per-learner intercept vectors (for +d)
    # ---- axis control ----
    force_same_y_limits=True,       # same symmetric limits for BOTH curves & ALL subplots
    global_ylim=None,               # optional override (ymin, ymax); if None inferred from both curves
    overlay_on_single_axis=False,   # draw both curves on left axis; right axis used but ticks hidden
    num_cols=8,
    individual_learner_alpha=0.08,
    group1_indices=[], group2_indices=[],
    group1_color=(0.6, 0.8, 1.0, 0.3), group2_color=(1.0, 0.75, 0.5, 0.3),
    group3_indices=None, group3_color=None, group3_label=None,
    group1_label="High Impact", group2_label="Nullified",
    filename='', figsize=(20, 8), show=False,
    indices_to_display=None,
    feature_group_labels_map=None,
    # ---- annotations ----
    show_coeff_labels=True,         # annotate each panel with avg b and avg d
):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    from matplotlib.ticker import FormatStrFormatter
    import torch, os

    D_total = X_train_tensor_scaled.shape[1]
    marital_status_map = {0:'Divorced',1:'Married',2:'Single',3:'Unknown',4:'Widowed'}

    # which features to plot
    if indices_to_display is None:
        features_to_process_indices = range(D_total)
    else:
        features_to_process_indices = indices_to_display

    num_features_to_plot = len(features_to_process_indices)
    if num_features_to_plot == 0:
        print("No features to plot."); return

    num_cols = max(1, num_cols)
    num_rows = int(np.ceil(num_features_to_plot / num_cols))
    adj_figsize = (figsize[0], max(figsize[1], num_rows * 2.5 + 2)) if indices_to_display is None else figsize

    fig = plt.figure(figsize=adj_figsize, constrained_layout=True)
    gs = gridspec.GridSpec(num_rows, num_cols, figure=fig, hspace=0.01, wspace=0.075)

    feature_preds_data = []
    global_min = np.inf
    global_max = -np.inf

    FMT_2DP = FormatStrFormatter('%.2f')

    def _padded_sym_limits_from_m(m, pad_frac=0.06, min_pad=0.02):
        if (not np.isfinite(m)) or (m < 1e-12):
            m = 1.0
        m_pad = m * (1.0 + pad_frac) + min_pad
        ylim = (-m_pad, m_pad)
        ticks = np.array([-m_pad, 0.0, m_pad], dtype=float)
        return ylim, ticks
    
    def _resolve_param(param, ell, j, D):
        # param can be: None | scalar | (D,) array | list of (D,) arrays
        if param is None:
            return None
        import numpy as np
        if isinstance(param, list):
            return float(np.asarray(param[ell], dtype=float)[j])
        arr = np.asarray(param, dtype=float)
        if arr.ndim == 0:
            return float(arr)                       # broadcast later if you ever use vector ops
        if arr.ndim == 1 and arr.shape[0] == D:
            return float(arr[j])
        raise ValueError(f"Unsupported param shape {arr.shape}; expected scalar, (D,), or list of (D,).")


    # # Precompute avg b,d across learners for annotations
    # avg_b_vec = None; avg_d_vec = None
    # if b_list is not None:
    #     try:
    #         avg_b_vec = np.mean(np.stack(b_list, axis=0), axis=0)
    #     except Exception:
    #         pass
    # if d_list is not None:
    #     try:
    #         avg_d_vec = np.mean(np.stack(d_list, axis=0), axis=0)
    #     except Exception:
    #         pass

    # --- helpers at top of the function (after imports) ---
    import numpy as np
    def _to_feature_vec(x, D):
        """Return a float (D,) vector from list/ndarray/scalar; broadcast scalars."""
        if x is None:
            return None
        x = np.asarray(x)
        if x.ndim == 0:
            return np.full(D, float(x), dtype=float)
        if x.ndim == 1:
            if x.shape[0] != D:
                raise ValueError(f"Vector length {x.shape[0]} != D={D}")
            return x.astype(float, copy=False)
        # allow shape (1,D) or (D,1)
        if x.ndim == 2 and (1 in x.shape) and (max(x.shape) == D):
            return x.reshape(-1).astype(float, copy=False)
        raise ValueError(f"Unsupported shape for feature vector: {x.shape}")
    
    # --- inside plot_shape_functions, before the precompute loop ---
    D = X_train_tensor_scaled.shape[1]

    # Robust ensemble averages for annotations (support list, 1-D array, or scalar)
    def _avg_over_learners(maybe_list_or_vec):
        if maybe_list_or_vec is None:
            return None
        if isinstance(maybe_list_or_vec, list):
            mats = np.stack([_to_feature_vec(v, D) for v in maybe_list_or_vec], axis=0)  # (L, D)
            return np.nanmean(mats, axis=0)
        # ndarray or scalar
        return _to_feature_vec(maybe_list_or_vec, D)

    avg_b_vec = _avg_over_learners(b_list)
    avg_d_vec = _avg_over_learners(d_list)
    avg_w_vec = _avg_over_learners(w_list)

    # avg g = average of (b*w). If lists -> mean of per-learner products; else product of vectors.
    if isinstance(b_list, list) and isinstance(w_list, list) and len(b_list) == len(w_list) and len(b_list) > 0:
        bw_stack = np.stack([_to_feature_vec(b, D) * _to_feature_vec(w, D) for b, w in zip(b_list, w_list)], axis=0)
        avg_g_vec = np.nanmean(bw_stack, axis=0)
    elif (avg_b_vec is not None) and (avg_w_vec is not None):
        avg_g_vec = avg_b_vec * avg_w_vec
    else:
        avg_g_vec = None



    # ---------- Precompute curves + (if needed) global limits ----------
    for feature_idx in features_to_process_indices:
        if feature_idx >= D_total:
            print(f"Warning: feature_idx {feature_idx} out of bounds. Skipping.")
            continue

        is_cont = feature_idx in continuous_feature_indices
        vmin = X_train_tensor_scaled[:, feature_idx].min().item()
        vmax = X_train_tensor_scaled[:, feature_idx].max().item()
        if is_cont:
            x_vals_scaled = torch.linspace(vmin, vmax, steps=200, device=X_train_tensor_scaled.device)
        else:
            x_vals_scaled = torch.sort(torch.unique(X_train_tensor_scaled[:, feature_idx])).values.to(X_train_tensor_scaled.device)

        X_mean = torch.mean(X_train_tensor_scaled, dim=0, keepdim=True)
        X_query = X_mean.repeat(len(x_vals_scaled), 1)
        X_query[:, feature_idx] = x_vals_scaled

        preds_tilde_all = []
        preds_pgm_all   = []

        for ell, learner in enumerate(model.models):
            learner.eval(); learner.to(X_train_tensor_scaled.device)
            with torch.no_grad():
                _, _, me_out, _ = learner.forward(X_query)
                f_raw = me_out[:, feature_idx].detach().cpu().numpy().ravel()

            # w_j = 1.0
            # if w_list is not None and ell < len(w_list) and feature_idx < len(w_list[ell]):
            #     w_j = float(w_list[ell][feature_idx])

            # b_j = 1.0
            # if b_list is not None and ell < len(b_list) and feature_idx < len(b_list[ell]):
            #     b_j = float(b_list[ell][feature_idx])

            # d_j = 0.0
            # if d_list is not None and ell < len(d_list) and feature_idx < len(d_list[ell]):
            #     d_j = float(d_list[ell][feature_idx])

            # Resolve w_j (supports per-learner list OR single (D,) vector)
            if w_list is None:
                w_j = 1.0
            elif isinstance(w_list, np.ndarray) and w_list.ndim == 1:
                w_j = float(w_list[feature_idx])                    # global-from-Zmean
            else:
                w_j = float(w_list[ell][feature_idx])              # per-learner

            # Resolve b_j, d_j similarly
            if b_list is None:
                b_j = 1.0
            elif isinstance(b_list, np.ndarray) and b_list.ndim == 1:
                b_j = float(b_list[feature_idx])                   # global-from-Zmean
            else:
                b_j = float(b_list[ell][feature_idx])              # per-learner

            # if d_list is None:
            #     d_j = 0.0
            # elif isinstance(d_list, np.ndarray) and d_list.ndim == 1:
            #     d_j = float(d_list[feature_idx])                   # global-from-Zmean
            # else:
            #     d_j = float(d_list[ell][feature_idx])              # per-learner

            tilde = w_j * f_raw
            pgm   = b_j * tilde

            preds_tilde_all.append(tilde)
            preds_pgm_all.append(pgm)


        preds_tilde_all = np.stack(preds_tilde_all, axis=0) if preds_tilde_all else np.zeros((1, len(x_vals_scaled)))
        # preds_pgm_all   = np.stack(preds_pgm_all,   axis=0) if preds_pgm_all   else np.zeros((1, len(x_vals_scaled)))
        mean_tilde = np.mean(preds_tilde_all, axis=0)
        # mean_pgm   = np.mean(preds_pgm_all,   axis=0)

        # If b,d are global 1D arrays, define red mean from mean_tilde; don't draw per-learner red
        use_global_bd = (isinstance(b_list, np.ndarray) and b_list.ndim == 1)
        if use_global_bd and isinstance(d_list, np.ndarray) and d_list.ndim == 1:
            mean_pgm = float(b_list[feature_idx]) * mean_tilde + float(d_list[feature_idx])
            preds_pgm_all = None  # suppress per-learner red
        else:
            # keep your existing per-learner red construction
            mean_pgm = np.mean(preds_pgm_all, axis=0)
            preds_pgm_all   = np.stack(preds_pgm_all,   axis=0) if preds_pgm_all   else np.zeros((1, len(x_vals_scaled)))


        # track combined global extrema (only used if force_same_y_limits=True)
        local_min = np.nanmin([mean_tilde.min(initial=np.inf),  mean_pgm.min(initial=np.inf)])
        local_max = np.nanmax([mean_tilde.max(initial=-np.inf), mean_pgm.max(initial=-np.inf)])
        if np.isfinite(local_min): global_min = min(global_min, local_min)
        if np.isfinite(local_max): global_max = max(global_max, local_max)

        # x-axis in original units (continuous)
        x_plot = x_vals_scaled.detach().cpu().numpy()
        if is_cont and scaler is not None:
            dummy = np.zeros((len(x_vals_scaled), len(continuous_feature_indices)))
            cont_idx = list(continuous_feature_indices).index(feature_idx)
            dummy[:, cont_idx] = x_plot
            x_plot = scaler.inverse_transform(dummy)[:, cont_idx]

        feature_preds_data.append({
            "feature_idx": feature_idx,
            "feature_name": feature_names[feature_idx],
            "x_plot_axis": x_plot,
            "is_continuous": is_cont,
            "preds_raw_all": preds_tilde_all,               # black
            "mean_raw_preds": mean_tilde,
            "preds_scaled_pattern_all": preds_pgm_all,      # red (with +d)
            "mean_scaled_pattern_preds": mean_pgm,
        })

    # ---------- Global limits (only if requested) ----------
    if force_same_y_limits:
        if global_ylim is not None:
            lo, hi = global_ylim
            m = max(abs(lo), abs(hi))
        else:
            if not np.isfinite(global_min) or not np.isfinite(global_max):
                m = 1.0
            else:
                m = max(abs(global_min), abs(global_max))
        global_ylim_used, global_ticks_used = _padded_sym_limits_from_m(m, pad_frac=0.06, min_pad=0.02)
    else:
        global_ylim_used = None
        global_ticks_used = None

    # ---------- Plot ----------
    for plot_idx, data in enumerate(feature_preds_data):
        r, c = divmod(plot_idx, num_cols)
        ax = fig.add_subplot(gs[r, c])
        ax2 = None if overlay_on_single_axis else ax.twinx()

        # KDE inset
        feat_idx = data['feature_idx']
        try:
            ax_inset = ax.inset_axes([0.05, 1.02, 0.9, 0.15], transform=ax.transAxes)
            sns.kdeplot(X_train_original.iloc[:, feat_idx], ax=ax_inset, color='darkgrey', warn_singular=False, bw_adjust=0.5)
            ax_inset.axis('off')
        except Exception:
            pass

        x_plot = data['x_plot_axis']
        is_cont = data['is_continuous']
        preds_tilde_all = data['preds_raw_all']
        mean_tilde = data['mean_raw_preds']
        preds_pgm_all = data['preds_scaled_pattern_all']
        mean_pgm = data['mean_scaled_pattern_preds']

        if is_cont:
            plot_kwargs = {'linewidth': 0.75}
            plot_mean_kwargs = {'linewidth': 1.0}
            plot_left = ax.plot
            plot_right = (ax.plot if overlay_on_single_axis else ax2.plot)
            x_axis = x_plot
            prep = lambda y: y
        else:
            codes = np.sort(x_plot)
            if len(codes) > 1:
                edges = np.concatenate(([codes[0]-0.5], (codes[:-1]+codes[1:])/2.0, [codes[-1]+0.5]))
            elif len(codes) == 1:
                edges = [codes[0]-0.5, codes[0]+0.5]
            else:
                edges = [-0.5, 0.5]
            def _step(y): return np.append(y, y[-1])
            plot_kwargs = {'where': 'post', 'linewidth': 1.5}
            plot_mean_kwargs = {'where': 'post', 'linewidth': 1.0}
            plot_left = ax.step
            plot_right = (ax.step if overlay_on_single_axis else ax2.step)
            x_axis = edges
            prep = _step

        # limits/ticks: global (same for all) OR per-panel (computed here)
        if force_same_y_limits:
            ylim_use  = global_ylim_used
            yticks_use = global_ticks_used
        else:
            m1 = float(np.nanmax(np.abs(mean_tilde))) if mean_tilde.size else 0.0
            m2 = float(np.nanmax(np.abs(mean_pgm)))   if mean_pgm.size   else 0.0
            m  = max(m1, m2)
            ylim_use, yticks_use = _padded_sym_limits_from_m(m, pad_frac=0.06, min_pad=0.02)

        # # per-learner curves
        # for i in range(preds_tilde_all.shape[0]):
        #     plot_left(x_axis, prep(preds_tilde_all[i, :]), color='black', alpha=individual_learner_alpha, **plot_kwargs)
        #     plot_right(x_axis, prep(preds_pgm_all[i, :]),  color='red',   alpha=individual_learner_alpha, **plot_kwargs)

        # # means
        # plot_left(x_axis,  prep(mean_tilde), color='black', alpha=1.0, **plot_mean_kwargs)
        # plot_right(x_axis, prep(mean_pgm),   color='red',   alpha=1.0, **plot_mean_kwargs)

        # per-learner curves
        for i in range(preds_tilde_all.shape[0]):
            plot_left(x_axis, prep(preds_tilde_all[i, :]), color='black',
                    alpha=individual_learner_alpha, **plot_kwargs)
            if preds_pgm_all is not None:
                plot_right(x_axis, prep(preds_pgm_all[i, :]),  color='red',
                        alpha=individual_learner_alpha, **plot_kwargs)

        # means
        plot_left(x_axis,  prep(mean_tilde), color='black', alpha=1.0, **plot_mean_kwargs)
        plot_right(x_axis, prep(mean_pgm),   color='red',   alpha=1.0, **plot_mean_kwargs)


        # apply limits & ticks (left on all subplots; right ticks hidden)
        ax.set_ylim(ylim_use);  ax.set_yticks(yticks_use); ax.yaxis.set_major_formatter(FMT_2DP)
        if not overlay_on_single_axis:
            ax2.set_ylim(ylim_use); ax2.set_yticks([])

        # group shading
        if group1_indices and feat_idx in group1_indices: ax.set_facecolor(group1_color)
        elif group2_indices and feat_idx in group2_indices: ax.set_facecolor(group2_color)
        elif group3_indices and group3_color and feat_idx in group3_indices: ax.set_facecolor(group3_color)

        # x ticks / labels
        if not is_cont:
            ax.set_xticks(x_plot)
            fname = feature_names[feat_idx]
            if fname.lower().replace(" ", "_") == "marital_status":
                xtick_labels = [marital_status_map.get(int(val), str(val)) for val in x_plot]
            elif categorical_mappings and fname in categorical_mappings:
                mapping = categorical_mappings[fname]
                xtick_labels = [mapping.get(int(val), str(val)).replace('/', '/\n') for val in x_plot]
            else:
                xtick_labels = [str(int(val)) for val in x_plot]
            ax.set_xticklabels(xtick_labels, fontsize=13)
            ax.tick_params(axis='x', rotation=20, labelsize=13, pad=0.1)
        else:
            ax.tick_params(axis='x', rotation=0, labelsize=13)

        ax.xaxis.set_label_position('top')
        pretty = feature_names[feat_idx].replace(' mean','').replace('Temp C','Temperature').replace('_',' ').replace('24h','')
        ax.set_xlabel(pretty, fontsize=14, labelpad=-20)
        ax.tick_params(axis='y', rotation=0, labelsize=13)

        # # coefficient annotation (ensemble averages)
        # if show_coeff_labels and (avg_b_vec is not None or avg_d_vec is not None):
        #     b_val = None if avg_b_vec is None else float(avg_b_vec[feat_idx])
        #     d_val = None if avg_d_vec is None else float(avg_d_vec[feat_idx])
        #     lines = []
        #     if b_val is not None: lines.append(rf"$\bar b$: {b_val:.3f}")
        #     if d_val is not None: lines.append(rf"$\bar d$: {d_val:.3f}")
        #     if lines:
        #         ax.text(
        #             0.02, 0.95, "\n".join(lines),
        #             transform=ax.transAxes, ha='left', va='top',
        #             fontsize=9, color='dimgray',
        #             bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.2', alpha=0.8)
        #         )

        # coefficient annotation (ensemble averages)
        if show_coeff_labels:
            lines = []
            if (avg_w_vec is not None) and (group1_indices is None and group2_indices is None and group3_indices is None): lines.append(rf"$w_j$: {avg_w_vec[feat_idx]:.3f}")
            if (avg_b_vec is not None): lines.append(rf"$b_j$: {avg_b_vec[feat_idx]:.3f}")
            # if (avg_g_vec is not None): lines.append(rf"$\bar g=\bar b\,\bar w$: {avg_g_vec[feat_idx]:.3f}")
            if (avg_d_vec is not None): lines.append(rf"$\bar d$: {avg_d_vec[feat_idx]:.3f}")
            if lines:
                ax.text(0.7, 0.15, "\n".join(lines),
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=12, color='dimgray',
                        bbox=dict(facecolor='white', edgecolor='lightgray',
                                boxstyle='round,pad=0.2', alpha=0.8))


        ax.grid(True, linestyle='--', alpha=0.3)

    # Legend (group shading)
    legend_elements = []
    if group1_indices and any(idx in features_to_process_indices for idx in group1_indices):
        legend_elements.append(mpatches.Patch(facecolor=group1_color, label=group1_label))
    if group2_indices and any(idx in features_to_process_indices for idx in group2_indices):
        legend_elements.append(mpatches.Patch(facecolor=group2_color, label=group2_label))
    if group3_indices and any(idx in features_to_process_indices for idx in group3_indices):
        legend_elements.append(mpatches.Patch(facecolor=group3_color, label=group3_label))
    if legend_elements:
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=14, bbox_to_anchor=(0.5, -0.075))

    # Titles
    if group1_indices and group2_indices and group3_indices:
        title_y_pos = 0.99
        title_x1 = 0.0125
        title_x2 = 0.9875
    else:
        title_y_pos = 0.995
        title_x1 = 0.015
        title_x2 = 0.9875
    fig.text(title_x1, title_y_pos, r'$f^\text{GAM}$', ha='left', va='top', fontsize=16, color='black')
    fig.text(title_x2, title_y_pos, r'$f^\text{PGAM}$', ha='right', va='top', fontsize=16, color='red')

    # Save/show
    if filename:
        try:
            filepath_hires = os.path.join(figure_path, f'{filename}_hires.png')
            filepath_lowres = os.path.join(figure_path, f'{filename}.png')
            plt.savefig(filepath_hires, bbox_inches='tight', dpi=300)
            plt.savefig(filepath_lowres, bbox_inches='tight', dpi=150)
            print(f"Saved plots to {filepath_hires} and {filepath_lowres}")
        except Exception as e:
            print(f"Error saving figure: {e}")
    if show: plt.show()
    plt.close(fig)

    return {
        "feature_preds_data": feature_preds_data,
        "used_global_ylim": global_ylim_used if force_same_y_limits else None,
        "force_same_y_limits": force_same_y_limits,
        "overlay_on_single_axis": overlay_on_single_axis,
    }





####################################################################

if __name__ == "__main__":

    print(f"======= Processing Window: {window_hours} hours =======")
    processed_data_file = os.path.join(output_path, f'mimic_mortality_{window_hours}h_processed.csv')
    mapping_file = os.path.join(output_path, f'categorical_mappings_{window_hours}h.pkl')
    target = 'hospital_expire_flag'; group_col = 'subject_id'; id_cols = ['subject_id', 'hadm_id']

    try:
        df_loaded = pd.read_csv(processed_data_file)
        with open(mapping_file, 'rb') as f: categorical_mappings = pickle.load(f)
        print("Data loaded successfully.")
    
    except FileNotFoundError: print(f"Error: Data files not found."); exit()
    # except Exception as e: print(f"Error loading data: {e}"); exit() # User preference: less try-except

    y = df_loaded[target]
    groups = df_loaded[group_col]
    # feature_cols = [col for col in df_loaded.columns if col not in [target, group_col] + id_cols]
    feature_cols = [col for col in df_loaded.columns if col not in [target, group_col] + id_cols]
    
    # --- Drop 'language' feature ---
    feature_to_drop = 'language'
    if feature_to_drop in feature_cols:
        print(f"Dropping feature: {feature_to_drop}")
        feature_cols.remove(feature_to_drop)
    else:
        print(f"Feature '{feature_to_drop}' not found in feature_cols.")

    X = df_loaded[feature_cols].copy()

    iqr_columns = X.select_dtypes(include=['number']).columns.tolist()
    keep_mask = get_outlier_mask_iqr(X, iqr_columns, k=5.0)
    X = X[keep_mask].reset_index(drop=True)
    y = y[keep_mask].reset_index(drop=True)
    groups = groups[keep_mask].reset_index(drop=True)

    print(X.shape)
    print(y.value_counts())

    categorical_cols = [col for col in categorical_mappings.keys() if col in X.columns]
    continuous_cols = [col for col in X.columns if col not in categorical_cols]
    for col in categorical_cols: X[col] = X[col].astype(int)
    
    print("Splitting data...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    X_train_original = X.iloc[train_idx].copy() 

    print("Scaling continuous features...")
    scaler = StandardScaler()
    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

    print("Converting data to tensors...")
    X_train_tensor_scaled = torch.as_tensor(X_train[feature_cols].values).to(dtype=torch.float32).to(device)
    y_train_tensor = torch.as_tensor(y_train.values).to(dtype=torch.float32).to(device)
    X_test_tensor_scaled = torch.as_tensor(X_test[feature_cols].values).to(dtype=torch.float32).to(device)
    y_test_tensor = torch.as_tensor(y_test.values).to(dtype=torch.float32).to(device)

    print("Training NAM model...")
    interaction_pairs = []
    model = NAMClassifier(
        num_epochs=50, num_learners=100, metric='auroc',
        interaction_pairs=interaction_pairs, early_stop_mode='max',
        monitor_loss=False, n_jobs=10, random_state=random_state
    )
    model.fit(X_train_tensor_scaled, y_train_tensor)

    torch.save({'model_state_dicts': [
        {k: v.cpu() for k, v in learner.state_dict().items()} for learner in model.models
    ]}, './models/mimic_nam_model.pth')

    print("\nEvaluating NAM model...")
    pred_proba = model.predict_proba(X_test_tensor_scaled)
    if isinstance(pred_proba, torch.Tensor): pred_proba = pred_proba.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy() if isinstance(y_test_tensor, torch.Tensor) else y_test.values
    auroc = sk_metrics.roc_auc_score(y_test_np, pred_proba)
    print(f"MIMIC NAM AUROC = {auroc:.4f}")

    # # w_list, b_list, avg_b = compute_patterns(model, X_train_tensor_scaled, y_train_tensor)
    # w_list, b_list, d_list, avg_b = compute_patterns(
    #     model, X_train_tensor_scaled, y_train_tensor, standardize_Z=False  # or True
    # )

    w_list, b_list, d_list = compute_patterns_over_Zmean(
        model, X_train_tensor_scaled, y_train_tensor,
        device=device, include_interactions=False, standardize_Z=False
    )



    # --- Calculate Feature Correlations with Target ---
    print("\n--- Calculating Feature Correlations with Target ---")
    # Ensure all necessary variables are available
    if ('y_train_tensor' in locals() or 'y_train' in locals()) and \
        model is not None and \
        'X_train_original' in locals() and \
        'X_train_tensor_scaled' in locals() and \
        'feature_cols' in locals():

        # Determine y_train_np from available y_train tensor or series/array
        if 'y_train_tensor' in locals() and y_train_tensor is not None:
            y_data_for_corr = y_train_tensor.cpu().numpy() if isinstance(y_train_tensor, torch.Tensor) else y_train_tensor.values
        elif 'y_train' in locals() and y_train is not None: # If y_train is a pandas Series from the split
            y_data_for_corr = y_train.values
        else:
            print("y_train data not available. Skipping correlation analysis.")
            y_data_for_corr = None

        if y_data_for_corr is not None:
            # all_learners_all_feature_effects_raw = []
            # Check model structure
            all_learners_main_effects_tensors = []
            if hasattr(model, 'models') and isinstance(model.models, list) and len(model.models) > 0:
                for learner_idx, nam_ensemble_member in enumerate(model.models):
                    if hasattr(nam_ensemble_member, 'eval'):
                        nam_ensemble_member.eval() # Set to evaluation mode
                    try:
                        with torch.no_grad():
                            # Assuming the 3rd output from NAM.forward() is a single tensor
                            # of shape (n_samples_train, num_features) containing all main feature effects.
                            _pred, _bias, main_effects_tensor_for_learner, _int_effects = nam_ensemble_member(X_train_tensor_scaled)
                            
                            # Basic shape check (optional, for debugging)
                            # expected_shape = (X_train_tensor_scaled.shape[0], X_train_tensor_scaled.shape[1])
                            # if main_effects_tensor_for_learner.shape != expected_shape:
                            #    print(f"Warning: Learner {learner_idx} main_effects_tensor shape mismatch. Expected {expected_shape}, got {main_effects_tensor_for_learner.shape}")
                            #    all_learners_main_effects_tensors = [] # Invalidate
                            #    break

                            all_learners_main_effects_tensors.append(main_effects_tensor_for_learner)
                    except Exception as e:
                        # print(f"Warning: Could not get effects from learner {learner_idx}: {e}") # Concise
                        all_learners_main_effects_tensors = [] # Signal failure
                        break
            
            if all_learners_main_effects_tensors: # Check if we got any tensors
                num_features = X_train_tensor_scaled.shape[1]
                n_samples_train = X_train_tensor_scaled.shape[0]
                avg_feature_effects_all_features = [] # This will be a list of (n_samples_train,) arrays

                for feature_jdx in range(num_features): # For each feature column index
                    # Collect the j-th column (effects for current feature_jdx) from all learners
                    feature_j_outputs_across_learners = [] # List of (n_samples_train,) numpy arrays
                    
                    for learner_tensor in all_learners_main_effects_tensors:
                        # learner_tensor is shape (n_samples_train, num_features)
                        # Extract the column corresponding to feature_jdx
                        feature_j_column_for_this_learner = learner_tensor[:, feature_jdx] # Shape (n_samples_train,)
                        feature_j_outputs_across_learners.append(feature_j_column_for_this_learner.cpu().numpy())
                    
                    if not feature_j_outputs_across_learners: # Should not happen if all_learners_main_effects_tensors is not empty
                        avg_feature_effects_all_features.append(np.full(n_samples_train, np.nan))
                        continue
                    
                    # Stack arrays: each is (n_samples_train,), result is (n_samples_train, num_learners)
                    stacked_feature_j_effects = np.stack(feature_j_outputs_across_learners, axis=1)
                    # Average across learners (axis=1) to get (n_samples_train,)
                    avg_f_j_output = np.mean(stacked_feature_j_effects, axis=1)
                    avg_feature_effects_all_features.append(avg_f_j_output)
                
                # Now, avg_feature_effects_all_features[i] should have the correct shape (n_samples_train,)
                # The rest of your correlation calculation code (starting from the "if len(avg_feature_effects_all_features) == num_features:" check)
                # should then work correctly.
                if len(avg_feature_effects_all_features) == num_features:
                    correlation_results = []
                    for i in range(num_features):
                        feature_name = feature_cols[i]
                        
                        # Original feature vs target
                        x_original = X_train_original.iloc[:, i].values
                        mask_before = ~np.isnan(x_original) & ~np.isnan(y_data_for_corr)
                        x_original_clean = x_original[mask_before]
                        y_clean_before = y_data_for_corr[mask_before]
                        
                        corr_before, p_val_before = np.nan, np.nan
                        if len(x_original_clean) >= 2 and np.std(x_original_clean) > 1e-6 and np.std(y_clean_before) > 1e-6:
                            corr_before, p_val_before = pearsonr(x_original_clean, y_clean_before)
                        
                        # Transformed feature (f_i output) vs target
                        f_i_outputs = avg_feature_effects_all_features[i] # This should now be (12492,)
                        mask_after = ~np.isnan(f_i_outputs) & ~np.isnan(y_data_for_corr)
                        f_i_outputs_clean = f_i_outputs[mask_after]
                        y_clean_after = y_data_for_corr[mask_after]

                        corr_after, p_val_after = np.nan, np.nan
                        if len(f_i_outputs_clean) >= 2 and np.std(f_i_outputs_clean) > 1e-6 and np.std(y_clean_after) > 1e-6:
                            corr_after, p_val_after = pearsonr(f_i_outputs_clean, y_clean_after)
                        
                        correlation_results.append({
                            "Feature": feature_name,
                            "Corr Original": corr_before,
                            "P-val Original": p_val_before,
                            "Corr Transformed f(x)": corr_after,
                            "P-val Transformed f(x)": p_val_after
                        })
                    
                    results_df = pd.DataFrame(correlation_results)
                    pd.set_option('display.precision', 3)
                    # pd.set_option('display.max_colwidth', None) # Not strictly needed for this df
                    print("\nFeature Correlation with Target (Hospital Expire Flag):")
                    print(results_df.to_string(index=False, float_format="%.3f"))
                    pd.reset_option('display.precision')
                    # pd.reset_option('display.max_colwidth')
                else:
                    print("Could not compute average feature effects for all features. Skipping correlation analysis.")
            else:
                print("Could not retrieve main feature effects tensors from model learners. Skipping correlation analysis.")
   
    else:
        print("Required data for correlation analysis not found. Skipping.")

    print("Plotting results...")
    continuous_feature_indices = [feature_cols.index(col) for col in continuous_cols]
    categorical_feature_indices = [feature_cols.index(col) for col in categorical_cols]
    ints_str = '_0ints' if not interaction_pairs else f'_{len(interaction_pairs)}ints'

    print("Computing PatternGAM parameters...")
    # w_list, b_list, avg_b = compute_patterns(model, X_train_tensor_scaled, y_train_tensor)

    # w_list, b_list, d_list, avg_b = compute_patterns(
    #     model, X_train_tensor_scaled, y_train_tensor, standardize_Z=False  # or True
    # )


    full_plot_ax_limits = plot_shape_functions(
        model=model,
        X_train_tensor_scaled=X_train_tensor_scaled,
        X_train_original=X_train_original,
        patterns=None,
        w_list=w_list, b_list=b_list, 
        # d_list=d_list,   # <<<
        d_list=None,   # <<<
        feature_names=feature_cols,
        scaler=scaler,
        categorical_mappings=categorical_mappings,
        continuous_feature_indices=continuous_feature_indices,
        categorical_feature_indices=categorical_feature_indices,
        force_same_y_limits=False,
        global_ylim=None,
        num_cols=5,
        individual_learner_alpha=0.025,
        filename=f'cameraready_mimic_all_features',
        figsize=(30, max(8, 3.0 * np.ceil(len(feature_cols)/7)) + 5),
        show=False,
        indices_to_display=None,
        overlay_on_single_axis=False # set True if you want just one y-axis per panel
    )

    #########################################


    # full_plot_ax_limits = plot_shape_functions(
    #     model=model,
    #     X_train_tensor_scaled=X_train_tensor_scaled,
    #     X_train_original=X_train_original,
    #     patterns=None,
    #     w_list=w_list,
    #     b_list=b_list,
    #     feature_names=feature_cols,
    #     scaler=scaler,
    #     categorical_mappings=categorical_mappings,
    #     continuous_feature_indices=continuous_feature_indices,
    #     categorical_feature_indices=categorical_feature_indices,
    #     num_cols=7,
    #     individual_learner_alpha=0.05,
    #     filename=f'nam_mimic_{window_hours}h_shapes_all_features{ints_str}',
    #     figsize=(22, max(8, 3.0 * np.ceil(len(feature_cols)/7))),
    #     show=True,
    #     indices_to_display=None,
    #     # key bits 
    #     force_same_y_limits=True,
    #     global_ylim=None,            # let it infer from the data; or pass (-1.2, 1.2)
    #     overlay_on_single_axis=False # set True if you want just one y-axis per panel
    # )

    # # ============================
    # # Refined groups: High Impact vs Nullified
    # # ============================
    # # import numpy as np
    # # from sklearn import metrics as sk_metrics

    # def _safe_auroc(y_true, scores):
    #     y = np.asarray(y_true).ravel()
    #     s = np.asarray(scores).ravel()
    #     if np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
    #         return 0.5
    #     try:
    #         return sk_metrics.roc_auc_score(y, s)
    #     except Exception:
    #         return 0.5

    # D = len(feature_cols)
    # L = len(model.models)

    # # Build training matrices of \tilde f and b\tilde f, averaged across learners
    # with torch.no_grad():
    #     tilde_mats = []
    #     pgam_mats  = []
    #     for ell, learner in enumerate(model.models):
    #         learner.eval(); learner.to(X_train_tensor_scaled.device)
    #         _, _, Z_train_ell, _ = learner.forward(X_train_tensor_scaled)  # (n_train, D)
    #         Z_np = Z_train_ell.detach().cpu().numpy().astype(np.float64)

    #         w_vec = w_list[ell] if ell < len(w_list) else np.ones(D, dtype=np.float64)
    #         b_vec = b_list[ell] if ell < len(b_list) else np.ones(D, dtype=np.float64)

    #         tilde_np = Z_np * np.asarray(w_vec, dtype=np.float64)[None, :]
    #         pgam_np  = tilde_np * np.asarray(b_vec, dtype=np.float64)[None, :]

    #         tilde_mats.append(tilde_np)
    #         pgam_mats.append(pgam_np)

    # tilde_stack = np.stack(tilde_mats, axis=0)               # (L, n_train, D)
    # pgam_stack  = np.stack(pgam_mats,  axis=0)               # (L, n_train, D)

    # tilde_mean = np.mean(tilde_stack, axis=0)                # (n_train, D)
    # pgam_mean  = np.mean(pgam_stack,  axis=0)                # (n_train, D)

    # sd_tilde = np.std(tilde_mean, axis=0)                    # (D,)
    # sd_pgam  = np.std(pgam_mean,  axis=0)                    # (D,)

    # # Optional discrimination term (kept for completeness; not needed for visual nullification)
    # if isinstance(y_train_tensor, torch.Tensor):
    #     y_train_np = y_train_tensor.detach().cpu().numpy().astype(np.int32)
    # else:
    #     y_train_np = np.asarray(y_train, dtype=np.int32)
    # auroc_per_feature = np.array([_safe_auroc(y_train_np, pgam_mean[:, j]) for j in range(D)], dtype=np.float64)
    # discr = 2.0 * (auroc_per_feature - 0.5)

    # # ---- Selections ----
    # N_PER_GROUP = 3
    # eps = 1e-12
    # suppression_ratio = sd_pgam / (sd_tilde + eps)           # ~0 => strongly nullified

    # # High Impact: rank by sd_pgam (or sd_pgam * max(discr,0))
    # impact_score = sd_pgam  # or sd_pgam * np.maximum(discr, 0.0)
    # gA_sorted = np.argsort(impact_score)[::-1]
    # gA = []
    # for idx in gA_sorted:
    #     if np.isfinite(impact_score[idx]):
    #         gA.append(int(idx))
    #     if len(gA) == N_PER_GROUP:
    #         break

    # # Nullified: high sd_tilde AND low sd_pgam
    # tilde_gate = np.quantile(sd_tilde[np.isfinite(sd_tilde)], 0.70) if np.isfinite(sd_tilde).any() else 0.0
    # pgam_low   = np.quantile(sd_pgam[np.isfinite(sd_pgam)],  0.30) if np.isfinite(sd_pgam).any()  else 0.0

    # mask = (sd_tilde >= tilde_gate) & (sd_pgam <= pgam_low)
    # cand = np.where(mask)[0]
    # # Rank by smallest suppression ratio, tie-break by largest sd_tilde
    # order = np.lexsort((-sd_tilde[cand], suppression_ratio[cand]))
    # gB = []
    # for j in cand[order]:
    #     if j in gA: continue
    #     gB.append(int(j))
    #     if len(gB) == N_PER_GROUP: break

    # # If not enough, relax pgam_low to 0.45 and then 0.60 quantiles
    # for q in (0.45, 0.60):
    #     if len(gB) >= N_PER_GROUP: break
    #     pgam_low_rel = np.quantile(sd_pgam[np.isfinite(sd_pgam)], q) if np.isfinite(sd_pgam).any() else pgam_low
    #     mask_rel = (sd_tilde >= tilde_gate) & (sd_pgam <= pgam_low_rel)
    #     cand_rel = np.where(mask_rel)[0]
    #     order_rel = np.lexsort((-sd_tilde[cand_rel], suppression_ratio[cand_rel]))
    #     for j in cand_rel[order_rel]:
    #         if len(gB) >= N_PER_GROUP: break
    #         if j in gA or j in gB: continue
    #         gB.append(int(j))

    # # ---- Plot refined subset with SAME limits for both curves ----
    # final_selected = gA + gB
    # labels_map = {i: ("High Impact" if i in gA else "Nullified Effect") for i in final_selected}

    # print("\nHigh Impact:")
    # for j in gA: print(f" - {feature_cols[j]} | sd_tilde={sd_tilde[j]:.3g}, sd_pgam={sd_pgam[j]:.3g}, ratio={suppression_ratio[j]:.3g}")
    # print("\nNullified:")
    # for j in gB: print(f" - {feature_cols[j]} | sd_tilde={sd_tilde[j]:.3g}, sd_pgam={sd_pgam[j]:.3g}, ratio={suppression_ratio[j]:.3g}")

    # if final_selected:
    #     num_plots = len(final_selected)
    #     gA_plot_indices = [i for i in final_selected if labels_map[i] == "High Impact"]
    #     gB_plot_indices = [i for i in final_selected if labels_map[i] == "Nullified Effect"]
    #     color_gA_shade = (60/255, 180/255, 75/255, 0.2)
    #     color_gB_shade = (1, 225/255, 25/255, 0.2)

    #     # Use the same symmetric limits as the full plot (if you saved them), or recompute from these features:
    #     if 'full_plot_ax_limits' in locals() and full_plot_ax_limits.get('used_global_ylim', None) is not None:
    #         same_ylim = full_plot_ax_limits['used_global_ylim']
    #     else:
    #         # compute from selected features' mean curves
    #         sel_data = [d for d in full_plot_ax_limits['feature_preds_data'] if d['feature_idx'] in final_selected]
    #         vals = []
    #         for d in sel_data:
    #             vals.append(d['mean_raw_preds']); vals.append(d['mean_scaled_pattern_preds'])
    #         vmin = min(np.min(v) for v in vals if v.size)
    #         vmax = max(np.max(v) for v in vals if v.size)
    #         m = max(abs(vmin), abs(vmax)); m = 1.0 if not np.isfinite(m) or m < 1e-12 else float(m)
    #         same_ylim = (-m, m)

    #     full_plot_ax_limits = plot_shape_functions(
    #         model=model,
    #         X_train_tensor_scaled=X_train_tensor_scaled,
    #         X_train_original=X_train_original,
    #         patterns=None,
    #         w_list=w_list, b_list=b_list, d_list=d_list,   # <<<
    #         feature_names=feature_cols,
    #         scaler=scaler,
    #         categorical_mappings=categorical_mappings,
    #         continuous_feature_indices=continuous_feature_indices,
    #         categorical_feature_indices=categorical_feature_indices,
    #         num_cols=num_plots if num_plots > 0 else 1,
    #         individual_learner_alpha=0.05,
    #         filename=f'nam_mimic_{window_hours}h_refined_groups_shapes{ints_str}',
    #         figsize=(max(8, num_plots * 4), 3.5),
    #         show=True,
    #         indices_to_display=final_selected,
    #         feature_group_labels_map=labels_map,
    #         # critical bits ↓
    #         force_same_y_limits=True,
    #         global_ylim=same_ylim,         # same limits for both curves and all panels
    #         overlay_on_single_axis=False,  # set True if you want a single y-axis per panel
    #         # shading
    #         group1_indices=gA_plot_indices, group1_color=color_gA_shade, group1_label="High Impact",
    #         group2_indices=gB_plot_indices, group2_color=color_gB_shade, group2_label="Nullified Effect",
    #         group3_indices=None
    #     )

    #     # plot_shape_functions(
    #     #     model=model,
    #     #     X_train_tensor_scaled=X_train_tensor_scaled,
    #     #     X_train_original=X_train_original,
    #     #     patterns=None,
    #     #     w_list=w_list,
    #     #     b_list=b_list,
    #     #     feature_names=feature_cols,
    #     #     scaler=scaler,
    #     #     categorical_mappings=categorical_mappings,
    #     #     continuous_feature_indices=continuous_feature_indices,
    #     #     categorical_feature_indices=categorical_feature_indices,
    #     #     num_cols=num_plots if num_plots > 0 else 1,
    #     #     individual_learner_alpha=0.05,
    #     #     filename=f'nam_mimic_{window_hours}h_refined_groups_shapes{ints_str}',
    #     #     figsize=(max(8, num_plots * 4), 3.5),
    #     #     show=True,
    #     #     indices_to_display=final_selected,
    #     #     feature_group_labels_map=labels_map,
    #     #     # critical bits ↓
    #     #     force_same_y_limits=True,
    #     #     global_ylim=same_ylim,         # same limits for both curves and all panels
    #     #     overlay_on_single_axis=False,  # set True if you want a single y-axis per panel
    #     #     # shading
    #     #     group1_indices=gA_plot_indices, group1_color=color_gA_shade, group1_label="High Impact",
    #     #     group2_indices=gB_plot_indices, group2_color=color_gB_shade, group2_label="Nullified Effect",
    #     #     group3_indices=None
    #     # )
    # else:
    #     print("No features selected for the refined plot.")

    # # --- build avg_abs_b across learners ---
    # if isinstance(b_list, list) and len(b_list) > 0:
    #     # b_list: list of (D,) arrays
    #     b_stack = np.stack(b_list, axis=0)                     # (L, D)
    #     avg_abs_b = np.mean(np.abs(b_stack), axis=0)           # (D,)
    # elif isinstance(b_list, np.ndarray):
    #     # single (D,) array (no ensemble)
    #     avg_abs_b = np.abs(b_list)
    # else:
    #     raise RuntimeError("b_list must be a list of arrays or a (D,) numpy array.")

    # D = len(feature_cols)
    # if avg_abs_b.shape[0] != D:
    #     raise RuntimeError(f"avg_abs_b length {avg_abs_b.shape[0]} != number of features {D}")

    # # --- pick top/bottom by |b| ---
    # N_PER_GROUP = 3
    # order_asc = np.argsort(avg_abs_b)                          # ascending by |b|
    # bottom_idxs = [int(i) for i in order_asc[:N_PER_GROUP]]

    # order_desc = order_asc[::-1]                               # descending by |b|
    # top_idxs = []
    # for i in order_desc:
    #     if i not in bottom_idxs:
    #         top_idxs.append(int(i))
    #     if len(top_idxs) == N_PER_GROUP:
    #         break

    # final_selected = top_idxs + bottom_idxs
    # labels_map = {i: ("Highest |b|" if i in top_idxs else "Lowest  |b|")
    #             for i in final_selected}

    # print("\nHigh Impact (|b| high):")
    # for j in top_idxs:
    #     print(f" - {feature_cols[j]} | |b|~{avg_abs_b[j]:.4g}")

    # print("\nNullified (|b| low):")
    # for j in bottom_idxs:
    #     print(f" - {feature_cols[j]} | |b|~{avg_abs_b[j]:.4g}")

    # # --- choose global y-limits to share across ALL subplots and BOTH curves ---
    # # Prefer the limits used in the full plot if available; otherwise infer from these features
    # if 'full_plot_ax_limits' in locals() and full_plot_ax_limits.get('used_global_ylim', None) is not None:
    #     same_ylim = full_plot_ax_limits['used_global_ylim']
    # else:
    #     # Infer from precomputed mean curves if available in full_plot_ax_limits
    #     try:
    #         sel_data = [d for d in full_plot_ax_limits['feature_preds_data'] if d['feature_idx'] in final_selected]
    #         vals = []
    #         for d in sel_data:
    #             vals.append(d['mean_raw_preds'])               # \tilde f mean
    #             vals.append(d['mean_scaled_pattern_preds'])    # PGAM mean (with or without +d depending on your plot fn)
    #         vmin = min(np.min(v) for v in vals if v.size)
    #         vmax = max(np.max(v) for v in vals if v.size)
    #         m = max(abs(vmin), abs(vmax))
    #         m = 1.0 if (not np.isfinite(m) or m < 1e-12) else float(m)
    #         same_ylim = (-m, m)
    #     except Exception:
    #         same_ylim = (-1.2, 1.2)  # conservative fallback

    # # --- group shading colors ---
    # color_gA_shade = (60/255, 180/255, 75/255, 0.20)  # greenish
    # color_gB_shade = (1.0, 225/255, 25/255, 0.20)     # yellow


    # plot_shape_functions(
    #     model=model,
    #     X_train_tensor_scaled=X_train_tensor_scaled,
    #     X_train_original=X_train_original,
    #     patterns=None,
    #     w_list=w_list,
    #     b_list=b_list,
    #     d_list=d_list,
    #     feature_names=feature_cols,
    #     scaler=scaler,
    #     categorical_mappings=categorical_mappings,
    #     continuous_feature_indices=continuous_feature_indices,
    #     categorical_feature_indices=categorical_feature_indices,
    #     num_cols=len(final_selected) if final_selected else 1,
    #     individual_learner_alpha=0.025,
    #     filename=f'nam_mimic_{window_hours}h_refined_groups_top_bottom_b{ints_str}',
    #     figsize=(max(8, 4 * len(final_selected))+2, 3.5),
    #     show=False,
    #     indices_to_display=final_selected,
    #     feature_group_labels_map=labels_map,
    #     # per-panel limits; left y labels on ALL subplots; right ticks hidden
    #     force_same_y_limits=False,
    #     overlay_on_single_axis=False,
    #     group1_indices=top_idxs, group1_color=color_gA_shade, group1_label="Highest |b|",
    #     group2_indices=bottom_idxs, group2_color=color_gB_shade, group2_label="Lowest  |b|",
    #     group3_indices=None
    # )



    # ============================
    # Refined features: +b top-2, -b bottom-2, |b| near-zero 2
    # ============================

    import numpy as np

    # --- build avg_b (signed) across learners ---
    if isinstance(b_list, list) and len(b_list) > 0:
        b_stack = np.stack(b_list, axis=0)            # (L, D)
        avg_b = np.mean(b_stack, axis=0)              # (D,)
    elif isinstance(b_list, np.ndarray):
        avg_b = b_list.astype(float).copy()           # (D,)
    else:
        raise RuntimeError("b_list must be a list of (D,) arrays or a (D,) numpy array.")

    D = len(feature_cols)
    if avg_b.shape[0] != D:
        raise RuntimeError(f"avg_b length {avg_b.shape[0]} != number of features {D}")

    abs_b = np.abs(avg_b)

    # --- helpers to pick indices with no duplicates across groups ---
    def pick_top_k(idx_seq, taken, k):
        out = []
        for i in idx_seq:
            ii = int(i)
            if ii in taken: 
                continue
            out.append(ii)
            if len(out) == k:
                break
        return out

    # Candidates
    order_pos = np.argsort(-avg_b)                    # descending by signed b
    order_neg = np.argsort(avg_b)                     # ascending by signed b
    order_zero = np.argsort(abs_b)                    # ascending by |b|

    # Primary picks
    taken = set()
    pos_idxs = pick_top_k([i for i in order_pos if avg_b[i] > 0], taken, 2)
    taken.update(pos_idxs)

    neg_idxs = pick_top_k([i for i in order_neg if avg_b[i] < 0], taken, 2)
    taken.update(neg_idxs)

    zero_idxs = pick_top_k(order_zero, taken, 2)
    taken.update(zero_idxs)

    # Fallbacks to ensure exactly 2 per group
    if len(pos_idxs) < 2:
        # fill with next-largest b (could be zero/negative if necessary)
        pos_idxs += pick_top_k(order_pos, taken, 2 - len(pos_idxs)); taken.update(pos_idxs)
    if len(neg_idxs) < 2:
        neg_idxs += pick_top_k(order_neg, taken, 2 - len(neg_idxs)); taken.update(neg_idxs)
    if len(zero_idxs) < 2:
        zero_idxs += pick_top_k(order_zero, taken, 2 - len(zero_idxs)); taken.update(zero_idxs)

    final_selected = pos_idxs + neg_idxs + zero_idxs
    labels_map = {}
    for i in pos_idxs:  labels_map[i] = "Strongest positive b"
    for i in neg_idxs:  labels_map[i] = "Strongest negative b"
    for i in zero_idxs: labels_map[i] = "Smallest |b|"

    # Diagnostics
    print("\nHighest +b:")
    for j in pos_idxs:
        print(f" - {feature_cols[j]} | b~{avg_b[j]:.4g}")

    print("\nLowest -b:")
    for j in neg_idxs:
        print(f" - {feature_cols[j]} | b~{avg_b[j]:.4g}")

    print("\nSmallest |b|:")
    for j in zero_idxs:
        print(f" - {feature_cols[j]} | |b|~{abs_b[j]:.4g} (b~{avg_b[j]:.4g})")

    # --- group shading colors (3 groups) ---
    color_pos = (60/255, 180/255, 75/255, 0.20)   # green (Highest +b)
    color_neg = (1.00,  99/255,  71/255, 0.20)   # coral (Lowest -b)
    color_zero = (1.00, 225/255, 25/255, 0.20)   # yellow (Smallest |b|)

    # Prefer global limits from the full plot if you want shared limits; here we keep per-panel limits
    # per your current refined view. If you’d rather share, compute same_ylim like before and pass it.
    plot_shape_functions(
        model=model,
        X_train_tensor_scaled=X_train_tensor_scaled,
        X_train_original=X_train_original,
        patterns=None,
        w_list=w_list,
        b_list=b_list,
        # d_list=d_list,
        d_list=None,   # <<<
        feature_names=feature_cols,
        scaler=scaler,
        categorical_mappings=categorical_mappings,
        continuous_feature_indices=continuous_feature_indices,
        categorical_feature_indices=categorical_feature_indices,
        num_cols=len(final_selected) if final_selected else 1,
        individual_learner_alpha=0.025,
        filename=f'cameraready_mimic_refined_groups',
        figsize=(max(8, 4 * len(final_selected)) + 2, 3.5),
        show=True,
        indices_to_display=final_selected,
        feature_group_labels_map=labels_map,
        force_same_y_limits=False,          # per-panel symmetric limits (shared within each panel)
        overlay_on_single_axis=False,       # right axis used for red curve (ticks hidden in your fn)
        group1_indices=pos_idxs,  group1_color=color_pos,  group1_label="Strongest Positive b",
        group2_indices=neg_idxs,  group2_color=color_neg,  group2_label="Strongest Negative b",
        group3_indices=zero_idxs, group3_color=color_zero, group3_label="Smallest |b|",
    )





    print(f"\n--- Completed window_hours = {window_hours} ---")