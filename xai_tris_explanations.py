"""
Parallel explanations runner:
- Preserves original calculations for pattern_gam and pattern_qlr.
- Adds NAM and QLR metrics (formerly "tilde" metrics) with parallelized 1-D LRs.
- Parallelizes per-seed work across CPU processes.
- Avoids BLAS oversubscription inside parallel regions.

Requires: joblib, threadpoolctl
    pip install joblib threadpoolctl
"""

import os
import glob
import numpy as np
import pandas as pd
import pickle as pkl
import warnings
import math
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

# Core / Torch / Sklearn
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split  # (retained; not used directly)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import roc_auc_score
from scipy.stats import zscore

# XAI libs
import shap
from captum.attr import IntegratedGradients
from pattern import PatternNet, PatternAttribution
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.utils import measure_interactions
from nam.wrapper import NAMClassifier

# Parallel / threads control
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================
# ==== CONFIGURATION ======
# =========================

# Default seeds (you can override via env SEEDS as comma-separated ints)
# SEEDS = [2025, 1283123, 3043040, 8238238, 123123]
SEEDS = [2025]

SAVE_DIR = "./models/xai_tris"
EXPLANATIONS_DIR = "./explanations/xai_tris"
os.makedirs(EXPLANATIONS_DIR, exist_ok=True)

# Device pool: e.g. ["cuda:0", "cuda:1", "cuda:2", "cuda:3"] or ["cpu"]
# You can override via env DEVICE_POOL="cuda:0,cuda:1,cuda:2,cuda:3"
DEVICE_POOL = os.environ.get("DEVICE_POOL", "cpu").split(",")

# Per-seed parallelism (# processes). If None, auto = min(64, os.cpu_count() or 8).
# Override via env N_JOBS
N_JOBS = os.environ.get("N_JOBS")
if N_JOBS is None:
    N_JOBS = min(64, os.cpu_count() or 8)
else:
    N_JOBS = int(N_JOBS)

# Per-process internal BLAS threads. Avoids MKL/OPENBLAS over-threading when using joblib.
# Override via env BLAS_THREADS
BLAS_THREADS = int(os.environ.get("BLAS_THREADS", "1"))

# Disable Captum/SHAP GPU in parallel unless you explicitly set GPU devices.
# PatternNet uses PyTorch; with CPU device it's fine to parallelize aggressively.
# If you do use GPUs, each process will pick a device from DEVICE_POOL in round-robin.

RESULTS_SAVE_PATH = os.path.join(SAVE_DIR, "explanations_cameraready_parallel.pkl")

# =========================
# ======= MODELS ==========
# =========================

class MLP(torch.nn.Module):
    def __init__(self, input_dim, n_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, n_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(n_dim // 2, n_dim // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(n_dim // 4, n_dim // 8),
            torch.nn.ReLU(),
            torch.nn.Linear(n_dim // 8, 1)
        )
    def forward(self, x):
        return self.net(x)

# =========================
# ======= HELPERS =========
# =========================

def quadratic_features(xs: np.ndarray) -> np.ndarray:
    inds_0, inds_1 = np.triu_indices(xs.shape[1], 0)
    num_quadratic_terms = len(inds_0)
    quadratic = np.zeros((xs.shape[0], xs.shape[1] + num_quadratic_terms), dtype=np.float32)
    for i, x_sample in enumerate(xs):
        outer = np.outer(x_sample, x_sample)
        interaction_terms = outer[inds_0, inds_1]
        quadratic[i] = np.concatenate([x_sample, interaction_terms])
    return quadratic

def FAST(X_train, y_train, n_interactions, init_score=None, feature_names=None, feature_types=None):
    inter = measure_interactions(
        X_train,
        y_train,
        interactions=n_interactions,
        init_score=init_score,
        feature_names=feature_names,
        feature_types=feature_types
    )
    pairs = []
    for (i, j), _ in inter:
        pairs.append((i, j))
    return pairs

def compute_patterns(
    model,
    X_in,
    y_in,
    *,
    zs_source: str = "Z_mean",
    standardize_zs: bool = True,
    device: torch.device = torch.device("cpu"),
    random_state: int = 2025
):
    if isinstance(X_in, np.ndarray):
        X_tensor = torch.tensor(X_in, dtype=torch.float32, device=device)
    elif isinstance(X_in, torch.Tensor):
        X_tensor = X_in.to(dtype=torch.float32, device=device)
    else:
        raise TypeError("X_in must be numpy array or torch tensor")

    if isinstance(y_in, np.ndarray):
        y_tensor = torch.tensor(y_in, dtype=torch.long, device=device)
    elif isinstance(y_in, torch.Tensor):
        y_tensor = y_in.to(dtype=torch.long, device=device)
    else:
        raise TypeError("y_in must be numpy array or torch tensor")

    Z_list = []
    patterns_per_learner = []

    for learner in model.models:
        learner.eval()
        learner.to(device)
        with torch.no_grad():
            _, _, me_out, int_out = learner.forward(X_tensor)
            if int_out is not None and int_out.shape[1] > 0:
                Z = torch.cat([me_out, int_out], dim=1)
            else:
                Z = me_out
            Z_np = Z.detach().cpu().numpy()
        Z_list.append(Z_np)

        cov_Z_l = np.cov(Z_np, rowvar=False)
        if not np.all(np.isfinite(cov_Z_l)):
            cov_Z_l = np.eye(Z_np.shape[1])
        lr_l = LogisticRegression(penalty=None, fit_intercept=True, max_iter=5000, random_state=random_state)
        with threadpool_limits(limits=BLAS_THREADS):
            lr_l.fit(Z_np, y_tensor.cpu().numpy().ravel())
        w_l = lr_l.coef_[0]
        yhat_l = Z_np @ w_l
        var_yhat_l = np.var(yhat_l, ddof=0) if np.var(yhat_l, ddof=0) > 0 else 1e-12
        patterns_per_learner.append((cov_Z_l @ w_l) / var_yhat_l)

    Z_all = np.stack(Z_list, axis=0)    # (L, N, D)
    Z_mean = np.mean(Z_all, axis=0)     # (N, D)

    if zs_source == "Z_mean":
        Z_used = Z_mean
    elif zs_source == "Z_all":
        L, N, D = Z_all.shape
        Z_used = Z_all.reshape(L * N, D)
    else:
        raise ValueError("zs_source must be 'Z_mean' or 'Z_all'")

    if standardize_zs:
        Z_used = zscore(Z_used, axis=0, ddof=0)
        Z_used = np.nan_to_num(Z_used, nan=0.0, posinf=0.0, neginf=0.0)

    cov_Z = np.cov(Z_used, rowvar=False)
    if not np.all(np.isfinite(cov_Z)):
        cov_Z = np.eye(Z_used.shape[1])
    lr = LogisticRegression(penalty=None, fit_intercept=True, max_iter=5000, random_state=random_state)
    with threadpool_limits(limits=BLAS_THREADS):
        lr.fit(Z_used, y_tensor.cpu().numpy().ravel())
    w_Z = lr.coef_[0]

    yhat = Z_used @ w_Z
    var_yhat = np.var(yhat, ddof=0) if np.var(yhat, ddof=0) > 0 else 1e-12

    avg_pattern = (cov_Z @ w_Z) / var_yhat
    return avg_pattern, patterns_per_learner, w_Z

def compute_discr(
    model,
    X_in,
    y_in,
    *,
    zs_source: str = "Z_mean",
    standardize_zs: bool = True,
    nonnegative: bool = True,
    device: torch.device = torch.device("cpu")
):
    if isinstance(X_in, np.ndarray):
        X_tensor = torch.tensor(X_in, dtype=torch.float32, device=device)
    else:
        X_tensor = X_in.to(dtype=torch.float32, device=device)

    if isinstance(y_in, np.ndarray):
        y_tensor = torch.tensor(y_in, dtype=torch.long, device=device)
    else:
        y_tensor = y_in.to(dtype=torch.long, device=device)

    Z_list = []
    for learner in model.models:
        learner.eval()
        learner.to(device)
        with torch.no_grad():
            _, _, me_out, int_out = learner.forward(X_tensor)
            if int_out is not None and int_out.shape[1] > 0:
                Z = torch.cat([me_out, int_out], dim=1)
            else:
                Z = me_out
            Z_np = Z.detach().cpu().numpy()
        Z_list.append(Z_np)

    Z_all = np.stack(Z_list, axis=0)   # (L, N, D)
    Z_mean = np.mean(Z_all, axis=0)    # (N, D)

    if zs_source == "Z_mean":
        Z_used = Z_mean
        y_np = y_tensor.cpu().numpy().ravel().astype(int)
    elif zs_source == "Z_all":
        L, N, D = Z_all.shape
        Z_used = Z_all.reshape(L * N, D)
        y_np = np.repeat(y_tensor.cpu().numpy().ravel().astype(int), L)
    else:
        raise ValueError("zs_source must be 'Z_mean' or 'Z_all'")

    if standardize_zs:
        Z_used = zscore(Z_used, axis=0, ddof=0)
        Z_used = np.nan_to_num(Z_used, nan=0.0, posinf=0.0, neginf=0.0)

    D = Z_used.shape[1]
    discr = np.zeros(D, dtype=float)
    for d in range(D):
        s = Z_used[:, d]
        try:
            auc = roc_auc_score(y_np, s)
        except Exception:
            auc = 0.5
        val = 2.0 * (auc - 0.5)
        if nonnegative:
            val = max(0.0, val)
        discr[d] = val
    return discr

def fpi_ksvm(pattern, X, num_epochs=10, lamb=1e-5):
    sample = np.mean(X, axis=0)
    x_preimage = sample.copy()
    preimages = []
    for _ in range(num_epochs):
        num = 0
        denom = 0
        for i in range(X.shape[0]):
            k_val = rbf_kernel(X[i].reshape(1, -1), x_preimage.reshape(1, -1))[0, 0]
            num += pattern[i] * k_val * X[i] + lamb * sample
            denom += pattern[i] * k_val + lamb
        if np.abs(denom) < 1e-9:
            if len(preimages) > 0:
                x_preimage = preimages[-1]
            else:
                x_preimage = np.zeros_like(sample)
        else:
            x_preimage = num / denom
        preimages.append(x_preimage)
    return np.mean(preimages, axis=0)

# =========================
# == NAM / QLR METRICS ====
# =========================

def extract_Z_mean(nam_model, X_tensor, device):
    X_tensor = X_tensor.to(dtype=torch.float32, device=device)
    Zs = []
    with torch.no_grad():
        for learner in nam_model.models:
            learner.eval()
            learner.to(device)
            _, _, me_out, int_out = learner.forward(X_tensor)
            Z = torch.cat([me_out, int_out], dim=1) if (int_out is not None and int_out.shape[1] > 0) else me_out
            Zs.append(Z.detach().cpu().numpy())
    Zs = np.stack(Zs, axis=0)
    return Zs.mean(axis=0)  # RAW (N, U)

def _safe_auc(scores, y):
    try:
        return 2.0 * (roc_auc_score(y, scores) - 0.5)
    except Exception:
        return 0.0

def _fit_univariate_lr(ti: np.ndarray, y: np.ndarray, rs: int) -> Tuple[float, float]:
    """Fit y ~ 1 + ti (1D LR on RAW ti); returns (b_i, d_i). Uses BLAS thread limit."""
    try:
        lr_1d = LogisticRegression(
            penalty=None, solver='lbfgs', fit_intercept=True, max_iter=10000, random_state=rs
        )
        with threadpool_limits(limits=BLAS_THREADS):
            lr_1d.fit(ti.reshape(-1, 1), y)
        return float(lr_1d.coef_[0, 0]), float(lr_1d.intercept_[0])
    except Exception:
        return 0.0, 0.0

def compute_nam_metrics_parallel(
    nam_model, X_tensor, y_tensor, device, *, random_state=1, n_jobs: int = 1
) -> Dict[str, np.ndarray]:
    """
    Parallel version of NAM metrics:
      Z_mean RAW; multivariate LR -> w
      For each unit i (parallel):
        ti = w[i] * Z[:, i]
        fit 1D LR on ti -> b_i, d_i
        SD_NAM[i] = std(ti); SDb_NAM[i] = std(b_i * ti); DISCR_NAM[i] = AUC-derived
    """
    y = y_tensor.detach().cpu().numpy().astype(int).ravel()
    Z = extract_Z_mean(nam_model, X_tensor, device)  # RAW (N, U)
    U = Z.shape[1]

    lr_all = LogisticRegression(
        penalty=None, solver='lbfgs', fit_intercept=True, max_iter=10000, random_state=random_state
    )
    with threadpool_limits(limits=BLAS_THREADS):
        lr_all.fit(Z, y)
    w = lr_all.coef_[0]  # (U,)

    # Precompute all ti
    T = [w[i] * Z[:, i] for i in range(U)]

    # Parallel univariate LRs
    jobs = (delayed(_fit_univariate_lr)(T[i], y, random_state + 17 + i) for i in range(U))
    bd = Parallel(n_jobs=n_jobs, prefer="processes")(jobs)
    b_uni = np.array([b for (b, d) in bd], dtype=float)
    d_uni = np.array([d for (b, d) in bd], dtype=float)

    SD_NAM = np.array([float(np.std(ti, ddof=0)) for ti in T], dtype=float)
    SDb_NAM = np.array([float(np.std(b_uni[i] * T[i], ddof=0)) for i in range(U)], dtype=float)
    DISCR_NAM = np.array([_safe_auc(T[i], y) for i in range(U)], dtype=float)

    return dict(
        w_NAM=w, b_uni_NAM=b_uni, d_uni_NAM=d_uni,
        SD_NAM=SD_NAM, SDb_NAM=SDb_NAM, DISCR_NAM=DISCR_NAM
    )

def compute_qlr_metrics_parallel(
    X_np, y_np, quadratic_features_fn, *, standardize_for_fit=False, fit_intercept=True,
    random_state=1, n_jobs: int = 1
) -> Dict[str, np.ndarray]:
    Z = quadratic_features_fn(X_np).astype(np.float32)  # (N, Uq) RAW
    y = np.asarray(y_np, int).ravel()
    Uq = Z.shape[1]

    if standardize_for_fit:
        mu = Z.mean(axis=0)
        sd = Z.std(axis=0, ddof=0)
        sd_safe = np.where(sd > 0, sd, 1.0)
        Z_fit = (Z - mu) / sd_safe
    else:
        sd_safe = None
        Z_fit = Z

    lr_all = LogisticRegression(
        penalty=None, solver='lbfgs', fit_intercept=fit_intercept, max_iter=10000, random_state=random_state
    )
    with threadpool_limits(limits=BLAS_THREADS):
        lr_all.fit(Z_fit, y)
    w_fit = lr_all.coef_[0]
    w_raw = (w_fit / sd_safe) if standardize_for_fit else w_fit

    # Precompute ti
    T = [w_raw[i] * Z[:, i] for i in range(Uq)]

    # Parallel univariate LRs
    jobs = (delayed(_fit_univariate_lr)(T[i], y, random_state + 37 + i) for i in range(Uq))
    bd = Parallel(n_jobs=n_jobs, prefer="processes")(jobs)
    b_uni = np.array([b for (b, d) in bd], dtype=float)
    d_uni = np.array([d for (b, d) in bd], dtype=float)

    SD_QLR = np.array([float(np.std(ti, ddof=0)) for ti in T], dtype=float)
    SDb_QLR = np.array([float(np.std(b_uni[i] * T[i], ddof=0)) for i in range(Uq)], dtype=float)
    DISCR_QLR = np.array([_safe_auc(T[i], y) for i in range(Uq)], dtype=float)

    return dict(
        w_QLR=w_raw, b_uni_QLR=b_uni, d_uni_QLR=d_uni,
        SD_QLR=SD_QLR, SDb_QLR=SDb_QLR, DISCR_QLR=DISCR_QLR
    )

# =========================
# ====== LOADERS ==========
# =========================

def load_scenario(data_path: str):
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    # Ensure float32 for speed/memory
    X_train_tensor = data.x_train.float()
    y_train_tensor = data.y_train
    X_test_tensor  = data.x_test.float()
    y_test_tensor  = data.y_test
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def load_models_for_scenario(scenario: str, seed: int, X_train_tensor: torch.Tensor, y_train_tensor: torch.Tensor, device_str: str):
    device = torch.device(device_str)
    loaded_models = {}

    # MLP
    mlp_model_path = os.path.join(SAVE_DIR, f"{scenario}_mlp_{seed}.pth")
    if os.path.exists(mlp_model_path):
        input_dim = X_train_tensor.shape[1]
        hidden_dim = 64
        mlp_model = MLP(input_dim, hidden_dim).to(device)
        try:
            mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=device))
            mlp_model.eval()
            loaded_models['mlp'] = mlp_model
        except Exception as e:
            print(f"  Error loading MLP: {e}")

    # NAM
    nam_model_path = os.path.join(SAVE_DIR, f"{scenario}_nam_{seed}.pth")
    if os.path.exists(nam_model_path):
        pairs = []
        if 'xor' in scenario:
            try:
                pairs = FAST(X_train_tensor, y_train_tensor, n_interactions=128)
            except Exception as e:
                print(f"  FAST failed: {e}")
                pairs = []
        nam_model_instance = NAMClassifier(
            num_epochs=1, interaction_pairs=pairs, device=device_str,
            hidden_sizes=[16, 16, 16], random_state=seed,
        )
        try:
            # one-epoch fit to build modules
            nam_model_instance.fit(X_train_tensor.to(device), y_train_tensor.to(device))
            saved_state = torch.load(nam_model_path, map_location=device)
            model_state_dicts = saved_state['model_state_dicts']
            if len(nam_model_instance.models) == len(model_state_dicts):
                for i, state_dict in enumerate(model_state_dicts):
                    nam_model_instance.models[i].load_state_dict(state_dict)
                    nam_model_instance.models[i].eval()
                    nam_model_instance.models[i].to(device)
                nam_model_instance.fitted = True
                loaded_models['nam'] = nam_model_instance
            else:
                print("  NAM loading error: mismatch in learners.")
        except Exception as e:
            print(f"  Error loading NAM: {e}")

    # QLR
    qlr_model_path = os.path.join(SAVE_DIR, f"{scenario}_qlr_{seed}.pkl")
    if os.path.exists(qlr_model_path):
        try:
            with open(qlr_model_path, "rb") as f:
                loaded_models['qlr'] = pkl.load(f)
        except Exception as e:
            print(f"  Error loading QLR: {e}")

    # EBM
    ebm_model_path = os.path.join(SAVE_DIR, f"{scenario}_ebm_{seed}.pkl")
    if os.path.exists(ebm_model_path):
        try:
            with open(ebm_model_path, "rb") as f:
                loaded_models['ebm'] = pkl.load(f)
        except Exception as e:
            print(f"  Error loading EBM: {e}")

    return loaded_models

# =========================
# == PER-SEED WORK UNIT ===
# =========================

def per_seed_work(
    data_path: str,
    seed: int,
    device_str: str,
    seed_index: int,
    per_unit_n_jobs: int
) -> Tuple[str, Dict[str, List[Any]]]:
    """
    Performs all computations for one (scenario, seed). Returns (scenario_name, local_results_dict)
    The local_results_dict contains lists (single entry appended) for each metric key.
    """
    # Select device for this worker (round-robin from DEVICE_POOL)
    dev = DEVICE_POOL[seed_index % len(DEVICE_POOL)]
    device_str = dev.strip()
    device = torch.device(device_str)

    scenario = os.path.basename(data_path).replace('.pkl', '')
    if 'translations' in scenario:
        return scenario, {}

    np.random.seed(seed)
    torch.manual_seed(seed)

    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_scenario(data_path)

    # Convert to numpy as needed
    X_train_np = X_train_tensor.cpu().numpy().astype(np.float32)
    y_train_np = y_train_tensor.cpu().numpy()
    X_test_np  = X_test_tensor.cpu().numpy().astype(np.float32)
    y_test_np  = y_test_tensor.cpu().numpy()

    models = load_models_for_scenario(scenario, seed, X_train_tensor, y_train_tensor, device_str)

    # Initialize local result containers aligned with your global structure
    local = {}
    method_keys = ['pattern_gam', 'pattern_qlr', 'kernel_svm', 'ebm',
                   'shap', 'ig', 'nam', 'pattern_net', 'pattern_attribution']
    for k in method_keys:
        local[k] = []

    # New metric containers
    for k in ['w_NAM', 'b_uni_NAM', 'd_uni_NAM', 'SD_NAM', 'SDb_NAM', 'DISCR_NAM',
              'w_QLR', 'b_uni_QLR', 'd_uni_QLR', 'SD_QLR', 'SDb_QLR', 'DISCR_QLR',
              'discr']:
        local[k] = []

    # ====== MLP-based explanations ======
    if 'mlp' in models:
        mlp_model = models['mlp']
        try:
            # SHAP
            background_data_shap = shap.sample(X_train_tensor.to(device), min(100, X_train_tensor.shape[0]))
            explainer_shap = shap.DeepExplainer(mlp_model, background_data_shap)
            shap_values = explainer_shap.shap_values(X_test_tensor.to(device))
            if isinstance(shap_values, list):
                shap_values_np = [s.detach().cpu().numpy() if torch.is_tensor(s) else s for s in shap_values]
                if len(shap_values_np) == 1:
                    shap_values_np = shap_values_np[0]
            elif torch.is_tensor(shap_values):
                shap_values_np = shap_values.detach().cpu().numpy()
            else:
                shap_values_np = shap_values
            local['shap'].append(shap_values_np)
        except Exception as e:
            print(f"[{scenario}|seed={seed}] SHAP error: {e}")

        try:
            # IG
            ig = IntegratedGradients(mlp_model)
            baselines_ig = torch.zeros_like(X_test_tensor[0:1]).to(device)
            ig_attr = ig.attribute(X_test_tensor.to(device), baselines=baselines_ig, target=None)
            local['ig'].append(ig_attr.detach().cpu().numpy())
        except Exception as e:
            print(f"[{scenario}|seed={seed}] IG error: {e}")

        # PatternNet / PatternAttribution
        pattern_net_instance = None
        try:
            train_dataset = TensorDataset(X_train_tensor.to(device), y_train_tensor.to(device))
            pattern_batch_size = min(512, X_train_tensor.shape[0])
            train_dataloader = DataLoader(train_dataset, batch_size=pattern_batch_size, shuffle=False)
            pattern_net_instance = PatternNet(mlp_model)
            pattern_net_instance.train(train_dataloader, input_key=0)
            pn_attr, _ = pattern_net_instance.attribute(X_test_tensor.to(device))
            local['pattern_net'].append(pn_attr.detach().cpu().numpy())
        except Exception as e:
            print(f"[{scenario}|seed={seed}] PatternNet error: {e}")

        try:
            if pattern_net_instance and pattern_net_instance.patterns:
                pattern_attribution_instance = PatternAttribution(mlp_model)
                pattern_attribution_instance.patterns = pattern_net_instance.patterns
                pa_attr = pattern_attribution_instance.attribute(X_test_tensor.to(device), target=None)
                local['pattern_attribution'].append(pa_attr.detach().cpu().numpy())
        except Exception as e:
            print(f"[{scenario}|seed={seed}] PatternAttribution error: {e}")
    else:
        pass  # MLP missing is fine

    # ====== PatternGAM (NAM) + DISCR (standardized Z) ======
    if 'nam' in models:
        try:
            pattern_gam, patterns_per_learner, wZ = compute_patterns(
                models['nam'],
                X_train_tensor,   # tensor
                y_train_tensor,
                zs_source="Z_mean",
                standardize_zs=True,
                device=device,
                random_state=seed
            )
            local['pattern_gam'].append(pattern_gam)
            local['nam'].append(wZ)

            discr = compute_discr(
                models['nam'],
                X_train_tensor,
                y_train_tensor,
                zs_source="Z_mean",
                standardize_zs=True,
                nonnegative=True,
                device=device
            )
            local['discr'].append(discr)
        except Exception as e:
            print(f"[{scenario}|seed={seed}] PatternGAM/DISCR error: {e}")
    else:
        pass

    # ====== PatternQLR (standardized Z for fit, as in your version) ======
    try:
        Z_qlr = quadratic_features(X_train_np)  # RAW
        Z_used = zscore(Z_qlr, axis=0, ddof=0)
        Z_used = np.nan_to_num(Z_used, nan=0.0, posinf=0.0, neginf=0.0)
        lr = LogisticRegression(penalty=None, fit_intercept=True, max_iter=5000, random_state=seed)
        with threadpool_limits(limits=BLAS_THREADS):
            lr.fit(Z_used, y_train_np.ravel().astype(int))
        wZ = lr.coef_[0]

        cov_Z = np.cov(Z_used, rowvar=False)
        if not np.all(np.isfinite(cov_Z)):
            cov_Z = np.eye(Z_used.shape[1])

        yhat = Z_used @ wZ
        var_yhat = np.var(yhat, ddof=0)
        if var_yhat <= 0:
            var_yhat = 1e-12

        pattern_qlr = (cov_Z @ wZ) / var_yhat
        local['pattern_qlr'].append(pattern_qlr)
    except Exception as e:
        print(f"[{scenario}|seed={seed}] PatternQLR error: {e}")

    # ====== Kernel SVM pattern as in your script ======
    try:
        X = X_train_np
        K = rbf_kernel(X)  # initial (won't be used for fit if gamma differs later)
        svm_clf = SVC(kernel='precomputed')
        svm_clf.fit(K, y_train_np)

        # Recompute K using the resolved gamma_ from the model (to match your steps)
        gamma_ = svm_clf._gamma
        K = rbf_kernel(X_train_np, X_train_np, gamma=gamma_)

        alpha = np.zeros(X_train_np.shape[0])
        support_indices = svm_clf.support_
        if np.max(support_indices) >= len(alpha):
            raise IndexError("Support vector indices out of bounds.")
        if svm_clf.dual_coef_.shape[0] != 1:
            raise ValueError(f"Unexpected shape for dual_coef_: {svm_clf.dual_coef_.shape}")
        alpha[support_indices] = svm_clf.dual_coef_[0]

        n_samples = X_train_np.shape[0]
        H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
        pattern_ksvm_feat = (H @ K @ alpha) / n_samples
        pattern_ksvm = fpi_ksvm(pattern_ksvm_feat, X_train_np, num_epochs=10, lamb=1e-5)
        if pattern_ksvm.shape == (X_train_np.shape[1],):
            local['kernel_svm'].append(pattern_ksvm)
    except Exception as e:
        print(f"[{scenario}|seed={seed}] Kernel SVM error: {e}")

    # ====== EBM global ======
    if 'ebm' in models:
        try:
            ebm_model = models['ebm']
            ebm_global = ebm_model.explain_global()
            ebm_scores = ebm_global.data()['scores']
            local['ebm'].append(ebm_scores)
        except Exception as e:
            print(f"[{scenario}|seed={seed}] EBM error: {e}")
    else:
        try:
            pairs = []
            if 'xor' in scenario:
                pairs = FAST(X_train_tensor.to(device), y_train_tensor.to(device), n_interactions=128)
            ebm_model = ExplainableBoostingClassifier(interactions=pairs, random_state=seed)
            ebm_model.fit(X_train_tensor.to(device), y_train_tensor.to(device))
            ebm_global = ebm_model.explain_global()
            ebm_scores = ebm_global.data()['scores']
            local['ebm'].append(ebm_scores)
        except Exception as e:
            print(f"[{scenario}|seed={seed}] EBM (fit) error: {e}")

    # ====== NEW: NAM metrics (RAW Z_mean) parallelized ======
    if 'nam' in models:
        try:
            nam_metrics = compute_nam_metrics_parallel(
                models['nam'],
                X_train_tensor, y_train_tensor,
                device,
                random_state=seed,
                n_jobs=per_unit_n_jobs
            )
            for k, v in nam_metrics.items():
                local[k].append(v)
        except Exception as e:
            print(f"[{scenario}|seed={seed}] NAM metrics error: {e}")

    # ====== NEW: QLR metrics (RAW quadratic) parallelized ======
    try:
        qlr_metrics = compute_qlr_metrics_parallel(
            X_train_np, y_train_np,
            quadratic_features_fn=quadratic_features,
            standardize_for_fit=False, fit_intercept=True,
            random_state=seed,
            n_jobs=per_unit_n_jobs
        )
        for k, v in qlr_metrics.items():
            local[k].append(v)
    except Exception as e:
        print(f"[{scenario}|seed={seed}] QLR metrics error: {e}")

    return scenario, local

# =========================
# ========= MAIN ==========
# =========================

def merge_results(global_res: Dict[str, Dict[str, List[Any]]],
                  scenario: str,
                  local: Dict[str, List[Any]]):
    if scenario == "" or not local:
        return
    if scenario not in global_res:
        global_res[scenario] = {}
    for k, v_list in local.items():
        if k not in global_res[scenario]:
            global_res[scenario][k] = []
        # v_list is already a list with one element for this seed
        global_res[scenario][k].extend(v_list)

def main():
    data_paths = glob.glob('./data/xai_tris/*_0.pkl')
    if not data_paths:
        print("No data files found matching ./data/xai_tris/*.pkl")
        return

    # Optionally override seeds from env (comma-separated)
    if "SEEDS" in os.environ:
        seeds_env = os.environ["SEEDS"].strip()
        if seeds_env:
            seeds = [int(s) for s in seeds_env.split(",")]
        else:
            seeds = SEEDS
    else:
        seeds = SEEDS

    # Per-unit parallel: default to min(32, cpu_count//max(1, len(seeds_parallel))) inside workers.
    # To keep it simple, expose env PER_UNIT_JOBS; else derive later.
    per_unit_jobs_env = os.environ.get("PER_UNIT_JOBS")
    per_unit_jobs_default = max(1, (os.cpu_count() or 8) // max(1, min(N_JOBS, len(seeds))))
    per_unit_n_jobs = int(per_unit_jobs_env) if per_unit_jobs_env else min(32, per_unit_jobs_default)

    print(f"N_JOBS (per-seed processes): {N_JOBS}")
    print(f"BLAS_THREADS per process:   {BLAS_THREADS}")
    print(f"PER_UNIT_JOBS (1D LRs):     {per_unit_n_jobs}")
    print(f"DEVICE_POOL:                 {DEVICE_POOL}")

    results: Dict[str, Dict[str, List[Any]]] = {}

    # We parallelize per-seed for each scenario. To keep RAM bounded, do scenarios sequentially.
    for data_path in data_paths:
        scenario = os.path.basename(data_path).replace('.pkl', '')
        if "translations" in scenario:
            continue
        print(f"\n=== Scenario: {scenario} ===")

        # Prepare per-seed jobs
        jobs = (delayed(per_seed_work)(
                    data_path=data_path,
                    seed=seed,
                    device_str="cpu",          # overridden by round-robin in worker
                    seed_index=idx,
                    per_unit_n_jobs=per_unit_n_jobs
                )
                for idx, seed in enumerate(seeds))

        # Run in parallel
        seed_results = Parallel(n_jobs=N_JOBS, prefer="processes")(jobs)

        # Merge back
        for scenario_name, local in seed_results:
            merge_results(results, scenario_name, local)

    # Save once at end
    os.makedirs(SAVE_DIR, exist_ok=True)
    try:
        with open(RESULTS_SAVE_PATH, "wb") as f:
            pkl.dump(results, f)
        print(f"\nSaved results to {RESULTS_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()