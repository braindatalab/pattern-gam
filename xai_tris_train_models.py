import os
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as sk_metrics
from interpret.utils import measure_interactions
from nam.wrapper import NAMClassifier
from sklearn.svm import SVC
from joblib import Parallel, delayed
from filelock import FileLock
import pickle as pkl
from sklearn.metrics.pairwise import rbf_kernel # <<< Keep rbf_kernel import
from interpret.glassbox import ExplainableBoostingClassifier # <<< Import EBM

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


# === CONFIG ===
# SEEDS = [2025, 1283123, 3043040, 8238238, 123123, 321, 45743865, 987654, 1010101, 5]
SEEDS = [2025, 1283123, 3043040, 8238238, 123123]

SAVE_DIR = "./models/xai_tris"
os.makedirs(SAVE_DIR, exist_ok=True)
OUT_FILE = "training_results_xai_tris.csv"
# DEVICES = ["cuda:1", "cuda:2", "cuda:3"] 
DEVICES = ['cpu']
# DEVICES = ['mps']



# === HELPER ===
def quadratic_features(xs):
    inds_0, inds_1 = np.triu_indices(xs.shape[1], 0)
    quadratic = np.zeros((xs.shape[0], xs.shape[1] + len(inds_0)))
    for i, x in enumerate(xs):
        outer = np.outer(x, x)
        quadratic[i] = np.concatenate([x, outer[inds_0, inds_1]])
    return quadratic

def train_mlp(X_train, y_train, X_val, y_val, input_dim, hidden_dim, device,
epochs=100, batch_size=64, lr=1e-3, patience=50):
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
            return self.net(x).squeeze(1)

    model = MLP(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_loss, best_weights, wait = float('inf'), None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_train.size(0))
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_train[idx])
            loss = criterion(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val), y_val)

            if val_loss < best_loss:
                best_loss, best_weights, wait = val_loss, model.state_dict(), 0
            else:
                wait += 1
            if wait >= patience:
                break

    if best_weights:
        model.load_state_dict(best_weights)
    return model

def FAST(X_train, y_train, n_interactions, init_score=None, feature_names=None, feature_types=None):
    import time
    t0 = time.time()
    interactions = measure_interactions(
        X_train,
        y_train,
        interactions=n_interactions, 
        init_score=init_score,  # Can be a model or initial scores; set to None if not used
        feature_names = feature_names,
        feature_types = feature_types
    )
    
    pairs = []
    for (i, j), _ in interactions:
        pairs.append((i,j))
    return pairs, time.time() - t0



def train_one_seed(data_path, seed, device_str):
    device = torch.device(device_str)
    scenario = os.path.basename(data_path).replace('.pkl', '')
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    if 'translations' in scenario:
        return None
    print(f"[{device_str}] Training {scenario} | seed={seed}")

    with open(data_path, "rb") as f:
        data = pkl.load(f)

    X_train_tensor = data.x_train.float()
    y_train_tensor = data.y_train
    X_val_tensor = data.x_val.float()
    y_val_tensor = data.y_val
    X_test_tensor = data.x_test.float()
    y_test_tensor = data.y_test
    
    X_train = X_train_tensor.float().cpu().numpy()
    y_train = y_train_tensor.cpu().numpy()
    X_test = X_test_tensor.float().cpu().numpy()
    y_test = y_test_tensor.cpu().numpy()

    X_train_val_tensor = torch.cat((X_train_tensor, X_val_tensor), dim=0)
    y_train_val_tensor = torch.cat((y_train_tensor, y_val_tensor), dim=0)

    results = {}

    # === NAM with interactions ===
    d = X_train_tensor.shape[1]
    # pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    pairs = []
    if 'xor' in scenario:
        pairs, _ = FAST(X_train_tensor, y_train, n_interactions=128)

    ebm_model = ExplainableBoostingClassifier(interactions=pairs, random_state=seed)
    ebm_model.fit(X_train_val_tensor, y_train_val_tensor)
 
    results['ebm'] = ebm_model.score(X_test_tensor, y_test_tensor)

    # nam_model = NAMClassifier(
    #     num_epochs=100, num_learners=1, metric='auroc', interaction_pairs=pairs,
    #     early_stop_mode='max', device=device_str,
    #     hidden_sizes=[16,16,16],
    #     save_model_frequency=50,
    #     monitor_loss=False, n_jobs=10, random_state=seed
    # )
    # # nam_model.fit(X_train_tensor, y_train_tensor)
    # nam_model.fit(X_train_val_tensor, y_train_val_tensor)

    nam_model = NAMClassifier(
        num_epochs=100,
        num_learners=1,
        metric='auroc',
        interaction_pairs=pairs,
        hidden_sizes = [16,16,16],
        early_stop_mode='max',
        device=device_str,
        monitor_loss=False,
        # dropout=0.0,
        # feature_dropout=0.0,
        # output_reg=0.0,
        val_split=0.05, # 9000 train, 500 val -> 500/9500 roughly is 0.05.
        n_jobs=10,
        activation='relu',
        random_state=seed
    )
    nam_model.fit(X_train_val_tensor, y_train_val_tensor)

    pred50 = nam_model.predict(X_test_tensor)
    results['nam'] = sk_metrics.accuracy_score(y_test, pred50)

    torch.save({'model_state_dicts': [
    {k: v.cpu() for k, v in learner.state_dict().items()} for learner in nam_model.models
    ]}, os.path.join(SAVE_DIR, f"{scenario}_nam_{seed}.pth"))

    # === QLR ===
    Xq_train = quadratic_features(X_train_val_tensor.detach().numpy())
    Xq_test = quadratic_features(X_test)
    qlr = LogisticRegression(penalty=None, fit_intercept=False, max_iter=2000)
    qlr.fit(Xq_train, y_train_val_tensor.detach().numpy())
    results['qlr'] = qlr.score(Xq_test, y_test)
    with open(os.path.join(SAVE_DIR, f"{scenario}_qlr_{seed}.pkl"), "wb") as f:
        pkl.dump(qlr, f)

    # === MLP ===
    mlp_model = train_mlp(
        X_train_tensor.to(device), y_train_tensor.to(device),
        X_val_tensor.to(device), y_val_tensor.to(device),
        input_dim=X_train_tensor.shape[1], hidden_dim=64,
        device=device
    )
    mlp_model.eval()
    torch.save(mlp_model.state_dict(), os.path.join(SAVE_DIR, f"{scenario}_mlp_{seed}.pth"))
    with torch.no_grad():
        test_logits = mlp_model(X_test_tensor.to(device))
    test_preds = (torch.sigmoid(test_logits) > 0.5).int().cpu().numpy()
    results['mlp'] = sk_metrics.accuracy_score(y_test, test_preds)


    n_features = X_train_val_tensor.shape[1]
    variance = np.var(X_train_val_tensor.detach().numpy(), ddof=0)  # population variance (ddof=0) matches sklearn

    gamma = 1.0 / (n_features * variance)

    K = rbf_kernel(X_train_val_tensor.detach().numpy(), gamma=gamma)
    svm_clf = SVC(kernel='precomputed')
    svm_clf.fit(K, y_train_val_tensor.detach().numpy())

    K_test_vs_train = rbf_kernel(X_test, X_train_val_tensor.detach().numpy(), gamma=gamma)
    svm_preds = svm_clf.predict(K_test_vs_train)
    accuracy = sk_metrics.accuracy_score(y_test, svm_preds)

    print('KSVM ACCURACY:', accuracy)

    results['kernel_svm'] = accuracy

    with open(os.path.join(SAVE_DIR, f"{scenario}_kernel_svm_{seed}.pkl"), "wb") as f:
        pkl.dump(svm_clf, f)


    record = {"scenario": scenario, "seed": seed, **results}
    lock = FileLock("training_results_xai_tris.csv.lock")
    with lock:
        if os.path.exists(OUT_FILE):
            df_out = pd.read_csv(OUT_FILE)
        else:
            df_out = pd.DataFrame()
        df_out = pd.concat([df_out, pd.DataFrame([record])], ignore_index=True)
        df_out.to_csv(OUT_FILE, index=False)
    return None # Result is written, no need to return



# === MAIN ===
if __name__ == "__main__":
    if os.path.exists(OUT_FILE):
        df = pd.read_csv(OUT_FILE)
        done_keys = set(zip(df["scenario"], df["seed"]))
    else:
        df = pd.DataFrame()
        done_keys = set()

    # data_paths = glob.glob("data/xai_tris/*.pkl")
    data_paths = glob.glob('./data/xai_tris/*xor*dist*corr*.pkl')

    
    tasks = [(dp, seed) for dp in data_paths for seed in SEEDS
    if (os.path.basename(dp).replace('.csv', ''), seed) not in done_keys]

    results = Parallel(n_jobs=len(DEVICES), prefer="processes")(
    delayed(train_one_seed)(dp, seed, DEVICES[i % len(DEVICES)])
    for i, (dp, seed) in enumerate(tasks)
    )

# if results:
# df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
# df.to_csv(OUT_FILE, index=False)