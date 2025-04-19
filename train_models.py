# train_models.py

import os
import pickle
import json
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from models import QLR, EBM, QGAM, KernelSVM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CLASSES = {
    "qlr": QLR,
    "ebm": lambda: EBM(interactions=128),
    "qgam": lambda: QGAM(D=64, interactions=128, epochs=100),
    "kernel_svm": lambda: KernelSVM(kernel='rbf', gamma=0.1, var=1.0),
    "kernel_svm_poly": lambda: KernelSVM(kernel='poly', gamma=0.1, var=1.0)
}

SEEDS = [2025, 92392, 1234, 847239, 1832028]
DATASET_DIR = "data/xai_tris"
MODEL_SAVE_DIR = "models"
RESULT_SAVE_PATH = "results/summary.json"


def load_dataset(pkl_path):
    with open(pkl_path, "rb") as f:
        record = pickle.load(f)
    x_train = record.x_train.detach().numpy()
    y_train = record.y_train.detach().numpy().astype(int)
    x_val = record.x_val.detach().numpy()
    y_val = record.y_val.detach().numpy().astype(int)
    x_test = record.x_test.detach().numpy()
    y_test = record.y_test.detach().numpy().astype(int)
    return x_train, y_train, x_val, y_val, x_test, y_test


def run_all_models(dataset_name, dataset_path, existing_results):
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(dataset_path)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    scores_by_model = {}

    for model_name, model_constructor in MODEL_CLASSES.items():
        if dataset_name in existing_results and model_name in existing_results[dataset_name]:
            print(f"Skipping {model_name} on {dataset_name} — already in results")
            scores_by_model[model_name] = existing_results[dataset_name][model_name]["all"]
            continue

        print(f"Training {model_name} on {dataset_name}")
        scores = []
        for seed in SEEDS:
            np.random.seed(seed)
            torch.manual_seed(seed)

            model = model_constructor()

            if "qgam" in model_name:
                model.train(x_train_tensor, y_train_tensor, X_val=torch.tensor(x_val, dtype=torch.float32).to(device), y_val=torch.tensor(y_val, dtype=torch.float32).to(device))
                score = model.score(x_test_tensor, y_test_tensor)

            elif "kernel_svm" in model_name:
                model.train(x_train, y_train)
                K_test = model.var * rbf_kernel(x_test, x_train, gamma=model.gamma)
                preds = model.model.decision_function(K_test)
                preds_bin = (preds > 0).astype(int)
                score = (preds_bin == y_test).mean()
            else:
                model.train(x_train, y_train)
                score = model.score(x_test, y_test)

            scores.append(score)

            model_dir = os.path.join(MODEL_SAVE_DIR, dataset_name, model_name)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"seed{seed}.pkl")
            model.save(model_path)

        scores_by_model[model_name] = scores

    return scores_by_model


def main():
    os.makedirs("results", exist_ok=True)

    if os.path.exists(RESULT_SAVE_PATH):
        with open(RESULT_SAVE_PATH, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for fname in os.listdir(DATASET_DIR):
        if not fname.endswith(".pkl"):
            continue
        dataset_name = fname.replace(".pkl", "")
        dataset_path = os.path.join(DATASET_DIR, fname)

        scores_by_model = run_all_models(dataset_name, dataset_path, all_results)

        if dataset_name not in all_results:
            all_results[dataset_name] = {}

        for model, scores in scores_by_model.items():
            all_results[dataset_name][model] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "all": [float(s) for s in scores]
            }

            print(f"Training {model} on {dataset_name}: avg acc:", all_results[dataset_name][model]['mean'])

            with open(RESULT_SAVE_PATH, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nFinal results saved to {RESULT_SAVE_PATH}")


if __name__ == "__main__":
    main()
