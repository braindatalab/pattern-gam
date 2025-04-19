# models.py

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
import joblib
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.utils import measure_interactions
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def FAST(X_train, y_train, n_interactions, init_score=None, feature_names=None, feature_types=None):
    top_interactions = measure_interactions(
        X_train, y_train,
        interactions=n_interactions,
        init_score=init_score,
        feature_names=feature_names,
        feature_types=feature_types
    )
    return [(i, j) for (i, j), _ in top_interactions], None


def redistribute_even(pattern, interaction_pairs, D):
    input_signal = pattern[:D].copy()
    inter_signals = pattern[D:]
    for idx, (i, j) in enumerate(interaction_pairs):
        input_signal[i] += 0.5 * inter_signals[idx]
        input_signal[j] += 0.5 * inter_signals[idx]
    return input_signal


def quadratic_features(xs):
    inds_0, inds_1 = np.triu_indices(xs.shape[1], 0)
    quad_terms = np.array([np.outer(x, x)[inds_0, inds_1] for x in xs])
    return np.hstack([xs, quad_terms])


def center_kernel(K):
    N = K.shape[0]
    H = np.eye(N) - np.ones((N, N)) / N
    return H @ K @ H


class BaseModel:
    def train(self, X, y): raise NotImplementedError
    def predict(self, X): raise NotImplementedError
    def score(self, X, y): raise NotImplementedError
    def transform(self, X): raise NotImplementedError
    def explain_global(self): raise NotImplementedError
    def inverse_transform(self, *args, **kwargs): raise NotImplementedError
    def save(self, path): raise NotImplementedError
    def load(self, path): raise NotImplementedError


class QLR(BaseModel):
    def __init__(self):
        self.clf = LogisticRegression(penalty=None, fit_intercept=False, max_iter=1000, random_state=2025)
        self.pattern = None

    def train(self, X, y):
        self.X_quad = quadratic_features(X)
        self.clf.fit(self.X_quad, y)
        self.pattern = np.cov(self.X_quad.T) @ self.clf.coef_[0]

    def predict(self, X):
        return self.clf.predict_proba(quadratic_features(X))[:, 1]

    def score(self, X, y):
        return self.clf.score(quadratic_features(X), y)

    def transform(self, X):
        return quadratic_features(X)

    def explain_global(self):
        return self.pattern

    def inverse_transform(self):
        raise NotImplementedError("Not implemented for QLR")

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


class KernelSVM(BaseModel):
    def __init__(self, kernel='rbf', gamma=None, degree=3, var=1.0):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.var = var
        self.model = None
        self.pattern = None

    def train(self, X, y):
        self.X_train = X
        if self.kernel == 'rbf':
            K = self.var * rbf_kernel(X, X, gamma=self.gamma)
        elif self.kernel == 'poly':
            K = self.var * polynomial_kernel(X, X, degree=2, gamma=self.gamma)
        else:
            raise ValueError("Only RBF and Polynomial kernel supported")

        self.model = SVC(kernel='precomputed')
        self.model.fit(K, y)

        alpha = np.zeros(X.shape[0])
        alpha[self.model.support_] = self.model.dual_coef_

        H = np.eye(X.shape[0]) - np.ones((X.shape[0], X.shape[0])) / X.shape[0]
        self.pattern = (H @ K @ alpha) / X.shape[0]

    def predict(self, X):
        raise NotImplementedError("Use precomputed kernel externally")

    def score(self, X, y):
        raise NotImplementedError("Use precomputed kernel externally")

    def transform(self, X):
        raise NotImplementedError("Not supported for KernelSVM")

    def explain_global(self):
        return self.pattern

    def inverse_transform(self, sample, num_epochs=10, lamb=1e-5):
        x_preimage = sample.copy()
        preimages = []
        for _ in range(num_epochs):
            num = 0
            denom = 0
            for i in range(self.X_train.shape[0]):
                k_val = rbf_kernel(self.X_train[i].reshape(1, -1), x_preimage.reshape(1, -1), gamma=self.gamma)[0, 0]
                num += self.pattern[i] * k_val * self.X_train[i] + lamb * sample
                denom += self.pattern[i] * k_val + lamb
            x_preimage = num / denom
            preimages.append(x_preimage)
        return preimages

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


class EBM(BaseModel):
    def __init__(self, interactions=64, random_state=42):
        self.model = ExplainableBoostingClassifier(interactions=interactions, random_state=random_state)
        self._w = None
        self.pattern = None
        self.interaction_pairs = []

    def train(self, X, y):
        self.model.fit(X, y)
        Z = self.model.eval_terms(X)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Z, y)
        self._w = clf.coef_[0]
        self.pattern = np.cov(Z.T) @ self._w
        self.interaction_pairs = [tuple(g) for g in self.model.term_features_ if len(g) == 2]

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def score(self, X, y):
        return self.model.score(X, y)

    def transform(self, X):
        return self.model.eval_terms(X)

    def explain_global(self):
        return self.pattern

    def inverse_transform(self):
        input_dim = max(max(g) for g in self.model.term_features_) + 1
        input_signal = np.zeros(input_dim)
        for i, group in enumerate(self.model.term_features_):
            for feat in group:
                input_signal[feat] += self.pattern[i] / len(group)
        return input_signal

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


class ExULayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.log_weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(in_dim))
    def forward(self, x):
        w = torch.exp(self.log_weight)
        return (x - self.bias) @ w.T


class FeatureSubNet(nn.Module):
    def __init__(self, hidden_dim_main):
        super().__init__()
        self.net = nn.Sequential(
            ExULayer(1, hidden_dim_main),
            nn.SiLU(),
            nn.Linear(hidden_dim_main, 1, bias=False)
        )
    def forward(self, x): return self.net(x)


class InteractionSubNet(nn.Module):
    def __init__(self, hidden_dim_inter):
        super().__init__()
        self.net = nn.Sequential(
            ExULayer(2, hidden_dim_inter),
            nn.SiLU(),
            nn.Linear(hidden_dim_inter, 1, bias=False)
        )
    def forward(self, x): return self.net(x)


class QGAMNet(nn.Module):
    def __init__(self, D, interaction_pairs, hidden_dim_main, hidden_dim_inter):
        super().__init__()
        self.D = D
        self.main = nn.ModuleList([FeatureSubNet(hidden_dim_main) for _ in range(self.D)])
        self.inter = nn.ModuleList([InteractionSubNet(hidden_dim_inter) for _ in interaction_pairs])
        self.final = nn.Linear(self.D + len(interaction_pairs), 1)
    def forward(self, X, interaction_pairs):
        m_outs = [net(X[:, j:j+1]) for j, net in enumerate(self.main)]
        i_outs = [net(torch.cat([X[:, i:i+1], X[:, j:j+1]], dim=1))
                  for net, (i, j) in zip(self.inter, interaction_pairs)]
        Z = torch.cat(m_outs + i_outs, dim=1)
        return self.final(Z).squeeze(), Z


class QGAM(BaseModel):
    def __init__(self, D, interactions=64, hidden_dim_main=32, hidden_dim_inter=64, epochs=100):
        super().__init__()
        self.D = D
        self.interactions = interactions
        self.hidden_dim_main = hidden_dim_main
        self.hidden_dim_inter = hidden_dim_inter
        self.epochs = epochs
        self.interaction_pairs = []
        self.model = None
        self.clf = None
        self.pattern = None

    def _build_model(self, hidden_dim_main, hidden_dim_inter):
        self.model = QGAMNet(
            D=self.D,
            interaction_pairs=self.interaction_pairs,
            hidden_dim_main=hidden_dim_main,
            hidden_dim_inter=hidden_dim_inter
        )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if isinstance(self.interactions, int):
            self.interaction_pairs, _ = FAST(X_train.cpu().numpy(), y_train.cpu().numpy(), self.interactions)
        self._build_model(self.hidden_dim_main, self.hidden_dim_inter)
        self.model.to(X_train.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        best_val_loss = float('inf')
        best_state = None

        for _ in range(self.epochs):
            self.model.train()
            logits, Z = self.model(X_train, self.interaction_pairs)
            loss = F.binary_cross_entropy_with_logits(logits, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits, _ = self.model(X_val, self.interaction_pairs)
                    val_loss = F.binary_cross_entropy_with_logits(val_logits, y_val)
                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        best_state = {k: v.clone().detach() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        with torch.no_grad():
            _, Z = self.model(X_train, self.interaction_pairs)
        Z_np = Z.detach().cpu().numpy()
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Z_np, y_train.detach().cpu().numpy())
        self.clf = clf
        cov_Z = np.cov(Z_np.T)
        self.pattern = cov_Z @ clf.coef_[0]


    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(X, self.interaction_pairs)
        return torch.sigmoid(logits).detach().cpu().numpy()

    def score(self, X, y):
        Z = X
        if X.shape[1] == self.D:
            Z = self.transform(X)
        return self.clf.score(Z, y.detach().cpu().numpy())

    def transform(self, X):
        self.model.eval()
        with torch.no_grad():
            _, Z = self.model(X, self.interaction_pairs)
        return Z.detach().cpu().numpy()

    def explain_global(self):
        return self.pattern

    def inverse_transform(self):
        return redistribute_even(self.pattern, self.interaction_pairs, D=self.D)

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path, map_location=device)
