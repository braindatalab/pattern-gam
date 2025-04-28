from typing import List, Dict, Tuple

import numpy as np
import scipy
from numpy.random import Generator, default_rng
from scipy.stats import multivariate_normal, norm
from scipy.stats._multivariate import _eigvalsh_to_eps
from sklearn.preprocessing import MinMaxScaler
import random 
import os 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

def set_random_states(seed: int) -> Generator:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return default_rng(seed)


DEFAULT_NUM_OF_TARGETS = 1
NUM_TARGET_DEPENDENT_LATENTS_SUPPRESSORS = 0


def random_orthogonal_matrix(
    size: int, rng: Generator, tolerance: float = 1e-6
) -> np.ndarray:
    """
    % M = RANDORTHMAT(n)
    % generates a random n x n orthogonal real matrix.
    %
    % M = RANDORTHMAT(n,tol)
    % explicitly specifies a thresh value that measures linear dependence
    % of a newly formed column with the existing columns. Defaults to 1e-6.
    %
    % In this version the generated matrix distribution *is* uniform over the manifold
    % O(n) w.r.t. the induced R^(n^2) Lebesgue measure, at a slight computational
    % overhead (randn + normalization, as opposed to rand ).
    %
    % (c) Ofek Shilon , 2006.
    """

    M = np.zeros(shape=(size, size))
    vi = rng.normal(size=(size,))

    M[:, 0] = vi / np.linalg.norm(vi)
    for k in range(1, size):
        norm_vi = 0
        c = 0
        while norm_vi < tolerance:
            c += 1
            vi = rng.normal(size=(size,))
            vi = vi - np.dot(M[:, 0 : (k - 1)], np.dot(M[:, 0 : (k - 1)].T, vi))
            norm_vi = np.linalg.norm(vi)

        M[:, k] = vi / norm_vi
    return M


def is_semidefinite_matrix(eigen_values: np.ndarray) -> bool:
    is_semidefinite = True
    eps = _eigvalsh_to_eps(eigen_values)
    if np.min(eigen_values) < -eps:
        # The matrix is not positive semidefinite')
        is_semidefinite = False
    return is_semidefinite


def is_singular_matrix(eigen_values: np.ndarray) -> bool:
    is_singular = False
    eps = _eigvalsh_to_eps(eigen_values)
    d = eigen_values[eigen_values > eps]
    if len(d) < len(eigen_values):
        is_singular = True
    return is_singular


def random_covariance_matrix(size: int, rng: Generator) -> np.ndarray:
    is_singular = True
    C = np.zeros(shape=(size, size))
    while is_singular:
        V = random_orthogonal_matrix(size=size, rng=rng)
        diagonal_elements = 0.1 + 0.99 * rng.uniform(size=size)
        D = np.diag(diagonal_elements)
        C = np.dot(V, np.dot(D, V.T))
        s, _ = scipy.linalg.eigh(C, lower=True, check_finite=True)
        if not is_semidefinite_matrix(eigen_values=s) or is_singular_matrix(
            eigen_values=s
        ):
            continue
        else:
            is_singular = False
    return C


def sample_covariance_and_mean(
    size: int, rng: Generator
) -> Tuple[np.ndarray, np.ndarray]:
    if 1 < size:
        cov = random_covariance_matrix(size=size, rng=rng)
    else:
        cov = rng.uniform(low=1e-2, high=0.1, size=size)
    mean = rng.uniform(size=size, low=0, high=1)  # , low=-1, high=1)
    return mean, cov


def sample_from_mixture_of_multivariate_gaussians(
    params: List, latents: np.ndarray
) -> np.ndarray:
    x = np.zeros_like(latents[:, 0])
    for k, (mean, cov) in enumerate(params):
        x += (1) ** k * multivariate_normal.pdf(x=latents, mean=mean, cov=cov)
    return x


def sample_from_mixture_of_gaussians(params: List, latents: np.ndarray) -> np.ndarray:
    x = np.zeros((latents.shape[0], 1))
    for k, (mean, variance) in enumerate(params):
        x += (1) ** k * norm.pdf(x=latents, loc=mean, scale=variance)
    return x.flatten()


def sample_gaussian_mixture_params(
    num_gaussians: int, size_covariance: int, rng: Generator
) -> List:
    output = list()
    for k in range(num_gaussians):
        output += [sample_covariance_and_mean(size=size_covariance, rng=rng)]
    return output


def generate_gaussian_mixture_data(params: List, latents: List) -> List[np.ndarray]:
    data = list()
    for k, (p, l) in enumerate(zip(params, latents)):
        if 1 < l.shape[1]:
            data += [sample_from_mixture_of_multivariate_gaussians(params=p, latents=l)]
        else:
            data += [sample_from_mixture_of_gaussians(params=p, latents=l)]
    return data


def generate_gaussian_mixture_params(
    num: int, num_gaussians: int, size_covariance: int, rng: Generator
) -> List:
    out = list()
    for k in range(num):
        out += [
            sample_gaussian_mixture_params(
                num_gaussians=num_gaussians, size_covariance=size_covariance, rng=rng
            )
        ]
    return out


def remove_linear_dependency_between_target_and_input(
    x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    y_aux, z = np.hstack([y, np.ones_like(y)]), np.zeros_like(x)
    for k in range(x.shape[1]):
        z[:, k] = x[:, k] - 1 * np.dot(
            y_aux, np.linalg.lstsq(y_aux, x[:, k], rcond=None)[0]
        )
    return z


def linear_transform(
    x: np.ndarray, new_x_min: float = 0, new_x_max: float = 1
) -> np.ndarray:
    a = new_x_min - new_x_max
    b = np.min(x) - np.max(x)
    return a / b * (x - np.max(x)) + new_x_max


def make_classification_problem(
    x: np.ndarray, y: np.ndarray, threshold: float
) -> Tuple:
    """
    We use the lower `threshold` and upper `threshold` of the target to divide the target
    into two classes.
    """
    class_1 = y <= np.quantile(y, q=threshold)
    class_2 = y > np.quantile(y, q=1-threshold)
    class_filter = (class_1 | class_2).flatten()
    x_new = x[class_filter, :]
    y_categorical = np.ones_like(y)
    y_categorical[class_2] = 0
    y_categorical = y_categorical[class_filter]
    return x_new, y_categorical


class GaussianMixtureLatentModel:
    def __init__(
        self,
        num_samples: int,
        num_target_dependent_latents: int,
        num_gaussians: int,
        num_latents_in_target: int,
        num_latents: int,
        num_features: int,
        rng: Generator,
    ):
        self.num_samples = num_samples
        self.num_latents = num_latents
        self.num_target_dependent_latents = num_target_dependent_latents
        self.num_gaussians = num_gaussians
        self.num_features = num_features
        self.num_latents_in_target = num_latents_in_target
        self.num_target_independent_latents = (
            self.num_latents - self.num_target_dependent_latents
        )
        self.rng = rng
        self.artifact_names = ["features", "params", "latents", "indices_of_latents"]
        self.artifacts = {k: list for k in self.artifact_names}

    def _choose_indices_of_latents(self, total_num_latents: int) -> List[Tuple]:
        indices_of_latents = list()
        for k in range(self.num_features):
            indices_of_latents += [
                self._choose_indices_of_latents_per_feature(
                    total_num_latents=total_num_latents
                )
            ]
        return indices_of_latents

    def _choose_indices_of_latents_per_feature(self, total_num_latents: int) -> Dict:
        """We implicitly assume the first 'num_target_latents' of the latent variables
        are the latent variables which are used to generate the target.
        """
        indices_target_independent_latents = np.arange(
            self.num_latents_in_target, total_num_latents
        )
        indices_target_dependent_latents = np.arange(self.num_latents_in_target)
        chosen_indices_target_dependent = self.rng.choice(
            indices_target_dependent_latents,
            size=self.num_target_dependent_latents,
            replace=True,
            shuffle=True,
        )
        chosen_indices_target_independent = self.rng.choice(
            indices_target_independent_latents,
            size=self.num_target_independent_latents,
            replace=True,
            shuffle=True,
        )
        return {
            "dependent": chosen_indices_target_dependent,
            "independent": chosen_indices_target_independent,
        }

    def _choose_latents(self, indices: List, latents: np.ndarray) -> List[np.ndarray]:
        list_of_latents = list()
        for idx in indices:
            latent_filter = np.hstack([idx["dependent"], idx["independent"]])
            list_of_latents += [latents[:, latent_filter]]
        return list_of_latents

    def _generate_features(self, latents: np.ndarray) -> np.ndarray:
        indices_of_latents = self._choose_indices_of_latents(
            total_num_latents=latents.shape[1]
        )
        chosen_latents = self._choose_latents(
            indices=indices_of_latents, latents=latents
        )

        params_of_features = generate_gaussian_mixture_params(
            num=self.num_features,
            num_gaussians=self.num_gaussians,
            size_covariance=self.num_latents,
            rng=self.rng,
        )

        features = generate_gaussian_mixture_data(
            params=params_of_features, latents=chosen_latents
        )

        self.artifacts["features"] = features
        self.artifacts["params"] = params_of_features
        self.artifacts["indices_of_latents"] = indices_of_latents

        return np.vstack(features).T if features else np.empty((latents.shape[0], 0))

    def _add_noise(self, x):
        covariance = 0.001 * np.eye(x.shape[1])
        noise = self.rng.multivariate_normal(
            mean=np.zeros(
                x.shape[1],
            ),
            cov=covariance,
            size=self.num_samples,
        )
        x += noise
        return x

    def __call__(self, latents: np.ndarray) -> Tuple:
        x = self._generate_features(latents=latents)
        return x, self.artifacts


class GaussianMixtureTargetLatentModel:
    def __init__(
        self,
        num_samples: int,
        num_gaussians: int,
        num_latents: int,
        num_targets: int,
        rng: Generator,
    ):
        self.num_samples = num_samples
        self.num_latents = num_latents
        self.num_gaussians = num_gaussians
        self.num_targets = num_targets
        self.rng = rng
        self.artifacts_names = ["features", "params", "latents", "indices_of_latents"]
        self.artifacts = {k: list for k in self.artifacts_names}

    def _generate_targets(self, latents: np.ndarray) -> np.ndarray:
        params_of_features = generate_gaussian_mixture_params(
            num=self.num_targets,
            num_gaussians=self.num_gaussians,
            size_covariance=self.num_latents,
            rng=self.rng,
        )
        target_latents = [
            latents[:, : self.num_latents] for _ in range(self.num_targets)
        ]
        features = generate_gaussian_mixture_data(
            params=params_of_features, latents=target_latents
        )

        self.artifacts["features"] = features
        self.artifacts["params"] = params_of_features
        self.artifacts["latents"] = target_latents
        latent_indices = list()
        for _ in range(self.num_targets):
            latent_indices += [
                dict(dependent=np.arange(self.num_latents), independent=np.array([]))
            ]
        self.artifacts["indices_of_latents"] = latent_indices

        return np.vstack(features).T

    def _add_noise(self, x):
        covariance = 0.001 * np.eye(x.shape[1])
        noise = self.rng.multivariate_normal(
            mean=np.zeros(
                x.shape[1],
            ),
            cov=covariance,
            size=self.num_samples,
        )
        x += noise
        return x

    def __call__(self, latents: np.ndarray) -> Tuple:
        x = self._generate_targets(latents=latents)
        return x, self.artifacts


def make_mixture_of_gaussians_latent_model(
    num_samples: int,
    num_features: int,
    num_informative_features: int,
    num_suppressor_features: int,
    num_uninformative_features: int,
    num_gaussians: int,
    num_total_latents: int,
    num_latents_for_informative_features: int,
    num_latents_for_suppressor_features: int,
    num_target_dependent_latents_for_informative_features: int,
    num_latents_target: int,
    seed: int,
    classification_threshold: float,
    classification_problem: bool = True,
    return_artifacts: bool = False,
) -> Tuple:
    """
    For a given list of features {f_1, f_2, ..., f_d},
    we assume that the first `num_informative_features` are the informative features,
    the next `num_suppressor_features` are the suppressor features and the last
    `num_uninformative_features` are the uninformative features.
    From this assumption, we determine the ground truth explanation as follows:
    ground_truth_explanations[:, :num_informative_features] = 1
    ground_truth_explanations[:, num_informative_features:num_informative_features + num_suppressor_features] = 0

    :return:
    """
    if (
        num_features
        != num_informative_features
        + num_suppressor_features
        + num_uninformative_features
    ):
        raise ValueError(
            f"Number of features must coincide with the "
            f"sum of the number of informative and suppressor "
            f"features. {num_features}:"
            f"{num_informative_features + num_suppressor_features + num_uninformative_features}"
        )

    if (
        num_latents_for_suppressor_features
        > num_total_latents - num_latents_for_informative_features
    ):
        raise ValueError(
            f"Number of latent variables used to create"
            f"suppressor features must be smaller or equal"
            f"num_total_latents - num_latents_for_informative_features."
        )

    rng = set_random_states(seed=seed)
    latent_variables = rng.uniform(size=(num_samples, num_total_latents))

    x_informative, artifacts_informative = GaussianMixtureLatentModel(
        num_samples=num_samples,
        num_target_dependent_latents=num_target_dependent_latents_for_informative_features,
        num_gaussians=num_gaussians,
        num_latents_in_target=num_latents_target,
        num_latents=num_latents_for_informative_features,
        num_features=num_informative_features,
        rng=rng,
    )(latent_variables)

    x_suppressor, artifacts_suppressor = GaussianMixtureLatentModel(
        num_samples=num_samples,
        num_target_dependent_latents=NUM_TARGET_DEPENDENT_LATENTS_SUPPRESSORS,
        num_gaussians=num_gaussians,
        num_latents_in_target=num_latents_target,
        num_latents=num_latents_for_suppressor_features,
        num_features=num_suppressor_features,
        rng=rng,
    )(latent_variables)

    x_uninformative = rng.uniform(size=(num_samples, num_uninformative_features))

    target, artifacts_target = GaussianMixtureTargetLatentModel(
        num_samples=num_samples,
        num_gaussians=num_gaussians,
        num_latents=num_latents_target,
        num_targets=DEFAULT_NUM_OF_TARGETS,
        rng=rng,
    )(latent_variables)

    artifacts = dict(
        informative_features=artifacts_informative,
        suppressor_features=artifacts_suppressor,
        target=artifacts_target,
    )

    x = np.hstack([x_informative, x_suppressor, x_uninformative])
    z = remove_linear_dependency_between_target_and_input(x=x, y=target)

    if classification_problem is True and classification_threshold is not None:
        x, y = make_classification_problem(
            x=z, y=target, threshold=classification_threshold
        )
    else:
        x, y = z, target

    ground_truth_explanations = np.zeros(shape=(num_samples, num_features))
    ground_truth_explanations[:, :num_informative_features] = 1

    return (
        (x, y, ground_truth_explanations, latent_variables, artifacts)
        if return_artifacts
        else (x, y, ground_truth_explanations)
    )

params = {
    "num_samples": 50000,
    "num_features": 10,
    "num_informative_features": 3,
    "num_suppressor_features": 3,
    "num_uninformative_features": 4,
    "num_gaussians": 3,
    "num_total_latents": 4,
    "num_latents_for_informative_features": 2,
    "num_latents_for_suppressor_features": 2,
    "num_target_dependent_latents_for_informative_features": 1,
    "num_latents_target": 2,
    "classification_threshold": 0.5, # threshold < 0.5 pushes classes further apart; threshold > 0.5 imbalances towards class 0
    "classification_problem": True,
    # "seed": [2025, 92392, 1273127, 12345, 4]
    "seed": 2025
}

x, y, ground_truth_explanations = make_mixture_of_gaussians_latent_model(**params)

SAVE_DIR = './data/nls'
os.makedirs(SAVE_DIR, exist_ok=True)
seeds = [2025]
# of the form [informative,suppressor,uninformative]
# feature_configs = [[3,0,7],[3,3,4],[3,7,0],[3,0,17],[3,8,9],[3,17,0]]

feature_configs = [[3,3,4]]
thresholds = [0.2,0.3,0.4,0.5]

for i,seed in enumerate(seeds):
    for feature_config in feature_configs:
        for threshold in thresholds:
            [inf,sup,uninf] = feature_config
            rng = set_random_states(seed=seed)
            params['seed'] = seed
            params["num_features"] =  int(np.sum(feature_config))
            params["num_informative_features"] = inf
            params["num_suppressor_features"] = sup
            params["num_uninformative_features"] = uninf
            params["classification_threshold"] = threshold
            x, y, ground_truth_explanations = make_mixture_of_gaussians_latent_model(**params)

            # Convert to DataFrame
            feature_names = [f"feature_{i}" for i in range(x.shape[1])]
            df = pd.DataFrame(x, columns=feature_names)
            df["label"] = y

            # Save to CSV
            df.to_csv(f"{SAVE_DIR}/nls_{inf}i{sup}s{uninf}u_{i}_{threshold}.csv", index=False)

            # Define feature groups
            num_features = x.shape[1]
            num_informative = params["num_informative_features"]
            num_suppressor = params["num_suppressor_features"]

            informative_features = {f"feature_{i}" for i in range(num_informative)}
            suppressor_features = {
                f"feature_{i}" for i in range(num_informative, num_informative + num_suppressor)
            }

            # Original hex colors
            informative_color_hex = "#e8f5e9"
            suppressor_color_hex = "#f3e5f5"


            alpha = 0.5

            informative_color_rgba = to_rgba(informative_color_hex, alpha=alpha)
            suppressor_color_rgba = to_rgba(suppressor_color_hex, alpha=alpha)

            default_color = "white"

            # Build DataFrame
            feature_names = [f"feature_{i}" for i in range(num_features)]
            df = pd.DataFrame(x, columns=feature_names)
            df["label"] = y

            # Create pairplot with more visible markers
            g = sns.pairplot(
                df,
                vars=feature_names,
                hue="label",
                palette="coolwarm",
                plot_kws={"alpha": 0.8, "s": 20},  # bigger, more visible points
                corner=True,
            )

            # sns.set_theme(font_scale=2)

            # Color background based on feature roles
            for i, row_var in enumerate(g.x_vars):
                for j, col_var in enumerate(g.y_vars[:i]):
                    ax = g.axes[i, j]
                    row_info = row_var in informative_features
                    row_supp = row_var in suppressor_features
                    col_info = col_var in informative_features
                    col_supp = col_var in suppressor_features

                    if row_info or col_info:
                        ax.set_facecolor(informative_color_rgba)
                    elif row_supp or col_supp:
                        ax.set_facecolor(suppressor_color_rgba)
                    else:
                        ax.set_facecolor(default_color)

            # Custom feature-type legend
            feature_legend_patches = [
                Patch(facecolor=informative_color_hex, edgecolor="black", label="Informative"),
                Patch(facecolor=suppressor_color_hex, edgecolor="black", label="Suppressor"),
                Patch(facecolor=default_color, edgecolor="black", label="Uninformative"),
            ]


            # Remove default legend
            g._legend.remove()

            # Manually extract unique hue values and corresponding colors
            hue_order = sorted(df["label"].unique())
            palette = sns.color_palette("coolwarm", len(hue_order))
            handles = [
                Patch(facecolor=palette[i], label=str(hue_order[i]))
                for i in range(len(hue_order))
            ]

            # Create clean 2-column class label legend
            class_legend = g.fig.legend(
                handles=handles,
                title="Class Label",
                loc="upper right",
                bbox_to_anchor=(0.5, 0.9),
                ncol=2,
                fontsize=18,
                title_fontsize=18,
                frameon=True
            )


            # Add feature type legend at top
            g.fig.legend(
                handles=feature_legend_patches,
                loc="upper center",
                bbox_to_anchor=(0.44, 0.86),
                ncol=3,
                frameon=True,
                fontsize=18,
                title="Feature Type",
                title_fontsize=18
            )

            # Final touches
            # g.fig.suptitle("Pairwise Feature Plot\n(Shaded Background = Feature Type)", y=1.18, fontsize=15)
            g.savefig(f"./figures/nls_i{inf}s{sup}u{uninf}_{i}_{threshold}_pairplot.png", bbox_inches="tight")
            g.savefig(f"./figures/nls_i{inf}s{sup}u{uninf}_{i}_{threshold}_pairplot_hires.png", dpi=300, bbox_inches="tight")

            # plt.show()