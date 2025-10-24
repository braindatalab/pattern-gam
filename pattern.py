"""
Partially adapted from:
- https://github.com/KnurpsBram/PyTorch-PatternNet

"""

import torch
from torch import nn
from collections import defaultdict
from functools import partial
from dataclasses import dataclass
from tqdm import tqdm
from captum.attr._core.lrp import LRP
from captum.attr._core.lrp import EpsilonRule, SUPPORTED_NON_LINEAR_LAYERS
from captum.attr._utils.custom_modules import Addition_Module

__LAYERS__ = [
    torch.nn.Linear,
]


def replace_module(model, name, new_module):
    components = name.split(".")
    submodule = model
    for comp in components[:-1]:
        submodule = getattr(submodule, comp)
    setattr(submodule, components[-1], new_module)


def compute_pattern_matrix(x, a):
    """
    Learn pattern matrix A = E[x a^T] @ inv(E[a a^T]) for PatternNet.
    - X (N, D)
    - A (N, M)
    """

    E_xaT = x.T @ a / x.shape[0]  # (D, M)
    E_aaT = a.T @ a / x.shape[0]  # (M, M)
    try:
        A = E_xaT @ torch.linalg.pinv(E_aaT)  # (D, M)
    except RuntimeError:
        A = E_xaT @ torch.linalg.pinv(
            E_aaT + 1e-6 * torch.eye(E_aaT.size(0), device=E_aaT.device)
        )
    return A.T  # (M, D)


@dataclass
class Pattern:
    module: torch.nn.Module
    is_last_layer: bool
    init: bool = False
    pattern: torch.Tensor = None
    pattern_regr: torch.Tensor = None
    s_xy: torch.Tensor = None
    s_x: torch.Tensor = None
    s_y: torch.Tensor = None
    n_x: torch.Tensor = None
    n_y: int = None
    x_: list = None
    y_: list = None

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.init = False
        self.s_xy = None
        self.s_x = None
        self.s_y = None
        self.n_x = None
        self.n_y = None
        self.n_y = 0
        self.x_ = []
        self.y_ = []

    def update(self, x: torch.Tensor, y: torch.Tensor):
        """
        Update the pattern with the current input and output of the module.
        To be clear, y is the output of the module, not the model.
        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Output tensor.
        """
        w = self.module.weight.data
        if self.init is False:
            self.init = True
            self.s_xy = torch.zeros_like(w)
            self.s_x = torch.zeros_like(w)
            self.s_y = torch.zeros(w.size(0), device=w.device)
            self.n_x = torch.zeros(w.size(0), device=w.device)

        x = x[0]

        self.x_.append(x)
        self.y_.append(y)

        if not self.is_last_layer:
            m = (y > 0).float()
            self.s_x += (m[:, :, None] @ x[:, None, :]).sum(dim=0)
            self.s_y = y.sum(dim=0)
            self.s_xy += ((m * y)[:, :, None] @ x[:, None, :]).sum(dim=0)
            self.n_x += m.sum(dim=0)
            self.n_y += x.shape[0]

            if self.x_ is None:
                self.x_ = []
            if self.y_ is None:
                self.y_ = []

    def compute_linear_pattern(self, x, y):
        assert y.shape[1] == 1, "y should be a single output"
        E_XY = torch.mean(x * y, dim=0)
        E_X = torch.mean(x, dim=0)
        E_Y = torch.mean(y, dim=0)
        cov = E_XY - E_X * E_Y
        var_y = torch.var(y, dim=0)
        print("cov", cov)

        return (cov / var_y).unsqueeze(0)


    def compute(self):
        if not self.x_ or not self.y_:
            print(f"Warning: No activations collected for module {self.module}, cannot compute patterns.")
            self.pattern = None # Or an appropriate default
            return

        x = torch.cat(self.x_, dim=0)
        y = torch.cat(self.y_, dim=0)
        compute_device = self.module.weight.data.device
        x = x.to(compute_device)
        y = y.to(compute_device)

        if self.is_last_layer:
            self.pattern = self.compute_linear_pattern(x, y)
            if self.pattern is not None:
                 print(f"  Last layer pattern has NaNs: {torch.isnan(self.pattern).any()}, Infs: {torch.isinf(self.pattern).any()}")
        else:
            m = (y > 0).float()
            n_x_calc = m.sum(dim=0)
            n_x_safe = n_x_calc + 1e-9
            
            s_x_calc = torch.einsum('nm,nd->md', m, x)
            s_y_calc = (m * y).sum(dim=0)
            s_xy_calc = torch.einsum('nm,nm,nd->md', m, y, x)

            e_x = s_x_calc / n_x_safe[:, None]
            e_y = s_y_calc / n_x_safe
            e_xy = s_xy_calc / n_x_safe[:, None]
            
            e_x = torch.nan_to_num(e_x, nan=0.0, posinf=1e12, neginf=-1e12) # Clip inf too
            e_y = torch.nan_to_num(e_y, nan=0.0, posinf=1e12, neginf=-1e12)
            e_xy = torch.nan_to_num(e_xy, nan=0.0, posinf=1e12, neginf=-1e12)

            w = self.module.weight.data
            ex_ey = e_x * e_y[:, None]
            num = e_xy - ex_ey
            
            denom_matrix = num @ w.T
            denom = torch.diag(denom_matrix)
            denom_safe = denom + torch.sign(denom) * 1e-9
            denom_safe[denom_safe == 0] = 1e-9 # Avoid 0 in denom_safe

            self.pattern = num / denom_safe[:, None]
            # print(f"  Computed pattern has NaNs: {torch.isnan(self.pattern).any()}, Infs: {torch.isinf(self.pattern).any()}")
            if torch.isnan(self.pattern).any():
                print("  >>> NaN DETECTED IN PATTERN <<<")
                # Find the NaN row index
                nan_rows = torch.isnan(self.pattern).any(dim=1)
                if nan_rows.any():
                    print(f"  NaNs in pattern rows: {nan_rows.nonzero().squeeze().tolist()}")
                    first_nan_row_idx = nan_rows.nonzero().squeeze().tolist()
                    if isinstance(first_nan_row_idx, list): first_nan_row_idx = first_nan_row_idx[0]

                    print(f"  Values for first NaN row ({first_nan_row_idx}) in pattern:")
                    print(f"    num[{first_nan_row_idx}]: {num[first_nan_row_idx]}")
                    print(f"    denom_safe[{first_nan_row_idx}]: {denom_safe[first_nan_row_idx]}")
                    print(f"    e_x[{first_nan_row_idx}]: {e_x[first_nan_row_idx]}")
                    print(f"    e_y[{first_nan_row_idx}]: {e_y[first_nan_row_idx]}")
                    print(f"    e_xy[{first_nan_row_idx}]: {e_xy[first_nan_row_idx]}")
                    print(f"    n_x_calc[{first_nan_row_idx}]: {n_x_calc[first_nan_row_idx]}")
                    # print(f"    Relevant y values (output of this neuron for this row): y[:, {first_nan_row_idx}] range: [{y[:, first_nan_row_idx].min():.4e}, {y[:, first_nan_row_idx].max():.4e}]")


        self.pattern_regr = compute_pattern_matrix(a=y, x=x) # This might also fail if x,y are bad
        self.reset()

class PatternBase:
    def __init__(self, model):
        print("PatternBase Init")
        self.model = model
        self.device = next(model.parameters()).device
        self.patterns = {}

    def get_inputs(self, batch, input_key = 0):
        """
        Get the input tensor from the batch.
        Args:
            batch (tuple | dict | torch.Tensor): The input batch.
            input_key (int | str): The key to extract the input tensor.
        Returns:
            torch.Tensor: The input tensor.
        """
        if (
            isinstance(batch, tuple)
            or isinstance(batch, list)
            and isinstance(input_key, int)
        ):
            inputs = batch[input_key]
        elif isinstance(batch, dict):
            inputs = batch.get(input_key)
        elif isinstance(batch, torch.Tensor):
            inputs = batch
        else:
            raise ValueError("Unsupported data type for input extraction.")
        return inputs

    def train(self, dataloader: torch.utils.data.DataLoader, input_key = 0):
        def save_activation_hook(name, is_last_layer, module, input, output):
            if name not in self.patterns:
                self.patterns[name] = Pattern(module, is_last_layer)

            self.patterns[name].update(input, output)

        hooks = []
        last_layer = [child for name, child in self.model.named_modules()][-1]
        for name, module in self.model.named_modules():
            # Check if it is the output layer
            if isinstance(module, tuple(__LAYERS__)):
                print(f"Registering layer: {name}, Module: {module}")
                is_last_layer = module == last_layer
                print(f"Is last layer: {is_last_layer}")
                hook = module.register_forward_hook(
                    partial(save_activation_hook, name, is_last_layer)
                )
                hooks.append(hook)

        # Iterate through the dataloader to train the model
        for data in tqdm(dataloader, desc="Training patterns"):
            inputs = self.get_inputs(data, input_key)
            inputs = inputs.to(self.device)
            with torch.no_grad():
                self.model(inputs)

        # Remove hooks after training
        for hook in hooks:
            hook.remove()

        for name, pattern in self.patterns.items():
            pattern.compute()


class PatternNet(PatternBase):
    def _backward_hook(self, name, m, i, o):
        """
        Multiply the gradient with the pattern instead of the weight.
        """
        pattern = self.patterns[name].pattern
        # print(pattern)
        # print(o[0]) 
        return (o[0] @ pattern,)

    def attribute(
        self,
        inputs,
        target=None,
    ):
        req_grad = inputs.requires_grad
        inputs.requires_grad_(True)

        self.model.zero_grad()

        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, tuple(__LAYERS__)):
                h = module.register_full_backward_hook(
                    partial(self._backward_hook, name)
                )
                hooks.append(h)

        outputs = self.model(inputs)
        outputs.backward(gradient=outputs)
        signal = inputs.grad

        inputs.requires_grad_(req_grad)
        for hook in hooks:
            hook.remove()

        return signal, outputs


class PatternAttribution(LRP, PatternBase):
    def __init__(self, model):
        LRP.__init__(self, model)
        PatternBase.__init__(self, model)

    def _check_and_attach_rules(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "rule"):
                layer.activations = {}  # type: ignore
                layer.rule.relevance_input = defaultdict(list)  # type: ignore
                layer.rule.relevance_output = {}  # type: ignore
                pass
            elif type(layer) in SUPPORTED_LAYERS_WITH_RULES_PATTERN.keys():
                layer.activations = {}  # type: ignore
                layer.rule = SUPPORTED_LAYERS_WITH_RULES_PATTERN[type(layer)]()  # type: ignore
                layer.rule.relevance_input = defaultdict(list)  # type: ignore
                layer.rule.relevance_output = {}  # type: ignore
            elif type(layer) in SUPPORTED_NON_LINEAR_LAYERS:
                layer.rule = None  # type: ignore
            else:
                raise TypeError(
                    (
                        f"Module of type {type(layer)} has no rule defined and no"
                        "default rule exists for this module type. Please, set a rule"
                        "explicitly for this module and assure that it is appropriate"
                        "for this type of layer."
                    )
                )

        for name, module in self.model.named_modules():
            if isinstance(module, tuple(__LAYERS__)) and hasattr(module, "rule"):
                if name not in self.patterns:
                    raise Exception(
                        f"Pattern for layer {name} not found. Please train the pattern "
                        "before using it."
                    )

                module.rule.set_pattern(self.patterns[name].pattern)


class EpsilonPatternRule(EpsilonRule):
    def set_pattern(self, pattern: torch.Tensor):
        self.pattern = pattern

    def _manipulate_weights(self, module, inputs, outputs):
        if hasattr(module, "weight") and self.pattern is not None:
            module.weight.data = self.pattern * module.weight.data


SUPPORTED_LAYERS_WITH_RULES_PATTERN = {
    nn.MaxPool1d: EpsilonPatternRule,
    nn.MaxPool2d: EpsilonPatternRule,
    nn.MaxPool3d: EpsilonPatternRule,
    nn.Conv2d: EpsilonPatternRule,
    nn.AvgPool2d: EpsilonPatternRule,
    nn.AdaptiveAvgPool2d: EpsilonPatternRule,
    nn.Linear: EpsilonPatternRule,
    nn.BatchNorm2d: EpsilonPatternRule,
    Addition_Module: EpsilonPatternRule,
}
