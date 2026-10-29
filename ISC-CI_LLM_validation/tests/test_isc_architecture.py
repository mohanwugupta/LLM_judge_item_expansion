import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn, optim

from semantic_retraining.modeling import ISCModel


def released_class():
    path = (
        Path(__file__).resolve().parents[1]
        / "upstream/IntegratedSemanticsControl/models.py"
    )
    module = ast.parse(path.read_text())
    definition = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ISCModel"
    )
    namespace = {
        "torch": torch,
        "nn": nn,
        "np": np,
        "optim": optim,
        "utils": SimpleNamespace(set_torch_device=lambda: "cpu"),
        "BCEMetric": lambda: None,
        "TPRMetric": lambda: None,
    }
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["ISCModel"]


@pytest.mark.parametrize("biases", [False, True])
def test_initialization_and_forward_match_unedited_released_graph(biases):
    parameters = dict(
        num_objects=4, num_output=7, num_tasks=3, device="cpu", biases=biases
    )
    torch.manual_seed(10)
    original = released_class()(**parameters)
    torch.manual_seed(10)
    rebuilt = ISCModel(**parameters)
    assert original.state_dict().keys() == rebuilt.state_dict().keys()
    for name, weight in original.state_dict().items():
        assert torch.equal(weight, rebuilt.state_dict()[name]), name
    inputs = [torch.eye(4), torch.eye(3)[torch.tensor([0, 1, 2, 0])]]
    for sigmoid in [True, False]:
        torch.testing.assert_close(
            original(inputs, take_sigmoid=sigmoid),
            rebuilt(inputs, take_sigmoid=sigmoid),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        original.get_context_independent_rep(inputs),
        rebuilt.get_context_independent_rep(inputs),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        original.get_context_dependent_rep(inputs),
        rebuilt.get_context_dependent_rep(inputs),
        rtol=0,
        atol=0,
    )


def test_full_vector_bce_gradients_and_task_independent_hub():
    torch.manual_seed(2)
    model = ISCModel(num_objects=4, num_output=7, num_tasks=3)
    inputs = [torch.eye(4), torch.eye(3)[torch.tensor([0, 1, 2, 0])]]
    before = model.get_context_independent_rep(inputs)
    other_task = [inputs[0], inputs[1].roll(1, dims=1)]
    assert torch.equal(before, model.get_context_independent_rep(other_task))
    assert not torch.equal(model(inputs), model(other_task))
    assert before.shape == (4, 64)
    assert model.get_task_context_rep(inputs).shape == (4, 16)
    assert model.get_context_dependent_rep(inputs).shape == (4, 128)
    targets = torch.zeros(4, 7)
    targets[0, -1] = 1
    model.loss_fn(model(inputs, take_sigmoid=False), targets).backward()
    assert model.item_input_to_hub_weights.weight.grad.abs().sum() > 0
    assert model.context_input_to_task_context_rep_weights.weight.grad.abs().sum() > 0
    assert model.hub_to_output_weights.bias.requires_grad
    assert model.item_input_to_hub_weights.bias is None


def test_neural_module_train_and_eval_modes_are_not_shadowed():
    model = ISCModel(num_objects=2, num_output=2, num_tasks=2)
    assert model.eval().training is False
    assert model.train().training is True
