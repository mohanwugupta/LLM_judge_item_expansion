from __future__ import annotations

import torch
from torch import nn


class ISCModel(nn.Module):
    """Original ISC graph and initialization, with standard nn.Module modes.

    Matches upstream revision 4ecc76d4817fb927d2ad79dfbe1ff44ea72affc0.
    In released no-hidden-bias mode, the direct output bias starts at -2 but
    remains trainable. This differs from the paper's fixed-bias description.
    No checkpoint is read during initialization.
    """

    def __init__(
        self,
        num_objects: int,
        num_output: int,
        num_tasks: int,
        num_hub_hidden_units: int = 64,
        num_context_dependent_hidden_units: int = 128,
        num_task_context_units: int = 16,
        device: str = "cpu",
        biases: bool = False,
    ) -> None:
        super().__init__()
        self.num_objects = num_objects
        self.num_output = num_output
        self.num_tasks = num_tasks
        self.num_context_dependent_hidden_units = num_context_dependent_hidden_units
        self.device = device
        self.item_input_to_hub_weights = nn.Linear(
            num_objects, num_hub_hidden_units, device=device, bias=biases
        )
        self.context_input_to_task_context_rep_weights = nn.Linear(
            num_tasks, num_task_context_units, device=device, bias=biases
        )
        self.task_context_rep_to_context_dependent_rep_weights = nn.Linear(
            num_task_context_units,
            num_context_dependent_hidden_units,
            device=device,
            bias=biases,
        )
        self.hub_to_context_dependent_rep_weights = nn.Linear(
            num_hub_hidden_units,
            num_context_dependent_hidden_units,
            device=device,
            bias=biases,
        )
        self.context_dependent_rep_to_output_weights = nn.Linear(
            num_context_dependent_hidden_units, num_output, device=device, bias=biases
        )
        self.hub_to_output_weights = nn.Linear(
            num_hub_hidden_units, num_output, device=device
        )
        layers = [
            self.item_input_to_hub_weights,
            self.context_input_to_task_context_rep_weights,
            self.task_context_rep_to_context_dependent_rep_weights,
            self.hub_to_context_dependent_rep_weights,
            self.context_dependent_rep_to_output_weights,
            self.hub_to_output_weights,
        ]
        for layer in layers:
            nn.init.uniform_(layer.weight, a=-0.01, b=0.01)
        if biases:
            for layer in layers:
                nn.init.uniform_(layer.bias, a=-0.01, b=0.01)
        else:
            with torch.no_grad():
                self.hub_to_output_weights.bias.fill_(-2)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def get_context_independent_rep(self, x: list[torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(self.item_input_to_hub_weights(x[0]))

    def get_task_context_rep(self, x: list[torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(self.context_input_to_task_context_rep_weights(x[1]))

    def get_context_dependent_rep(self, x: list[torch.Tensor]) -> torch.Tensor:
        hub = self.get_context_independent_rep(x)
        task = self.get_task_context_rep(x)
        return torch.sigmoid(
            self.task_context_rep_to_context_dependent_rep_weights(task)
            + self.hub_to_context_dependent_rep_weights(hub)
        )

    def forward(self, x: list[torch.Tensor], take_sigmoid: bool = True) -> torch.Tensor:
        hub = self.get_context_independent_rep(x)
        task = self.get_task_context_rep(x)
        dependent = torch.sigmoid(
            self.task_context_rep_to_context_dependent_rep_weights(task)
            + self.hub_to_context_dependent_rep_weights(hub)
        )
        output = self.hub_to_output_weights(
            hub
        ) + self.context_dependent_rep_to_output_weights(dependent)
        return torch.sigmoid(output) if take_sigmoid else output
