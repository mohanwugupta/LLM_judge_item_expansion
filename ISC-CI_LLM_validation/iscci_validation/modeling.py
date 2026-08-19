from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CICOModel(nn.Module):
    """Released ISC-CI architecture with a variable-size task-label head."""

    def __init__(
        self,
        input_d: int,
        task_output_d: int,
        embedding_d: int = 64,
        context_d: int = 128,
        context_dependent_d: int = 128,
        nonlinearity: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.input_to_independent = nn.Embedding(input_d, embedding_d)
        self.independent_to_context = nn.Linear(embedding_d, context_d)
        self.context_to_dependent = nn.Linear(context_d, context_dependent_d)
        self.independent_to_dependent = nn.Linear(embedding_d, context_dependent_d)
        self.dependent_to_ft_output = nn.Linear(context_dependent_d, 1, bias=False)
        self.context_to_task_output = nn.Linear(context_d, task_output_d)
        self.nonlinearity = nonlinearity()

    def get_independent_rep(self, x: torch.Tensor) -> torch.Tensor:
        return self.input_to_independent(x)

    def get_context_rep(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.get_independent_rep(x)
        average_embedding = embedding.mean(dim=1)
        return self.nonlinearity(self.independent_to_context(average_embedding))

    def get_context_dependent_rep(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        if context.ndim == 2:
            context = context.unsqueeze(1).repeat(1, x.shape[1], 1)
        return self.nonlinearity(
            self.independent_to_dependent(self.get_independent_rep(x))
            + self.context_to_dependent(context)
        )

    def get_ft_output(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.dependent_to_ft_output(
            self.get_context_dependent_rep(x, context)
        )

    def losses(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        support_c: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        context_rep = self.get_context_rep(support_x)
        task_pred = self.context_to_task_output(context_rep)
        support_pred = self.get_ft_output(support_x, context_rep).squeeze(-1)
        query_pred = self.get_ft_output(query_x, context_rep).squeeze(-1)
        task_loss = F.cross_entropy(task_pred, support_c[:, 0].long())
        support_loss = F.binary_cross_entropy_with_logits(
            support_pred, support_y.float()
        )
        query_loss = F.binary_cross_entropy_with_logits(query_pred, query_y.float())
        return task_loss, support_loss, query_loss, task_pred, support_pred, query_pred


def initialize_model(
    task_count: int,
    released_embedding: torch.Tensor,
    seed: int,
) -> CICOModel:
    torch.manual_seed(seed)
    model = CICOModel(
        input_d=released_embedding.shape[0],
        task_output_d=task_count,
        embedding_d=released_embedding.shape[1],
        context_d=128,
        context_dependent_d=128,
        nonlinearity=nn.ReLU,
    )
    with torch.no_grad():
        model.input_to_independent.weight.copy_(released_embedding)
    model.input_to_independent.weight.requires_grad_(False)
    return model
