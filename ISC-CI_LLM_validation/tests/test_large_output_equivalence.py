import copy

import pytest
import torch
import torch.nn.functional as F

from semantic_retraining.modeling import ISCModel
from semantic_retraining.train_isc import backward_bce, estimate_memory


@pytest.mark.parametrize("chunk_size", [1, 7, 19])
def test_chunked_full_vector_loss_and_every_parameter_gradient_match_dense(chunk_size):
    torch.manual_seed(11)
    dense = ISCModel(4, 19, 3).double()
    chunked = copy.deepcopy(dense)
    inputs = [
        torch.eye(4, dtype=torch.float64),
        torch.eye(3, dtype=torch.float64)[[0, 1, 2, 0]],
    ]
    targets = torch.randint(2, (4, 19)).double()
    expected = F.binary_cross_entropy_with_logits(
        dense(inputs, take_sigmoid=False), targets
    )
    expected.backward()
    actual = backward_bce(
        chunked, inputs, lambda start, stop: targets[:, start:stop], chunk_size
    )
    assert actual == pytest.approx(expected.item(), abs=1e-13)
    for (name, p), (other, q) in zip(
        dense.named_parameters(), chunked.named_parameters()
    ):
        assert name == other
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-10, atol=1e-13)


def test_loss_never_requests_dense_targets_when_chunking():
    model = ISCModel(3, 103, 2)
    sizes = []

    def target(start, stop):
        sizes.append(stop - start)
        return torch.zeros(3, stop - start)

    backward_bce(model, [torch.eye(3), torch.eye(2)[[0, 1, 0]]], target, 11)
    assert sum(sizes) == 103 and max(sizes) == 11
    assert (
        estimate_memory(87315, batch_size=64, chunk_size=4096)["estimated_bytes"]
        > estimate_memory(385)["estimated_bytes"]
    )
