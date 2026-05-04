import torch

from esm_probe.evaluation.tensors import cat_padded_batch_tensors


def test_cat_padded_batch_tensors_pads_sequence_dimension() -> None:
    first = torch.ones(2, 3, 4)
    second = torch.full((1, 5, 4), 2.0)

    result = cat_padded_batch_tensors([first, second])

    assert result.shape == (3, 5, 4)
    assert torch.equal(result[:2, :3], first)
    assert torch.equal(result[:2, 3:], torch.zeros(2, 2, 4))
    assert torch.equal(result[2:], second)


def test_cat_padded_batch_tensors_uses_bool_pad_value() -> None:
    first = torch.ones(1, 2, dtype=torch.bool)
    second = torch.ones(1, 4, dtype=torch.bool)

    result = cat_padded_batch_tensors([first, second], pad_value=False)

    assert result.shape == (2, 4)
    assert result.tolist() == [[True, True, False, False], [True, True, True, True]]
