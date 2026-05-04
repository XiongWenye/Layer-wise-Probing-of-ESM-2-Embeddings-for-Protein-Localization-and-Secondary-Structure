import torch

from esm_probe.models.pooling import MaskedPooling


def test_pooling_shapes() -> None:
    x = torch.randn(3, 5, 7)
    mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    for mode in ["mean", "max", "attention"]:
        pooled = MaskedPooling(mode, 7)(x, mask)
        assert pooled.shape == (3, 7)
