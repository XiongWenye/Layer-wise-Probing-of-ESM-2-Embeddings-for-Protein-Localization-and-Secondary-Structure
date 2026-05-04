from pathlib import Path

from esm_probe.config import load_config


def test_load_config_override(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
project:
  run_group: example
data:
  folds: [0, 1]
""",
        encoding="utf-8",
    )
    cfg = load_config(path, ["data.folds=[2]", "project.seed=7"])
    assert cfg.data.folds == [2]
    assert cfg.project.seed == 7
    assert cfg.project.run_group == "example"
