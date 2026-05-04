import pandas as pd

from esm_probe.baselines.secondary_structure import (
    evaluate_secondary_baseline,
    fit_secondary_baseline,
    predict_q3,
)


LABELS = ["H", "E", "C"]


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["p1", "p2", "p3"],
            "sequence": ["AAAA", "CCDD", "XXXX"],
            "ss_q3": ["HHHC", "CCEE", "CCCC"],
        }
    )


def test_majority_baseline_predicts_global_majority() -> None:
    model = fit_secondary_baseline(_frame(), ["p1", "p2", "p3"], "majority", LABELS)

    assert model.global_label == "C"
    assert predict_q3(model, "ACDZ") == "CCCC"


def test_aa_lookup_uses_residue_majority_and_fallback() -> None:
    model = fit_secondary_baseline(_frame(), ["p1", "p2", "p3"], "aa_lookup", LABELS)

    assert predict_q3(model, "ACDZ") == "HCEC"


def test_secondary_baseline_predictions_match_sequence_length() -> None:
    model = fit_secondary_baseline(_frame(), ["p1"], "aa_lookup", LABELS)

    metrics, predictions = evaluate_secondary_baseline(_frame(), ["p2"], model, LABELS)

    assert list(predictions["length"]) == [4]
    assert len(predictions.loc[0, "pred_q3"]) == 4
    assert "q3_accuracy" in metrics
