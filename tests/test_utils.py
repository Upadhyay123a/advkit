import logging

import pytest
import torch

from advkit.utils.metrics import perturbation_budget_report


def test_perturbation_budget_report_reports_norms_and_warns_on_epsilon_violation(caplog):
    original = torch.ones(2, 2, 2)
    adversarial = torch.zeros(2, 2, 2)

    with caplog.at_level(logging.WARNING):
        report = perturbation_budget_report(original, adversarial, epsilon=0.02)

    assert report["l2_norm"] == pytest.approx(2.8284271247461903)
    assert report["linf_norm"] == pytest.approx(1.0)
    assert "Perturbation exceeds epsilon budget" in caplog.text
