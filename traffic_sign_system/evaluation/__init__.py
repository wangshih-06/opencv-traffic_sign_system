"""Evaluation helpers for model bundles and classifier comparison."""

from .comparison import COMPARISON_COLUMNS, plot_comparison, run_comparison
from .error_analysis import (
    errors_per_class,
    plot_errors_per_class,
    plot_top_confusions,
    top_confusions,
)
from .evaluator import evaluate, run_evaluation

__all__ = [
    "COMPARISON_COLUMNS",
    "errors_per_class",
    "evaluate",
    "plot_comparison",
    "plot_errors_per_class",
    "plot_top_confusions",
    "run_comparison",
    "run_evaluation",
    "top_confusions",
]
