"""Manage Pytest-cov output on workflow."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
import sys

from github_actions_utils.color import score_to_hex_color


def check_output() -> float:
    """Check output of Pytest-cov.

    Raises
    ------
    ValueError
        If Pytest find failures.
    ValueError
        If coverage is below SCORE_MIN.

    Returns
    -------
    score: float
        Score of coverage.
    """
    args = sys.argv[1:]
    n_failures, score = -1, -1.0  # Default values
    for arg in args:
        if arg.startswith("--score="):
            score_percent = arg.split("=")[1]
            score = float(score_percent.split("%")[0])
        if arg.startswith("--n_failures="):
            n_failures_str = arg.split("=")[1]
            n_failures = 0 if n_failures_str == "" else int(n_failures_str)

    if n_failures == -1:
        raise ValueError(
            "Please specify the number of failures found by Pytest "
            "using the flag --n_failures=N.",
        )
    if score == -1:
        raise ValueError(
            "Please specify the score of coverage using the flag --score=N (in %).",
        )

    if n_failures > 0:
        raise ValueError(f"Pytest finds {n_failures} failure(s) on tests.")
    if score < COV_SCORE_MIN:
        raise ValueError(
            f"Pytest coverage {score}% is lower than minimum ({COV_SCORE_MIN}%)",
        )

    return score


def main() -> None:
    """Check score and print it."""
    score = check_output()
    # Print color to be used in GitHub Actions
    print(score_to_hex_color(score, COV_SCORE_MIN, COV_SCORE_MAX))


if __name__ == "__main__":
    # COV_SCORE_MIN can be changed safely depending on your needs.
    # NOTE: score on %
    COV_SCORE_MIN = 0
    COV_SCORE_MAX = 100
    main()
