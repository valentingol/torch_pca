"""Manage Pylint output on workflow."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
import sys

from github_actions_utils.color import score_to_hex_color


def check_output() -> float:
    """Check output of Pylint.

    Raises
    ------
    ValueError
        If Pylint score is below SCORE_MIN.

    Returns
    -------
    score: float
        Score of Pylint.
    """
    args = sys.argv[1:]
    score = -10000.0  # A default value
    for arg in args:
        if arg.startswith("--score="):
            score = float(arg.split("=")[1])
    if score == -10000.0:
        raise ValueError("Please specify the score of Pylint using the flag --score=N.")
    if score < PYLINT_SCORE_MIN:
        raise ValueError(
            f"Pylint score {score} is lower than minimum ({PYLINT_SCORE_MIN})"
        )

    return score


def main() -> None:
    """Check score and print it."""
    score = check_output()
    # Print color to be used in GitHub Actions
    print(score_to_hex_color(score, PYLINT_SCORE_MIN, PYLINT_SCORE_MAX))


if __name__ == "__main__":
    # PYLINT_SCORE_MIN can be changed safely depending on your needs.
    PYLINT_SCORE_MIN = 7.0
    PYLINT_SCORE_MAX = 10.0
    main()
