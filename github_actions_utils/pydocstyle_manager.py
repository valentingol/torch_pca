"""Manage Pydocstyle output on workflow."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
import sys


def check_output() -> None:
    """Check output of Pydocstyle.

    Raises
    ------
    ValueError
        If Pydocstyle find errors.
    """
    args = sys.argv[1:]
    n_errors = -1  # A default value
    for arg in args:
        if arg.startswith("--n_errors="):
            n_errors = int(arg.split("=")[1])
    if n_errors == -1:
        raise ValueError(
            "Please specify the number of errors found by Pydocstyle "
            "using the flag --n_errors=N."
        )
    if n_errors > 0:
        raise ValueError(
            f"Pydocstyle found {n_errors} errors in python "
            "docstrings. Please fix them.",
        )


if __name__ == "__main__":
    check_output()
