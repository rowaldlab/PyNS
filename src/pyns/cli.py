"""
CLI entry points for PyNS executables.

This module provides command-line interface entry points for the main PyNS scripts.
"""

import sys
import runpy


def run_discrete_simulations():
    """Entry point for run-discrete-simulations command."""
    # Run the module directly
    runpy.run_module('pyns.run_discrete_simulations', run_name='__main__')


def run_titrations():
    """Entry point for run-titrations command."""
    # Run the module directly
    runpy.run_module('pyns.run_titrations', run_name='__main__')


if __name__ == "__main__":
    print("This module is not meant to be run directly.")
    print("Use the installed console scripts:")
    print("  - run-discrete-simulations")
    print("  - run-titrations")
