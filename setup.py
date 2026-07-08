#!/usr/bin/env python
"""
Minimal setup.py for legacy compatibility.

Modern packaging metadata lives in pyproject.toml.
This file allows `pip install -e .` to work even with very old pip/setuptools.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
