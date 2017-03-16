"""
Trains a model using Google's seq2seq implementation (instead of recreating
everthing from scratch in TensorFlow, which has been exceedingly difficult
with their new API).

This file mainly provides preprocessing utils to get the data into the correct
format, so that it can be used by the main script.
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys


def main():
    """Combines the relevant preprocessing functions into a single function."""

    print('hello world!')

if __name__ == '__main__':
    main()
