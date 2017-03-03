#!/usr/bin/env python
"""
The neural network is trained on question-answer pairs from
the Yahoo L6 corpus. This is a really large file, so it isn't
included in the Git repository, but it can be downloaded from:
http://webscope.sandbox.yahoo.com/catalog.php?datatype=l
"""

from __future__ import absolute_import
from __future__ import print_function

import argparse
import yahoo

for a, b, c in yahoo.iterate_qa_data(32):
    print(a.shape, b.shape, c.shape)
