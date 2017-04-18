"""Command-line utility for experimenting with the API.

To use this script, run:
    python -m liveqa.api.command_line
"""

from __future__ import absolute_import
from __future__ import print_function

print('Initializing the API...')

import logging
logging.basicConfig(level=logging.DEBUG)

import argparse

parser = argparse.ArgumentParser(
    description='Command-line utility for the LiveQA API.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--mode',
                    default='answer',
                    choices=['question', 'answer'],
                    help='The mode to use.')
parser.add_argument('-l', '--use_lda',
                    default=False,
                    action='store_true',
                    help='If set, use LDA instead of neural network.')
args = parser.parse_args()


if args.use_lda:
    from liveqa.api.lda_api import get_question, get_answer
else:
    from liveqa.api import get_question, get_answer

if args.mode.lower() == 'question':
    api = get_question
elif args.mode.lower() == 'answer':
    api = get_answer
else:
    raise ValueError('Invalid mode: "%s".' % args.mode)

print('Running "%s" mode.' % args.mode)
query = raw_input('Enter a query (None to break): ')
while query:
    print('Response:', api(query))
    query = raw_input('Enter another query (None to end): ')
