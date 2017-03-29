"""Defines the default API behavior.

This gives a simple endpoint for accessing the default get_question and
get_answer APIs. For future work this will make it easy to switch the backend
type being used (for example, from LDA re-ranking to neural network re-ranking,
which seems to work marginally better).

Usage:
    from liveqa.api import get_question
    from liveqa.api import get_answer
"""

from __future__ import absolute_import

# Currently using the LDA version.
# from liveqa.api.lda_api import get_question
# from liveqa.api.lda_api import get_answer

from liveqa.api.nn_api import get_question
from liveqa.api.nn_api import get_answer
