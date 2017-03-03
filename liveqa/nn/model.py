"""
Defines the network model to use.

The current model is a seq2seq model that tries to predict the question title
and body from the answer body.
"""

import keras


def build_model(qtitle_len, qbody_len, abody_len):
    """Builds the model.

    Args:
        qtitle_len: int, the length of the question title.
        qbody_len: int, the length of the question body.
        abody_len: int, the length of the answer body.

    Returns:
        A model with the right shapes.
    """
