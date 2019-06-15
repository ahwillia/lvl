"""
Common utility functions.
"""
import numpy.random as npr


def get_random_state(seed_or_rs):
    """
    Converts an integer to a seeded RandomState instance.
    If input is already a RandomState instance, it returns
    it unchanged.
    """

    if isinstance(seed_or_rs, npr.RandomState):
        return seed_or_rs
    else:
        return npr.RandomState(seed_or_rs)
