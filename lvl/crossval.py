"""
Cross-validation routines.
"""
import numpy as np
import numpy.random as npr
from lvl.utils import get_random_state


def speckled_cv_scores(
        model, X, fit_params=None, strategy="speckled",
        heldout_frac=0.1, n_repeats=10, seed=None):
    """
    Estimate train and test error for a model by cross-validation.
    """

    # Initialize dictionary for fit keyword args.
    if fit_params is None:
        fit_params = dict()

    # Initialize random number generator.
    rs = get_random_state(seed)

    # Allocate space to store train/test scores.
    train_scores = np.empty(n_repeats)
    test_scores = np.empty(n_repeats)

    # Run cross-validation.
    for itr in range(n_repeats):

        # Generate a new holdout pattern.
        mask = speckled_mask(X.shape, heldout_frac, rs)

        # Fit model.
        model.fit(X, mask=mask)

        # Compute performance on train and test partitions.
        train_scores[itr] = model.score(X, mask=mask)
        test_scores[itr] = model.score(X, mask=~mask)

    return train_scores, test_scores


def speckled_mask(shape, heldout_frac, random_state):
    """
    Creates randomized speckled holdout pattern.
    """

    # Choose heldout indices.
    heldout_num = int(heldout_frac * data.size)
    i = random_state.choice(
        data.size, heldout_num, replace=False)

    # Construct mask.
    mask = np.ones(data.shape, dtype=bool)
    mask[np.unravel_index(i, data.shape)] = False

    # Ensure one observation per row and column.
    safe_entries = np.zeros_like(mask)
    n = np.max(shape)
    ii = rs.permutation(n) % mask.shape[0]
    jj = rs.permutation(n) % mask.shape[1]
    safe_entries[ii, jj] = True

    return mask | safe_entries


def bicv_scores(
        model, X, fit_params=None, strategy="speckled",
        heldout_frac=0.1, n_repeats=10, seed=None):
    """
    Estimate train and test error for a model by bi-cross-validation.
    """

    # Initialize dictionary for fit keyword args.
    if fit_params is None:
        fit_params = dict()
    m, n = X.shape

    # Initialize random number generator.
    rs = get_random_state(seed)

    # Allocate space to store train/test scores.
    train_scores = np.empty(n_repeats)
    test_scores = np.empty(n_repeats)

    # Run cross-validation.
    for itr in range(n_repeats):

        # Draw rows and columns for training set.
        ii = rs.choice(
            m, size=int(m - m * heldout_frac), replace=False)
        jj = rs.choice(
            n, size=int(n - n * heldout_frac), replace=False)

        ni = np.setdiff1d(np.arange(m), ii)
        nj = np.setdiff1d(np.arange(n), jj)

        A = X[ii][:, jj]
        B = X[ii][:, nj]
        C = X[ni][:, jj]

        # Fit model to training set.
        model.fit(A, mask=None)

        # Extend model factors.
        model.bicv_extend(B, C)

        # Construct mask for training set.
        train_mask = np.zeros((m, n), dtype=bool)
        train_mask[ii][:, jj] = True

        # Construct mask for test set.
        test_mask = np.zeros((m, n), dtype=bool)
        train_mask[ni][:, nj] = True

        # Compute performance on train and test partitions.
        train_scores[itr] = model.score(X, mask=train_mask)
        test_scores[itr] = model.score(X, mask=test_mask)

    return train_scores, test_scores
