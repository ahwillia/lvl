"""
Cross-validation routines.
"""
import numpy as np
import numpy.random as npr
from lvl.utils import get_random_state
from tqdm import trange


def speckled_cv_scores(
        model, X, fit_params=None, heldout_frac=0.1, n_repeats=10,
        resampler=None, return_params=False, seed=None,
        progress_bar=False):
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

    params = []

    # Run cross-validation.
    pbar = trange(n_repeats) if progress_bar else range(n_repeats)
    for itr in pbar:

        # If desired, resample X (e.g. apply random shuffle).
        if resampler is not None:
            Xsamp = resampler(X)
        else:
            Xsamp = X

        # Generate a new holdout pattern.
        mask = speckled_mask(X.shape, heldout_frac, rs)

        # Fit model.
        model.fit(Xsamp, mask=mask)

        # Save parameters.
        if return_params:
            params.append(tuple(p.copy() for p in model.factors))

        # Compute performance on train and test partitions.
        train_scores[itr] = model.score(Xsamp, mask=mask)
        test_scores[itr] = model.score(Xsamp, mask=~mask)

    # Return data.
    return (
        (train_scores, test_scores, params)
        if return_params else
        (train_scores, test_scores)
    )


def speckled_mask(shape, heldout_frac, rs):
    """
    Creates randomized speckled holdout pattern.
    """

    # Choose heldout indices.
    heldout_num = int(heldout_frac * np.prod(shape))
    i = rs.choice(
        np.prod(shape), heldout_num, replace=False)

    # Construct mask.
    mask = np.ones(shape, dtype=bool)
    mask[np.unravel_index(i, shape)] = False

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

        # Create shuffled view into data.
        ii = rs.permutation(m)
        jj = rs.permutation(n)
        Xs = np.copy(X[ii][:, jj])

        # Partition columns and rows.
        si = int(m - m * heldout_frac)
        sj = int(n - n * heldout_frac)

        # Fit model to training set.
        model.fit(Xs[:si, :sj], mask=None)

        # Extend model factors.
        model.bicv_extend(Xs[:si, sj:], Xs[si:, :sj])

        # Construct mask for training set.
        train_mask = np.zeros((m, n), dtype=bool)
        train_mask[:si, :sj] = True

        # Construct mask for test set.
        test_mask = np.zeros((m, n), dtype=bool)
        test_mask[si:, sj:] = True

        # Compute performance on train and test partitions.
        train_scores[itr] = model.score(Xs, mask=train_mask)
        test_scores[itr] = model.score(Xs, mask=test_mask)

    return train_scores, test_scores
