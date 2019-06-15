"""
Cross-validation routines.
"""
import numpy as np
import numpy.random as npr


def cv_scores(
        model, X, fit_params=None, strategy="speckled",
        heldout_frac=0.1, n_repeats=10, seed=None):
    """
    Estimate train and test error for a model by cross-validation.
    """

    # Determine holdout strategy.
    _s = {
        "speckled": SpeckledHoldout
    }
    if isinstance(strategy, HoldoutStrategy):
        get_mask = strategy
    else:
        get_mask = _s[strategy](heldout_frac, seed)

    # Initialize dictionary for fit keyword args.
    if fit_params is None:
        fit_params = dict()

    # Allocate space to store train/test scores.
    train_scores = np.empty(n_repeats)
    test_scores = np.empty(n_repeats)

    # Run cross-validation.
    for itr in range(n_repeats):

        # Generate a new holdout pattern.
        mask = get_mask(X)

        # Fit model.
        model.fit(X, mask=mask)

        # Compute performance on train and test partitions.
        train_scores[itr] = model.score(X, mask=mask)
        test_scores[itr] = model.score(X, mask=~mask)

    return train_scores, test_scores


class HoldoutStrategy:
    """
    Base class for cross-validation holdout strategies.
    """
    def __init__(self, heldout_frac, seed):
        self.heldout_frac = heldout_frac
        self.rs = npr.RandomState(seed)

    def __call__(self):
        raise NotImplementedError(
            "Base class must override __call__ method.")


class SpeckledHoldout(HoldoutStrategy):
    """
    Leave out data from a multi-dimensional array at
    random.
    """
    def __init__(self, *args, **kwargs):
        super(SpeckledHoldout, self).__init__(*args, **kwargs)

    def __call__(self, data):

        # Choose heldout indices.
        heldout_num = int(self.heldout_frac * data.size)
        i = self.rs.choice(data.size, heldout_num, replace=False)

        # Construct mask.
        mask = np.ones(data.shape, dtype=bool)
        mask[np.unravel_index(i, data.shape)] = False
        return mask
