"""
Common error messages.
"""


def raise_not_fitted(model_name, method_name):
    raise ValueError(
        "Attempted to call {}.{} before fitting the model."
        "Call {}.fit(...) first.".format(
            model_name, method_name, model_name))


def raise_no_method(model_name, method_name, implemented):
    raise ValueError(
        "Did not recognize optimization method '{}' for "
        "{} model. Choose from"
        "{}".format(model_name, method_name, implemented))


def raise_no_init(model_name, method_name, implemented):
    raise ValueError(
        "Did not recognize initialization method '{}' for "
        "{} model. Choose from"
        "{}".format(model_name, method_name, implemented))
