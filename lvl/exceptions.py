"""
Common error messages.
"""


def raise_not_fitted(model, method_name):
    model_name = type(model).__name__
    raise ValueError(
        "Attempted to call {}.{} before fitting the model."
        "Call {}.fit(...) first.".format(
            model_name, method_name, model_name))


def raise_no_method(model, method_name, implemented):
    model_name = type(model).__name__
    raise ValueError(
        "Did not recognize optimization method '{}' for "
        "{} model. Choose from"
        "{}".format(model_name, method_name, implemented))


def raise_no_init(model, method_name, implemented):
    model_name = type(model).__name__
    raise ValueError(
        "Did not recognize initialization method '{}' for "
        "{} model. Choose from"
        "{}".format(model_name, method_name, implemented))
