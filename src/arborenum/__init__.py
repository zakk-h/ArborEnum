import functools as _functools
import inspect as _inspect
from . import _api as _api

for _name, _value in vars(_api).items():
    if not _name.startswith("__") or _name == "__all__":
        globals()[_name] = _value

_DIVISIVE_SENTINEL = -2147483000


def _parse_reliance_mode(value):
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    if key in ("subtractive", "difference", "diff"):
        return False
    if key in ("divisive", "ratio"):
        return True
    raise ValueError("model_reliance must be 'subtractive' or 'divisive'.")


def _wrap_rid_method(method):
    signature = _inspect.signature(method)

    @_functools.wraps(method)
    def wrapped(self, *args, model_reliance="subtractive", **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        divisive = _parse_reliance_mode(model_reliance)

        if divisive:
            if int(bound.arguments["importance_interval_mode"]) != 1:
                raise ValueError(
                    "Divisive model reliance is supported only with importance_interval_mode=1."
                )
            if not bool(bound.arguments["lossless"]):
                raise ValueError("Divisive model reliance requires lossless=True.")
            bound.arguments["n_scramble_evals"] = _DIVISIVE_SENTINEL

        return method(*bound.args, **bound.kwargs)

    parameters = list(signature.parameters.values())
    parameters.append(
        _inspect.Parameter(
            "model_reliance",
            kind=_inspect.Parameter.KEYWORD_ONLY,
            default="subtractive",
        )
    )
    wrapped.__signature__ = signature.replace(parameters=parameters)
    return wrapped


ArborEnum.compute_rid = _wrap_rid_method(ArborEnum.compute_rid)
ArborEnum.compute_rid_binarized = _wrap_rid_method(ArborEnum.compute_rid_binarized)
ArborEnum.compute_rid_continuous_low_level = _wrap_rid_method(
    ArborEnum.compute_rid_continuous_low_level
)
ArborEnum.__module__ = __name__
