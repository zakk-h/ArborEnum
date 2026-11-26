import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from . import ArborEnum, ArborEnumRegressor, _resolve_proxy_matrix
from .single_tree import ArborEnumTree


class ArborEnumLogisticClassifier(ClassifierMixin, ArborEnumRegressor):
    @staticmethod
    def _new_core():
        from ._logistic_core import ArborEnum as LogisticCore
        return LogisticCore()

    def __init__(self, *, leaf_logit_clip=30.0, newton_max_iter=50, newton_tol=1e-10):
        self.leaf_logit_clip = leaf_logit_clip
        self.newton_max_iter = newton_max_iter
        self.newton_tol = newton_tol
        self._validate_parameters()
        ArborEnum.__init__(self)
        self._logistic_fitting_ = False
        self._logistic_native_parameters_ = None

    def _validate_parameters(self):
        if not np.isscalar(self.leaf_logit_clip) or not np.isfinite(self.leaf_logit_clip) or self.leaf_logit_clip <= 0:
            raise ValueError("leaf_logit_clip must be finite and positive.")
        if isinstance(self.newton_max_iter, (bool, np.bool_)) or not isinstance(self.newton_max_iter, (int, np.integer)) or self.newton_max_iter < 1:
            raise ValueError("newton_max_iter must be a positive integer.")
        if not np.isscalar(self.newton_tol) or not np.isfinite(self.newton_tol) or self.newton_tol <= 0:
            raise ValueError("newton_tol must be finite and positive.")

    def set_params(self, **params):
        BaseEstimator.set_params(self, **params)
        self._validate_parameters()
        return self

    def _resolve_proxy_matrix(self, *args, **kwargs):
        X = kwargs.get("X_original", args[0] if args else None)
        y = kwargs.get("y", args[1] if len(args) > 1 else None)
        if X.shape[1] == 0 or np.all(y == y[0]):
            proxy = kwargs.get("X_proxy")
            return np.empty((len(y), 0), dtype=np.uint8) if proxy is None else np.asarray(proxy, dtype=np.uint8)
        return _resolve_proxy_matrix(*args, task="classification", **kwargs)

    def _logistic_fit(self, method, args, kwargs, base_logits, X=None, y=None, prepared=False):
        active = self._logistic_fitting_
        if not active:
            self._validate_parameters()
            if y is not None:
                labels = np.asarray(y, dtype=float)
                if labels.ndim != 1 or not labels.size or not np.all((labels == 0) | (labels == 1)):
                    raise ValueError("Logistic targets must be a nonempty vector containing only zero and one.")
                self._logistic_training_y_ = labels.copy()
                self._logistic_training_X_ = X
                offsets = np.empty(0, dtype=float) if base_logits is None else np.asarray(base_logits, dtype=float)
            else:
                if not hasattr(self, "_logistic_training_y_"):
                    raise RuntimeError("Prepare training data before fitting a prepared problem.")
                labels = self._logistic_training_y_
                offsets = self.base_logits_ if base_logits is None else np.asarray(base_logits, dtype=float)
            if offsets.ndim != 1 or (offsets.size and offsets.size != labels.size) or not np.all(np.isfinite(offsets)):
                raise ValueError("base_logits must be a finite vector with one value per training observation.")
            parameters = (float(self.leaf_logit_clip), int(self.newton_max_iter), float(self.newton_tol))
            if not prepared or base_logits is not None or parameters != self._logistic_native_parameters_:
                self.base_logits_ = offsets.copy()
                self._model.set_logistic_options(offsets, *parameters)
                self._logistic_native_parameters_ = parameters
            self.classes_ = np.asarray([0, 1], dtype=np.int32)
        if kwargs.get("use_deferral", False):
            raise ValueError("Logistic deferral is not supported.")
        if method != "prepare_continuous_data":
            kwargs["greedy_continuous_mode"] = "numerical"
        self._logistic_fitting_ = True
        try:
            return getattr(super(), method)(*args, **kwargs)
        finally:
            self._logistic_fitting_ = active

    def fit(self, X, y, *, base_logits=None, **kwargs):
        return self._logistic_fit("fit", (X, y), kwargs, base_logits, X=X, y=y)

    def fit_binarized(self, X, y, *, base_logits=None, **kwargs):
        return self._logistic_fit("fit_binarized", (X, y), kwargs, base_logits, X=X, y=y)

    def fit_binarized_repeated_subsamples(self, X, y, *, base_logits=None, **kwargs):
        return self._logistic_fit("fit_binarized_repeated_subsamples", (X, y), kwargs, base_logits, X=X, y=y)

    def fit_then_extend(self, X, y, *, base_logits=None, **kwargs):
        return self._logistic_fit("fit_then_extend", (X, y), kwargs, base_logits, X=X, y=y)

    def fit_anytime(self, X, y, *, base_logits=None, **kwargs):
        return self._logistic_fit("fit_anytime", (X, y), kwargs, base_logits, X=X, y=y)

    def prepare_continuous_data(self, X_num, y, *, base_logits=None, **kwargs):
        return self._logistic_fit("prepare_continuous_data", (X_num, y), kwargs, base_logits, X=X_num, y=y)

    def fit_prepared(self, *, base_logits=None, **kwargs):
        return self._logistic_fit("fit_prepared", (), kwargs, base_logits, prepared=True)

    def fit_prepared_repeated_subsamples(self, *, base_logits=None, **kwargs):
        return self._logistic_fit("fit_prepared_repeated_subsamples", (), kwargs, base_logits, prepared=True)

    def fit_prepared_then_extend(self, *, base_logits=None, **kwargs):
        return self._logistic_fit("fit_prepared_then_extend", (), kwargs, base_logits, prepared=True)

    def fit_prepared_anytime(self, *, base_logits=None, **kwargs):
        return self._logistic_fit("fit_prepared_anytime", (), kwargs, base_logits, prepared=True)

    def count_trees_within_mult(self, mult):
        if not np.isfinite(mult) or mult < 0:
            raise ValueError("mult must be finite and nonnegative.")
        if getattr(self, "_single_tree_mode_", False):
            return 1
        threshold = self.get_min_objective() * (1 + mult)
        depth = self.depth_budget_ + getattr(self, "_proxy_delta_value", 0)
        relative = 16 * np.finfo(float).eps * (depth + 2) * (1 + np.log1p(self.root_n_))
        threshold = np.nextafter(threshold + relative * abs(threshold), np.inf)
        return sum(int(count) for objective, count in self.get_root_histogram() if objective <= threshold)

    def _prediction_offsets(self, X, base_logits):
        n = len(X)
        if base_logits is None:
            offsets = getattr(self, "base_logits_", np.empty(0))
            if not offsets.size:
                return 0.0
            if np.all(offsets == offsets[0]):
                return float(offsets[0])
            if X is self._logistic_training_X_:
                return offsets
            raise ValueError("Provide base_logits for predictions with observation-specific base logits.")
        offsets = np.asarray(base_logits, dtype=float)
        if offsets.ndim != 1 or offsets.size != n or not np.all(np.isfinite(offsets)):
            raise ValueError("base_logits must be a finite vector matching X rows.")
        return offsets

    def get_corrections(self, tree_index, X):
        return ArborEnum.get_predictions(self, tree_index, X)

    def get_decision_function(self, tree_index, X, *, base_logits=None):
        return self.get_corrections(tree_index, X) + self._prediction_offsets(X, base_logits)

    @staticmethod
    def _probabilities(logits):
        logits = np.asarray(logits, dtype=float)
        exponential = np.exp(-np.abs(logits))
        probabilities = np.where(logits >= 0, 1 / (1 + exponential), exponential / (1 + exponential))
        return np.stack((1 - probabilities, probabilities), axis=-1)

    def get_predict_proba(self, tree_index, X, *, base_logits=None):
        return self._probabilities(self.get_decision_function(tree_index, X, base_logits=base_logits))

    def get_predictions(self, tree_index, X, *, base_logits=None):
        return (self.get_decision_function(tree_index, X, base_logits=base_logits) >= 0).astype(np.int32)

    def get_all_predictions(self, X, *, base_logits=None, stack=False):
        corrections = ArborEnum.get_all_predictions(self, X, stack=stack)
        return (corrections + self._prediction_offsets(X, base_logits) >= 0).astype(np.int32)

    def predict(self, X, *, base_logits=None, tree_index=0):
        return self.get_predictions(tree_index, X, base_logits=base_logits)

    def predict_proba(self, X, *, base_logits=None, tree_index=0):
        return self.get_predict_proba(tree_index, X, base_logits=base_logits)

    def decision_function(self, X, *, base_logits=None, tree_index=0):
        return self.get_decision_function(tree_index, X, base_logits=base_logits)

    def score(self, X, y, *, base_logits=None, sample_weight=None):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X, base_logits=base_logits), sample_weight=sample_weight)

    def log_loss(self, X, y, *, base_logits=None, tree_index=0):
        labels = np.asarray(y, dtype=float)
        logits = self.get_decision_function(tree_index, X, base_logits=base_logits)
        if labels.shape != logits.shape or not np.all((labels == 0) | (labels == 1)):
            raise ValueError("y must contain one binary target per observation.")
        return float(np.mean(np.logaddexp(0, (1 - 2 * labels) * logits)))

    def _logistic_importance(self, method, X_eval, y_eval, budget_override, base_logits):
        labels = np.asarray(y_eval, dtype=float)
        if labels.ndim != 1 or not labels.size or labels.size != len(X_eval) or not np.all((labels == 0) | (labels == 1)):
            raise ValueError("y_eval must be a nonempty binary vector matching X_eval rows.")
        if self.n_features_in_ is None:
            raise RuntimeError("Fit the model first.")
        if budget_override is not None:
            if not np.isfinite(budget_override):
                raise ValueError("budget_override must be finite.")
            if budget_override >= 0 and self.get_min_objective() > np.nextafter(float(budget_override), np.inf):
                raise ValueError("budget_override excludes every represented tree.")
        offsets = self._prediction_offsets(X_eval, base_logits)
        if np.ndim(offsets) == 0:
            offsets = np.full(labels.size, offsets, dtype=float)
        self._model.set_logistic_evaluation_offsets(offsets)
        try:
            return getattr(ArborEnum, method)(self, X_eval, labels, budget_override)
        finally:
            self._model.set_logistic_evaluation_offsets([])

    def get_exact_replacement_importance_intervals(self, X_eval, y_eval, budget_override=None, *, base_logits=None):
        return self._logistic_importance("get_exact_replacement_importance_intervals", X_eval, y_eval, budget_override, base_logits)

    def get_exact_replacement_divisive_importance_intervals(self, X_eval, y_eval, budget_override=None, *, base_logits=None):
        return self._logistic_importance("get_exact_replacement_divisive_importance_intervals", X_eval, y_eval, budget_override, base_logits)

    def permutation_importance_bounds(self, X_eval, y_eval, *, budget_override=None, base_logits=None):
        import pandas as pd
        bounds = self.get_exact_replacement_importance_intervals(X_eval, y_eval, budget_override, base_logits=base_logits)
        return pd.DataFrame(bounds, index=pd.Index(self.feature_names_in_, name="feature", tupleize_cols=False),
                            columns=["min_importance", "max_importance"])

    def _logistic_rid(self, method, args, kwargs, base_logits):
        from functools import partial
        from inspect import signature
        from . import _parse_model_reliance
        from ._logistic_core import rid_subtractive_model_reliance, rid_subtractive_model_reliance_continuous
        self._validate_parameters()
        function = getattr(ArborEnum, method)
        bound = signature(function).bind(self, *args, **kwargs)
        labels = np.asarray(args[1], dtype=float)
        if labels.ndim != 1 or not labels.size or len(args[0]) != labels.size or not np.all((labels == 0) | (labels == 1)):
            raise ValueError("RID targets must be a nonempty binary vector matching X rows.")
        offsets = np.empty(0, dtype=float) if base_logits is None else np.asarray(base_logits, dtype=float)
        if offsets.ndim != 1 or (offsets.size and offsets.size != labels.size) or not np.all(np.isfinite(offsets)):
            raise ValueError("base_logits must be a finite vector matching RID rows.")
        if bound.arguments.get("use_deferral", False):
            raise ValueError("Logistic deferral is not supported.")
        if method != "compute_rid_binarized":
            bound.arguments["greedy_continuous_mode"] = "numerical"
        options = dict(base_logits=offsets, leaf_logit_clip=self.leaf_logit_clip,
                       newton_max_iter=self.newton_max_iter, newton_tol=self.newton_tol)
        previous = self._rid_core, self._rid_continuous_core
        self._rid_core = partial(rid_subtractive_model_reliance, **options)
        self._rid_continuous_core = partial(rid_subtractive_model_reliance_continuous, **options)
        try:
            result = function(*bound.args, **bound.kwargs)
        finally:
            self._rid_core, self._rid_continuous_core = previous
        result["loss"] = "log_loss"
        result["leaf_values"] = "fitted"
        result["base_logits_permutation"] = "fixed"
        result["model_reliance"] = "divisive" if _parse_model_reliance(bound.arguments.get("model_reliance", "subtractive")) else "subtractive"
        return result

    def compute_rid(self, X, y, *, base_logits=None, **kwargs):
        return self._logistic_rid("compute_rid", (X, y), kwargs, base_logits)

    def compute_rid_binarized(self, X, y, *args, base_logits=None, **kwargs):
        return self._logistic_rid("compute_rid_binarized", (X, y, *args), kwargs, base_logits)

    def compute_rid_continuous_low_level(self, X_num, y, *args, base_logits=None, **kwargs):
        return self._logistic_rid("compute_rid_continuous_low_level", (X_num, y, *args), kwargs, base_logits)

    def _unsupported_logistic_analysis(self, *args, **kwargs):
        raise NotImplementedError("This analysis is not implemented for logistic loss.")

    get_wald_divisive_model_reliance_intervals = _unsupported_logistic_analysis

    def save_builder_payload(self, path, *, X=None, y=None, base_logits=None, class_names=None, **kwargs):
        import json
        if (X is None) != (y is None):
            raise ValueError("Pass both X and y to embed an analysis dataset.")
        offsets = None
        if X is not None:
            labels = np.asarray(y, dtype=float)
            if labels.ndim != 1 or len(labels) != len(X) or not np.all((labels == 0) | (labels == 1)):
                raise ValueError("Logistic analysis targets must be binary and match X rows.")
            offsets = np.asarray(self._prediction_offsets(X, base_logits), dtype=float)
            if offsets.ndim == 0:
                offsets = np.full(len(X), offsets, dtype=float)
        elif base_logits is not None:
            raise ValueError("Pass X and y when embedding base logits.")
        names = [str(name) for name in (class_names if class_names is not None else self.classes_)]
        if len(names) != 2 or len(set(names)) != 2:
            raise ValueError("Logistic graphs require two distinct class names.")
        path = ArborEnum.save_builder_payload(self, path, X=X, y=y, **kwargs)
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload["meta"]
        meta.update(task="logistic", loss="log_loss", leaf_values="logit_correction",
                    leaf_logit_clip=self.leaf_logit_clip, classNames=names,
                    base_logits_permutation="fixed")
        if offsets is not None:
            meta["analysisDataset"].update(class_names=names, base_logits=offsets.tolist())
        path.write_text(json.dumps(payload, indent=kwargs.get("indent", 2)), encoding="utf-8")
        return path


class ArborEnumTreeLogisticClassifier(ClassifierMixin, ArborEnumTree, BaseEstimator):
    _estimator_class = ArborEnumLogisticClassifier

    def __init__(self, depth_budget=5, lambda_reg=0.005, proxy_mode="hybrid", lookahead_k=1,
                 heuristic_for_greedy=1, *, leaf_logit_clip=30.0, newton_max_iter=50, newton_tol=1e-10, **kwargs):
        super().__init__(depth_budget=depth_budget, lambda_reg=lambda_reg, proxy_mode=proxy_mode,
                         lookahead_k=lookahead_k, heuristic_for_greedy=heuristic_for_greedy, **kwargs)
        self.leaf_logit_clip = leaf_logit_clip
        self.newton_max_iter = newton_max_iter
        self.newton_tol = newton_tol

    def get_params(self, deep=True):
        names = ("depth_budget", "lambda_reg", "proxy_mode", "lookahead_k", "heuristic_for_greedy",
                 "leaf_logit_clip", "newton_max_iter", "newton_tol")
        return {**self._extra_fit_kwargs, **{name: getattr(self, name) for name in names}}

    def set_params(self, **params):
        valid = self.get_params()
        for name, value in params.items():
            if name not in valid:
                raise ValueError(f"Invalid parameter {name!r} for ArborEnumTreeLogisticClassifier.")
            if name in self._extra_fit_kwargs:
                self._extra_fit_kwargs[name] = value
            else:
                setattr(self, name, value)
        return self

    def _make_estimator(self):
        return self._estimator_class(leaf_logit_clip=self.leaf_logit_clip,
                                    newton_max_iter=self.newton_max_iter, newton_tol=self.newton_tol)

    def fit(self, X, y, *, base_logits=None, X_proxy=None):
        model = self._make_estimator()
        options = dict(self._extra_fit_kwargs, depth_budget=self.depth_budget, lambda_reg=self.lambda_reg,
                       proxy_mode=self.proxy_mode, lookahead_k=self.lookahead_k,
                       heuristic_for_greedy=self.heuristic_for_greedy, rashomon_mode=False,
                       use_deferral=self.use_deferral, eta_defer=self.eta_defer)
        model.fit(X, y, base_logits=base_logits, X_proxy=X_proxy, **options)
        self._model = model
        self.classes_ = model.classes_
        self.n_features_in_ = model.n_features_in_
        self.feature_names_in_ = model.feature_names_in_
        return self

    def predict(self, X, *, base_logits=None):
        return self._fitted().predict(X, base_logits=base_logits)

    def predict_proba(self, X, *, base_logits=None):
        return self._fitted().predict_proba(X, base_logits=base_logits)

    def decision_function(self, X, *, base_logits=None):
        return self._fitted().decision_function(X, base_logits=base_logits)

    def get_corrections(self, X):
        return self._fitted().get_corrections(0, X)

    def log_loss(self, X, y, *, base_logits=None):
        return self._fitted().log_loss(X, y, base_logits=base_logits)

    def get_exact_replacement_importance_intervals(self, X_eval, y_eval, budget_override=None, *, base_logits=None):
        return self._fitted().get_exact_replacement_importance_intervals(X_eval, y_eval, budget_override, base_logits=base_logits)

    def get_exact_replacement_divisive_importance_intervals(self, X_eval, y_eval, budget_override=None, *, base_logits=None):
        return self._fitted().get_exact_replacement_divisive_importance_intervals(X_eval, y_eval, budget_override, base_logits=base_logits)

    def permutation_importance_bounds(self, X_eval, y_eval, *, budget_override=None, base_logits=None):
        return self._fitted().permutation_importance_bounds(X_eval, y_eval, budget_override=budget_override, base_logits=base_logits)
