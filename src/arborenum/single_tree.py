from . import ArborEnum, ArborEnumRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class ArborEnumTree:
    _estimator_class = ArborEnum
    # Fit one decision tree using ArborEnum's single-tree solver.
    # depth_budget : int, default=5, Maximum decision tree depth.
    # lambda_reg : float, default=0.005, lower values include more complex trees
    # proxy_mode : {'continuous', 'hybrid', 'binarized'}, default='hybrid', how continuous split candidates are handled by the proxy.
    # lookahead_k : int, default=1, higher lookahead means better optimization. lookahead depth_budget-1 is optimal.
    def __init__(
        self,
        depth_budget=5,
        lambda_reg=0.005,
        proxy_mode="hybrid",
        lookahead_k=1,
        heuristic_for_greedy=1,
        *,
        eta_defer=0.0,
        use_deferral=False,
        **kwargs,
    ):
        if proxy_mode not in ("continuous", "hybrid", "binarized"):
            raise ValueError("proxy_mode must be 'continuous', 'hybrid', or 'binarized'.")
        if "rashomon_mode" in kwargs:
            raise ValueError("ArborEnumTree always uses rashomon_mode=False.")
        for name in ("depth_budget", "lambda_reg", "proxy_mode", "lookahead_k",
                     "heuristic_for_greedy", "eta_defer", "use_deferral"):
            if name in kwargs:
                raise ValueError(f"Specify {name} only once.")
        self.depth_budget = depth_budget
        self.lambda_reg = lambda_reg
        self.proxy_mode = proxy_mode
        self.lookahead_k = lookahead_k
        self.heuristic_for_greedy = heuristic_for_greedy
        self.eta_defer = eta_defer
        self.use_deferral = use_deferral
        self._extra_fit_kwargs = dict(kwargs)
        self._model = None

    def _make_estimator(self):
        return self._estimator_class()

    def fit(self, X, y, *, bb_pred=None, X_proxy=None):
        model = self._make_estimator()
        options = dict(self._extra_fit_kwargs)
        options.update(
            depth_budget=self.depth_budget,
            lambda_reg=self.lambda_reg,
            proxy_mode=self.proxy_mode,
            lookahead_k=self.lookahead_k,
            heuristic_for_greedy=self.heuristic_for_greedy,
            eta_defer=self.eta_defer,
            use_deferral=self.use_deferral,
            rashomon_mode=False,
        )
        if bb_pred is not None:
            options["bb_pred"] = bb_pred
        if X_proxy is not None:
            options["X_proxy"] = X_proxy
        model.fit(X, y, **options)
        self._model = model
        return self

    def _fitted(self):
        if self._model is None:
            raise RuntimeError("Call fit(X, y) before accessing the tree.")
        return self._model

    def get_tree(self):
        return self._fitted().get_tree_paths(0)

    def get_tree_paths(self):
        """Alias for get_tree()."""
        return self.get_tree()

    def get_tree_paths_str(self):
        return self._fitted().get_tree_paths_str(0)

    def get_objective(self):
        # return (raw_objective, normalized_objective)
        return self._fitted().get_tree_objective(0)

    def get_num_leaves(self):
        return self._fitted().get_tree_num_leaves(0)

    def plot_tree(self, feature_names=None, figsize=(8, 6), ax=None,
                  title=None, show=True):
        return self._fitted().plot_tree(
            0, feature_names=feature_names, figsize=figsize,
            ax=ax, title=title, show=show,
        )

    @property
    def arborenum_(self):
        return self._fitted()


class ArborEnumTreeRegressor(RegressorMixin, ArborEnumTree, BaseEstimator):
    _estimator_class = ArborEnumRegressor

    def __init__(self, depth_budget=5, lambda_reg=.005, proxy_mode="hybrid", lookahead_k=1,
                 heuristic_for_greedy=1, *, eta_defer=0., use_deferral=False,
                 equivalent_points=False, continuous_kmeans_bound=False,
                 kmeans_min_remaining_depth=2, movement_bound_mode="local", movement_cache_mb=128, **kwargs):
        super().__init__(depth_budget, lambda_reg, proxy_mode, lookahead_k, heuristic_for_greedy,
                         eta_defer=eta_defer, use_deferral=use_deferral, **kwargs)
        self.equivalent_points = bool(equivalent_points)
        self.continuous_kmeans_bound = bool(continuous_kmeans_bound)
        self.kmeans_min_remaining_depth = kmeans_min_remaining_depth
        self.movement_bound_mode = movement_bound_mode
        self.movement_cache_mb = movement_cache_mb

    def get_params(self, deep=True):
        names = ("depth_budget", "lambda_reg", "proxy_mode", "lookahead_k", "heuristic_for_greedy",
                 "eta_defer", "use_deferral", "equivalent_points", "continuous_kmeans_bound",
                 "kmeans_min_remaining_depth", "movement_bound_mode", "movement_cache_mb")
        return {**self._extra_fit_kwargs, **{name: getattr(self, name) for name in names}}

    def set_params(self, **params):
        valid = self.get_params()
        for name, value in params.items():
            if name not in valid:
                raise ValueError(f"Invalid parameter {name!r} for ArborEnumTreeRegressor.")
            if name in self._extra_fit_kwargs:
                self._extra_fit_kwargs[name] = value
            else:
                setattr(self, name, value)
        return self

    def fit(self, X, y, *, bb_pred=None, X_proxy=None):
        super().fit(X, y, bb_pred=bb_pred, X_proxy=X_proxy)
        self.n_features_in_ = self._fitted().n_features_in_
        self.feature_names_in_ = self._fitted().feature_names_in_
        return self

    def _make_estimator(self):
        return self._estimator_class(equivalent_points=self.equivalent_points,
                                     continuous_kmeans_bound=self.continuous_kmeans_bound,
                                     kmeans_min_remaining_depth=self.kmeans_min_remaining_depth,
                                     movement_bound_mode=self.movement_bound_mode,
                                     movement_cache_mb=self.movement_cache_mb)

    def get_movement_bound_stats(self):
        return self._fitted().get_movement_bound_stats()

    def transform(self, X):
        return self._fitted().transform(X)

    def get_internal_feature_info(self, internal_feature):
        return self._fitted().get_internal_feature_info(internal_feature)

    def get_continuous_cutpoints(self, continuous_group):
        return self._fitted().get_continuous_cutpoints(continuous_group)

    def permutation_importance_bounds(self, X_eval, y_eval, *, budget_override=None):
        return self._fitted().permutation_importance_bounds(X_eval, y_eval, budget_override=budget_override)

    def get_exact_replacement_importance_intervals(self, X_eval, y_eval, budget_override=None):
        return self._fitted().get_exact_replacement_importance_intervals(X_eval, y_eval, budget_override)

    def get_exact_replacement_divisive_importance_intervals(self, X_eval, y_eval, budget_override=None):
        return self._fitted()._single_tree_importance(X_eval, y_eval, budget_override, True)

    def predict(self, X, *, bb_pred=None):
        return self._fitted().get_predictions(0, X, bb_pred=bb_pred)

    def score(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))
