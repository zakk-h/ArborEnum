from . import ArborEnum


class ArborEnumTree:
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

    def fit(self, X, y, *, bb_pred=None, X_proxy=None):
        model = ArborEnum()
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
