import numpy as np
from arborenum import ArborEnumRegressor, ArborEnumTreeRegressor

rng = np.random.default_rng(7)
X = rng.normal(size=(80, 3))
y = 3 * (X[:, 0] > 0.2) - 2 * X[:, 1] + rng.normal(scale=0.2, size=80)

model = ArborEnumRegressor(equivalent_points=True, continuous_kmeans_bound=True)
model.fit(X, y, depth_budget=3, lambda_reg=0.01, lookahead_k=1,
          proxy_mode="continuous", greedy_continuous_mode="numerical",
          early_stopping=True, runtime_limit_seconds=5,
          rashomon_mult=0.05, max_refinement_rounds=3)
print(model.count_trees())
print(model.get_tree_objective(0))
print(model.predict(X[:5]))

tree = ArborEnumTreeRegressor(depth_budget=3, lambda_reg=0.01,
                              proxy_mode="continuous", lookahead_k=1,
                              greedy_continuous_mode="numerical").fit(X, y)
print(tree.get_objective())
print(tree.predict(X[:5]))
