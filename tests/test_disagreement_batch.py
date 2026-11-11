import numpy as np

from arborenum import ArborEnum


def test_budgeted_disagreement_batches():
    rng = np.random.default_rng(7)
    X = rng.integers(0, 2, size=(24, 5), dtype=np.uint8)
    y = ((X[:, 0] ^ X[:, 1]) | X[:, 2]).astype(np.int32)

    model = ArborEnum()
    model.fit(
        X,
        y,
        proxy_mode="continuous",
        lambda_reg=0.01,
        depth_budget=3,
        rashomon_mult=0.5,
        lookahead_k=1,
    )

    X_eval = np.concatenate([X, X[:7], X[::-1]], axis=0)
    for budget in (None, 0):
        batched = np.asarray(model.has_disagreement(X=X_eval, budget=budget), dtype=bool)
        individual = np.asarray(
            [model.has_disagreement(X=row, budget=budget) for row in X_eval],
            dtype=bool,
        )
        np.testing.assert_array_equal(batched, individual)

        if budget is None:
            training = np.asarray(
                [model.has_disagreement(sample_idx=i) for i in range(len(X))],
                dtype=bool,
            )
            np.testing.assert_array_equal(batched[:len(X)], training)


if __name__ == "__main__":
    test_budgeted_disagreement_batches()
