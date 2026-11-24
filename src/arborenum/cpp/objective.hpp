#pragma once

#if defined(ARBORENUM_REGRESSION)
static constexpr bool regression_objective = true;
using Objective = double;
using Prediction = double;
using OutputPrediction = double;
#else
static constexpr bool regression_objective = false;
using Objective = int;
using Prediction = int;
using OutputPrediction = uint8_t;
#endif

static inline Objective rounded_objective(double value) {
    if constexpr (regression_objective) {
        if (!std::isfinite(value) || value < 0) throw std::invalid_argument("Objective parameters must be finite and nonnegative.");
        return value;
    }
    else return static_cast<int>(std::llround(value));
}

static inline Objective objective_remainder(Objective budget, Objective other) {
    if constexpr (regression_objective) {
        const double upper = std::nextafter(budget, std::numeric_limits<double>::infinity());
        return std::nextafter(upper - other, std::numeric_limits<double>::infinity());
    } else return budget - other;
}
