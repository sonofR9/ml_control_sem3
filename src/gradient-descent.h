#pragma once

#include "runge-kutte.h"

#include <concepts>

namespace optimization {
template <typename F, int N>
concept GradientFunction = requires(F func, const Vector<N>& inp) {
  { func(inp) } -> std::same_as<Vector<N>>;
};

template <typename F, int N>
concept Regular1OutFunction = requires(F func, const Vector<N>& inp) {
  { func(inp) } -> std::same_as<double>;
};

template <int N, GradientFunction<N> G>
Vector<N> GradientDescent(const Vector<N>& qMin, const Vector<N>& qMax, G grad,
                          double ksi = 0.5) {
  const auto& qStar{ksi * (qMin + qMax)};
  const auto& gradStar{grad(qStar)};

  const auto golden = [qMin, qMax](const double kg, const Vector<N>& q,
                                   const Vector<N>& grad) -> Vector<N> {
    Vector<N> result;
    for (int i{0}; i < N; ++i) {
      const auto dq{q[i] - kg * grad[i]};
      if (dq > qMax) {
        result[i] = qMax;
      } else if (dq < qMin) {
        result[i] = qMin;
      } else {
        result[i] = dq;
      }
      return result;
    }
  };

  Vector<N> qWave{golden};

  Vector<N> q1;
  Vector<N> q2;
}

template <int N, Regular1OutFunction<N> G>
Vector<N> GradientDescent(const Vector<N>& qMin, const Vector<N>& qMax, G grad,
                          double ksi = 0.5) {
  auto q1{ksi * (qMin + qMax)};
  auto grad1{grad(q1)};
}

}  // namespace optimization
