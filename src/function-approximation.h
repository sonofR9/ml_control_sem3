#pragma once

#include "global.h"

#include <map>

namespace optimization {
template <int N>
class PiecewiseLinearApproximation {
 public:
  template <int T>
  PiecewiseLinearApproximation(double dt, Vector<T> points) {
    double t{0};
    for (int i = 0; i < N; ++i) {
      points_[t] = points[i];
      t += dt;
    }
  }

  Vector<N> operator()(double time) const {
    const auto& lb{points_.lower_bound(time)};
    if (lb == points_.end()) {
      return lb->second;
    }
    return (lb->second + (++lb)->second) / (lb->first - (--lb)->first) *
           (time - lb->first);
  }

  void insert(double time, const Vector<N>& point) {
    points_.insert({time, point});
  }

 private:
  std::map<double, Vector<N>> points_;
};
}  // namespace optimization
