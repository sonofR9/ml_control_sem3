#pragma once

#include "global.h"

#include <map>

namespace optimization {
template <typename T, int N, typename U>
concept VectorIterator = requires(const T& it) {
                           { *it } -> std::same_as<Vector<N, U>&>;
                         };

template <typename T, int N, typename U>
concept TimeAndVectorIterator =
    requires(const T& it) {
      { *it } -> std::same_as<std::pair<double, Vector<N>>&>;
    };

template <int N, typename U = double>
class PiecewiseLinearApproximation {
 public:
  template <VectorIterator<N, U> It>
  PiecewiseLinearApproximation(double dt, const It& begin, const It& end) {
    double t{0};
    It curr{begin};
    while (curr != end) {
      points_.insert({t, *curr});
      t += dt;
      ++curr;
    }
  }

  template <TimeAndVectorIterator<N, U> It>
  PiecewiseLinearApproximation(const It& begin, const It& end) {
    It curr{begin};
    while (curr != end) {
      points_.insert(*curr);
      ++curr;
    }
  }

  Vector<N, U> operator()(double time) const {
    auto lb{points_.lower_bound(time)};
    if (lb == points_.end()) {
      return lb->second;
    }
    return (lb->second + (++lb)->second) / (lb->first - (--lb)->first) *
           (time - lb->first);
  }

  void insert(double time, const Vector<N, U>& point) {
    points_.insert({time, point});
  }

 private:
  std::map<double, Vector<N, U>> points_;
};
}  // namespace optimization
