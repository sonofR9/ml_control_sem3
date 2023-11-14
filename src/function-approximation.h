#pragma once

#include "global.h"

#include <map>

namespace optimization {
template <typename T, int N>
concept VectorIterator = requires(const T& it) {
  { *it } -> std::same_as<Vector<N>>;
};

template <typename T, int N>
concept TimeAndVectorIterator = requires(const T& it) {
  { *it } -> std::same_as<std::pair<double, Vector<N>>>;
};

template <int N>
class PiecewiseLinearApproximation {
 public:
  template <VectorIterator<N> It>
  PiecewiseLinearApproximation(double dt, const It& begin, const It& end) {
    double t{0};
    It curr{begin};
    while (curr != end) {
      points_.insert({t, *curr});
      t += dt;
      ++curr;
    }
  }

  template <TimeAndVectorIterator<N> It>
  PiecewiseLinearApproximation(const It& begin, const It& end) {
    It curr{begin};
    while (curr != end) {
      points_.insert(*curr);
      ++curr;
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
