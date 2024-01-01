#pragma once

#include "global.h"

#include <iostream>
#include <map>

namespace optimization {
template <typename T, int N, typename U>
concept StaticTensorIterator = requires(const T& it) {
  { *it } -> std::same_as<StaticTensor<N, U>&>;
};

template <typename T, int N, typename U>
concept TimeAndStaticTensorIterator = requires(const T& it) {
  { *it } -> std::same_as<std::pair<double, StaticTensor<N>>&>;
};

template <int N, typename U = double>
class PiecewiseLinearApproximation {
 public:
  template <StaticTensorIterator<N, U> It>
  PiecewiseLinearApproximation(double dt, const It& begin, const It& end) {
    double t{0};
    It curr{begin};
    while (curr != end) {
      points_.insert({t, *curr});
      t += dt;
      ++curr;
    }
  }

  StaticTensor<N, U> operator()(double time) const {
    auto lb{points_.lower_bound(time)};
    auto next{lb};
    if (next != points_.end()) {
      ++next;
    }
    if (lb == points_.end() || next == points_.end()) {
      // can't make set iterators work((
      auto tmp = points_.end();
      lb = --tmp;
      next = lb--;
      return lb->second + (next->second - lb->second) /
                              (next->first - lb->first) * (time - lb->first);
    }

    const double partOfdt{1.0 / (next->first - lb->first) * (time - lb->first)};
    return lb->second + (next->second - lb->second) * partOfdt;
  }

  void insert(double time, const StaticTensor<N, U>& point) {
    points_.insert({time, point});
  }

 private:
  std::map<double, StaticTensor<N, U>> points_;
};
}  // namespace optimization
