#pragma once

#include "tensor.h"

#include <map>

namespace optimization {
template <typename T, typename U>
concept TensorIterator = std::is_same_v<typename T::value_type, Tensor<U>>;

template <typename T, typename U>
concept TimeAndTensorIterator =
    std::is_same_v<typename T::value_type, std::pair<double, Tensor<U>>>;

template <typename U = double>
class PiecewiseLinearApproximation {
 public:
  template <TensorIterator<U> It>
  PiecewiseLinearApproximation(double dt, const It& begin, const It& end) {
    double t{0};
    It curr{begin};
    while (curr != end) {
      points_.insert({t, *curr});
      t += dt;
      ++curr;
    }
  }

  Tensor<U> operator()(double time) const {
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

  void insert(double time, const Tensor<U>& point) {
    points_.insert({time, point});
  }

 private:
  std::map<double, Tensor<U>> points_;
};
}  // namespace optimization
