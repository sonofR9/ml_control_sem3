#pragma once

#include "tensor.h"

#include <map>

namespace optimization {
template <typename T, typename U, class Alloc>
concept TensorIterator =
    std::is_same_v<typename T::value_type, Tensor<U, Alloc>>;

template <typename T, typename U, class Alloc>
concept TimeAndTensorIterator =
    std::is_same_v<typename T::value_type, std::pair<double, Tensor<U, Alloc>>>;

template <typename U = double, class Alloc = std::allocator<U>>
class PiecewiseLinearApproximation {
 public:
  // TODO(novak) Ignored: can't infer Alloc2 for concept
  template <typename Alloc2, TensorIterator<U, Alloc2> It>
  PiecewiseLinearApproximation(double dt, const It& begin, const It& end) {
    double t{0};
    It curr{begin};
    while (curr != end) {
      points_.insert({t, *curr});
      t += dt;
      ++curr;
    }
  }

  template <TensorIterator<U, Alloc> It>
  PiecewiseLinearApproximation(double dt, const It& begin, const It& end) {
    // PiecewiseLinearApproximation<Alloc, It>(dt, begin, end);
    double t{0};
    It curr{begin};
    while (curr != end) {
      points_.insert({t, *curr});
      t += dt;
      ++curr;
    }
  }

  Tensor<U, Alloc> operator()(double time) const {
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
      const double partOfdt{1.0 / (next->first - lb->first) *
                            (time - lb->first)};
      return lb->second + (next->second - lb->second) * partOfdt;
    }

    const double partOfdt{1.0 / (next->first - lb->first) * (time - lb->first)};
    return lb->second + (next->second - lb->second) * partOfdt;
  }

  void insert(double time, const Tensor<U, Alloc>& point) {
    points_.insert({time, point});
  }

 private:
  std::map<double, Tensor<U, Alloc>> points_;
};
}  // namespace optimization
