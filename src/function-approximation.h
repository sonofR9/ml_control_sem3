#pragma once

#include "tensor.h"

#include <algorithm>
#include <cmath>
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
  // template <typename Alloc2, TensorIterator<U, Alloc2> It>
  // PiecewiseLinearApproximation(double dt, const It& begin, const It& end)
  //     : dt_{dt} {
  //   double t{0};
  //   It curr{begin};
  //   while (curr != end) {
  //     points_.insert({t, *curr});
  //     t += dt;
  //     ++curr;
  //   }
  // }

  // template <TensorIterator<U, Alloc> It>
  // PiecewiseLinearApproximation(double dt, const It& begin, const It& end)
  //     : dt_{dt} {
  //   // PiecewiseLinearApproximation<Alloc, It>(dt, begin, end);
  //   double t{0};
  //   It curr{begin};
  //   while (curr != end) {
  //     points_.insert({t, *curr});
  //     t += dt;
  //     ++curr;
  //   }
  // }

  PiecewiseLinearApproximation(double dt,
                               const Tensor<Tensor<U, Alloc>>& points)
      : dt_{dt}, points_{points} {
  }

  Tensor<U, Alloc> operator()(double time) const {
    // auto lb{points_.lower_bound(time)};

    // points_[std::min(time / dt, points.size()-1)]
    const auto index{
        std::min(static_cast<std::size_t>(time / dt_), points_.size() - 1)};
    auto lb{points_.begin() + index};

    auto next{lb};
    if (next != points_.end()) {
      ++next;
    }
    if (lb == points_.end() || next == points_.end()) {
      auto tmp = points_.end();
      lb = --tmp;
      next = lb--;
      double lbTime{dt_ * (points_.size() - 2)};
      const double partOfdt{1.0 / dt_ * (time - lbTime)};
      return *lb + (*next - *lb) * partOfdt;
    }

    double lbTime{dt_ * index};
    const double partOfdt{1.0 / dt_ * (time - lbTime)};
    return *lb + (*next - *lb) * partOfdt;
  }

  // void insert(double time, const Tensor<U, Alloc>& point) {
  //   points_.insert({time, point});
  // }

 private:
  double dt_;
  // std::map<double, Tensor<U, Alloc>> points_;
  const Tensor<Tensor<U, Alloc>>& points_;
};

template <typename U = double, class Alloc = std::allocator<U>>
class PiecewiseConstantApproximation {
 public:
  PiecewiseConstantApproximation(double dt,
                                 const Tensor<Tensor<U, Alloc>>& points)
      : dt_{dt}, points_{points} {
  }

  Tensor<U, Alloc> operator()(double time) const {
    const auto index{
        std::min(static_cast<std::size_t>(time / dt_), points_.size() - 1)};
    auto lb{points_.begin() + index};
    return *lb;
  }

 private:
  double dt_;
  const Tensor<Tensor<U, Alloc>>& points_;
};
}  // namespace optimization
