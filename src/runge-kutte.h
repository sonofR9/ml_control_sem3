#pragma once

#include "global.h"
#include "tensor.h"

#include <vector>
namespace optimization {
template <typename T, class Alloc, StateSpaceFunction<T, Alloc> F>
Tensor<T, Alloc> rungeKutteStep(double startT, const Tensor<T, Alloc>& startX,
                                F fun, double interestT, double delta = 0.001) {
  double curT{startT};
  Tensor<T, Alloc> curX{startX};

  while (curT < interestT) {
    auto k1 = fun(curX, curT);
    auto k2 = fun(curX + delta / 2 * k1, curT + delta / 2);
    auto k3 = fun(curX + delta / 2 * k2, curT + delta / 2);
    auto k4 = fun(curX + delta * k3, curT + delta);

    curX += delta / 6 * (k1 + 2 * (k2 + k3) + k4);
    curT += delta;
  }
  return curX;
}

/**
 * @tparam T type
 * @return std::vector<std::vector<T>> array of vectors with values of each
 * state coordinate and time. Given input =Tensor with shape (N...), output will
 * have shape (N+1, M) where M is the number of steps
 */
template <typename T, class Alloc, StateSpaceFunction<T, Alloc> F,
          class VectorAlloc = std::allocator<std::vector<T, Alloc>>>
std::vector<std::vector<T, Alloc>, VectorAlloc> solveDiffEqRungeKutte(
    double startT, const Tensor<T, Alloc>& startX, F fun, double lastT,
    double delta = 0.001) {
  auto result =
      std::vector<std::vector<T, Alloc>, VectorAlloc>(startX.size() + 1);
  double curT{startT};
  Tensor<T, Alloc> curX{startX};

  auto addPointToResult = [&result](const Tensor<T, Alloc>& point,
                                    double curT) -> void {
    for (std::size_t i{0}; i < result.size() - 1; ++i) {
      result[i].push_back(point[i]);
    }
    result[result.size() - 1].push_back(curT);
  };

  addPointToResult(curX, curT);

  while (curT < lastT - kEps) {
    curX = rungeKutteStep(
        curT, curX,
        std::function<Tensor<T, Alloc>(Tensor<T, Alloc>, double)>(
            fun),  // std::function<Tensor<3>(Tensor<3>, double)
        curT + delta, delta);
    curT += delta;
    addPointToResult(curX, curT);
  }

  return result;
}
}  // namespace optimization
