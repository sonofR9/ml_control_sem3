#include "function-approximation.h"
#include "global.h"
#include "parse-function.h"
#include "runge-kutte.h"
#include "two-wheel-robot.h"
#include "utils.h"

#include <cassert>

namespace two_wheeled_robot {
using namespace optimization;

/**
 * @brief StaticTensor<N, StaticTensor<2, double>>
 */
template <typename T, class Alloc, class TensorAlloc>
using ControlParams = Tensor<Tensor<T, Alloc>, TensorAlloc>;
template <typename T, class Alloc>
using ControlApproximation = PiecewiseLinearApproximation<T, Alloc>;
// using ControlApproximation = PiecewiseConstantApproximation<T, Alloc>;

/**
 * @arg solverResult shape (2 * N)
 * @result shape (N, 2)
 */
template <typename T = double, class Alloc = std::allocator<T>,
          class TensorAlloc = std::allocator<Tensor<T, Alloc>>>
ControlParams<T, Alloc, TensorAlloc> approximationFrom1D(
    const Tensor<T, Alloc>& solverResult) {
  ControlParams<T, Alloc, TensorAlloc> result{solverResult.size() / 2,
                                              Tensor<T, Alloc>(2)};
  for (std::size_t i = 0; i < solverResult.size() / 2; ++i) {
    result[i][0] = solverResult[2 * i];
    result[i][1] = solverResult[2 * i + 1];
  }
  return result;
}

template <typename T, class Alloc>
Tensor<T, Alloc> stdVectorToStaticTensor(const std::vector<T, Alloc>& vec) {
  Tensor<T, Alloc> result{};
  for (std::size_t i = 0; i < vec.size(); ++i) {
    result[i] = vec[i];
  }
  return result;
}

// template <typename T, class Alloc, StateSpaceFunction<T, Alloc> F>
// double integrate(double startT, const Tensor<T, Alloc>& startX, F fun,
//                  double interestT, double delta = 0.001) {
//   assert((startX.size() == fun(startX, startT).size()));
//   double curT{startT};
//   Tensor<T, Alloc> curX{startX};

//   while (curT < interestT) {
//     auto k1 = fun(curX, curT);
//     auto k2 = fun(curX + delta / 2 * k1, curT + delta / 2);
//     auto k3 = fun(curX + delta / 2 * k2, curT + delta / 2);
//     auto k4 = fun(curX + delta * k3, curT + delta);

//     curX += delta / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
//     curT += delta;
//   }
//   return curX;
// }

template <typename T, class Alloc,
          class VectorAlloc = std::allocator<std::vector<T, Alloc>>>
std::vector<std::vector<T, Alloc>, VectorAlloc> getTrajectoryFromControl(
    const Tensor<T, Alloc>& solverResult, double tMax, double dt = 0.01) {
  auto approx{approximationFrom1D<T>(solverResult)};
  assert((approx.size() * 2 == solverResult.size()));

  const ControlApproximation<T, Alloc> func{tMax / approx.size(), approx};
  assert((func(0).size() == 2));

  Tensor<T, Alloc> x0{10, 10, 0};
  double curT{0};
  double endT{tMax};

  // TODO(novak) code duplication

  if constexpr (CallableOneArgPreallocatedResult<decltype(func), double, T,
                                                 Alloc>) {
    const auto control = [&func](const Tensor<T, Alloc>&, double time,
                                 Tensor<T, Alloc>& preallocatedResult) -> void {
      return func(time, preallocatedResult);
    };
    Model<Alloc, decltype(control)> robot{control, 1};

    const auto result = solveDiffEqRungeKutte(curT, x0, robot, endT, dt);
    assert((result.size() == 3 + 1));
    return result;
  } else {
    const auto control = [&func](const Tensor<T, Alloc>&,
                                 double time) -> Tensor<T, Alloc> {
      return func(time);
    };
    Model<Alloc, decltype(control)> robot{control, 1};

    const auto result = solveDiffEqRungeKutte(curT, x0, robot, endT, dt);
    assert((result.size() == 3 + 1));
    return result;
  }
}

class Functional {
 public:
  struct Coeffients {
    double time;
    double terminal;
    double obstacle;
  };
  constexpr explicit Functional(Coeffients coefficients,
                                double terminalTolerance) noexcept
      : coeffients_{coefficients}, terminalTolerance_{terminalTolerance} {
  }

  /**
   * @param solverResult .shape (2 * N)
   * @param dt for integration
   */
  template <typename T = double, class Alloc = std::allocator<T>,
            class VectorAlloc = std::allocator<std::vector<T, Alloc>>>
  double operator()(const Tensor<T, Alloc>& solverResult, double tMax = 10,
                    double dt = 0.01) {
    const auto solvedX =
        getTrajectoryFromControl<T, Alloc, VectorAlloc>(solverResult, tMax, dt);

    Tensor<T, Alloc> xf{0, 0, 0};
    std::size_t i{0};
    double tEnd{0};
    for (; tEnd < tMax - kEps; tEnd += dt) {
      if (std::abs(solvedX[0][i] - xf[0]) + std::abs(solvedX[1][i] - xf[1]) +
              std::abs(solvedX[2][i] - xf[2]) <
          terminalTolerance_) {
        break;
      }
      ++i;
    }

    std::size_t iFinal{i == solvedX[0].size() ? i - 1 : i};

    const auto subIntegrative = [dt](const Tensor<T, Alloc>& point) -> double {
      auto mySqr = [](auto x) { return x * x; };
      const double h1{std::sqrt(2.5) -
                      std::sqrt(mySqr(point[0] - 2.5) + mySqr(point[1] - 2.5))};
      const double h2{std::sqrt(2.5) -
                      std::sqrt(mySqr(point[0] - 7.5) + mySqr(point[1] - 7.5))};
      if (h1 > 0 || h2 > 0) {
        const double kBigNumber = 1e5;
        return kBigNumber * dt;
      }
      return 0.0;
    };

    double integral{0};
    for (std::size_t i{0}; i < iFinal; ++i) {
      integral += subIntegrative({solvedX[0][i], solvedX[1][i], solvedX[2][i]});
    }

    return coeffients_.time * tEnd +
           coeffients_.terminal *
               std::sqrt(std::pow(solvedX[0][iFinal] - xf[0], 2) +
                         std::pow(solvedX[1][iFinal] - xf[1], 2) +
                         std::pow(solvedX[2][iFinal] - xf[2], 2)) +
           coeffients_.obstacle * integral;
  }

 private:
  Coeffients coeffients_;
  double terminalTolerance_;
};
}  // namespace two_wheeled_robot
