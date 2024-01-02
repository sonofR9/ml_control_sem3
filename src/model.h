#include "function-approximation.h"
#include "global.h"
#include "two-wheel-robot.h"

#include <cassert>

namespace two_wheeled_robot {
using namespace optimization;
constexpr double kEpsTrajectory{1e-1};

/**
 * @brief StaticTensor<N, StaticTensor<2, double>>
 */
template <typename T>
using ControlParams = Tensor<Tensor<T>>;
template <typename T>
using ControlApproximation = PiecewiseLinearApproximation<T>;

/**
 * @arg solverResult shape (2 * N)
 * @result shape (N, 2)
 */
template <typename T = double>
ControlParams<T> approximationFrom1D(const Tensor<T>& solverResult) {
  // TODO(novak) is it an error?
  ControlParams<T> result{solverResult.size() / 2, Tensor<double>(2)};
  for (std::size_t i = 0; i < solverResult.size() / 2; ++i) {
    result[i][0] = solverResult[2 * i];
    result[i][1] = solverResult[2 * i + 1];
  }
  return result;
}

template <typename T>
Tensor<double> stdVectorToStaticTensor(const std::vector<T>& vec) {
  Tensor<double> result{};
  for (std::size_t i = 0; i < vec.size(); ++i) {
    result[i] = vec[i];
  }
  return result;
}

template <typename T, StateSpaceFunction<T> F>
double integrate(double startT, const Tensor<T>& startX, F fun,
                 double interestT, double delta = 0.001) {
  assert((startX.size() == fun(startX, startT).size()));
  double curT{startT};
  Tensor<T> curX{startX};

  while (curT < interestT) {
    auto k1 = fun(curX, curT);
    auto k2 = fun(curX + delta / 2 * k1, curT + delta / 2);
    auto k3 = fun(curX + delta / 2 * k2, curT + delta / 2);
    auto k4 = fun(curX + delta * k3, curT + delta);

    curX += delta / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
    curT += delta;
  }
  return curX;
}

template <typename T>
std::vector<std::vector<T>> getTrajectoryFromControl(
    const Tensor<T>& solverResult, double tMax, double dt = 0.01) {
  auto approx{approximationFrom1D<T>(solverResult)};
  assert((approx.size() * 2 == solverResult.size()));

  const ControlApproximation<T>& func{tMax / approx.size(), approx.begin(),
                                      approx.end()};
  assert((func(0).size() == 2));

  const auto control = [&func](const Tensor<T>&, double time) -> Tensor<T> {
    return func(time);
  };
  Model robot{control, 1};

  Tensor<T> x0{10, 10, 0};
  double curT{0};
  double endT{tMax};
  const auto result = solveDiffEqRungeKutte(curT, x0, robot, endT, dt);
  assert((result.size() == 3 + 1));
  return result;
}

/**
 * @param solverResult .shape (2 * N)
 * @param dt for integration
 */
template <typename T = double>
double functional(const Tensor<T>& solverResult, double tMax = 10,
                  double dt = 0.01) {
  const auto solvedX = getTrajectoryFromControl<T>(solverResult, tMax, dt);

  Tensor<T> xf{0, 0, 0};
  std::size_t i{0};
  double tEnd{0};
  for (; tEnd < tMax - kEps; tEnd += dt) {
    if (std::abs(solvedX[0][i] - xf[0]) + std::abs(solvedX[1][i] - xf[1]) +
            std::abs(solvedX[2][i] - xf[2]) <
        kEpsTrajectory) {
      break;
    }
    ++i;
  }

  std::size_t iFinal{i == solvedX[0].size() ? i - 1 : i};

  const auto subIntegrative = [dt](const Tensor<T>& point) -> double {
    const double h1{std::sqrt(2.5) - std::sqrt(std::pow(point[0] - 2.5, 2) +
                                               std::pow(point[1] - 2.5, 2))};
    const double h2{std::sqrt(2.5) - std::sqrt(std::pow(point[0] - 7.5, 2) +
                                               std::pow(point[1] - 7.5, 2))};
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

  return tEnd +
         std::sqrt(std::pow(solvedX[0][iFinal] - xf[0], 2) +
                   std::pow(solvedX[1][iFinal] - xf[1], 2) +
                   std::pow(solvedX[2][iFinal] - xf[2], 2)) +
         integral;
}
}  // namespace two_wheeled_robot
