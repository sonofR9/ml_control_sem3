#include "function-approximation.h"
#include "global.h"
#include "two-wheel-robot.h"

namespace two_wheeled_robot {
using namespace optimization;
constexpr double kEpsTrajectory{1e-1};

template <int N>
using ControlParams = StaticTensor<N, StaticTensor<2, double>>;
using ControlApproximation = PiecewiseLinearApproximation<2, double>;

template <int N>
ControlParams<N> approximationFrom1D(const StaticTensor<2 * N>& solverResult) {
  ControlParams<N> result{};
  for (int i = 0; i < N; ++i) {
    result[i][0] = solverResult[2 * i];
    result[i][1] = solverResult[2 * i + 1];
  }
  return result;
}

template <int N>
StaticTensor<N, double> stdVectorToStaticTensor(
    const std::vector<double>& vec) {
  StaticTensor<N, double> result{};
  for (int i = 0; i < N; ++i) {
    result[i] = vec[i];
  }
  return result;
}

template <int T, StateSpaceFunction<T> F>
double integrate(double startT, const StaticTensor<T>& startX, F fun,
                 double interestT, double delta = 0.001) {
  double curT{startT};
  StaticTensor<T> curX{startX};

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

template <int N>
std::array<std::vector<double>, 3 + 1> getTrajectoryFromControl(
    const StaticTensor<2 * N, double>& solverResult, double tMax,
    double dt = 0.01) {
  auto approx{approximationFrom1D<N>(solverResult)};

  const ControlApproximation& func{tMax / N, approx.begin(), approx.end()};

  const auto control = [&func](const StaticTensor<3>&,
                               double time) -> StaticTensor<2> {
    return func(time);
  };
  Model robot{control, 1};

  StaticTensor<3> x0{10, 10, 0};
  double curT{0};
  double endT{tMax};
  return SolveDiffEqRungeKutte(curT, x0, robot, endT, dt);
}

template <int N>
double functional(const StaticTensor<2 * N, double>& solverResult,
                  double tMax = 10, double dt = 0.01) {
  const auto solvedX = getTrajectoryFromControl<N>(solverResult, tMax, dt);

  StaticTensor<3> xf{0, 0, 0};
  int i{0};
  double tEnd{0};
  for (; tEnd < tMax - kEps; tEnd += dt) {
    if (std::abs(solvedX[0][i] - xf[0]) + std::abs(solvedX[1][i] - xf[1]) +
            std::abs(solvedX[2][i] - xf[2]) <
        kEpsTrajectory) {
      break;
    }
    ++i;
  }

  int iFinal{i == solvedX[0].size() ? i - 1 : i};

  // TODO (novak) lower step and pass values from approximation
  const auto subIntegrative = [dt](const StaticTensor<3>& point) -> double {
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
  for (int i{0}; i < iFinal; ++i) {
    integral += subIntegrative({solvedX[0][i], solvedX[1][i], solvedX[2][i]});
  }

  return tEnd +
         std::sqrt(std::pow(solvedX[0][iFinal] - xf[0], 2) +
                   std::pow(solvedX[1][iFinal] - xf[1], 2) +
                   std::pow(solvedX[2][iFinal] - xf[2], 2)) +
         integral;
}
}  // namespace two_wheeled_robot
