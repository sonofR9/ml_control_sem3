#include "function-approximation.h"
#include "global.h"
#include "two-wheel-robot.h"

namespace two_wheeled_robot {
using namespace optimization;

template <int N>
using ControlParams = Vector<N, Vector<2, double>>;
template <int N>
using ControlApproximation = PiecewiseLinearApproximation<N, Vector<2, double>>;

template <int N>
ControlParams<N> approximationFrom1D(const Vector<2 * N>& solverResult) {
  ControlParams<N> result{};
  for (int i = 0; i < N; ++i) {
    result[i][0] = solverResult[2 * i];
    result[i][1] = solverResult[2 * i + 1];
  }
  return result;
}

template <int N>
Vector<N, double> stdVectorToVector(const std::vector<double>& vec) {
  Vector<N, double> result{};
  for (int i = 0; i < N; ++i) {
    result[i] = vec[i];
  }
  return result;
}

template <int T, StateSpaceFunction<T> F>
double integrate(double startT, const Vector<T>& startX, F fun,
                 double interestT, double delta = 0.001) {
  double curT{startT};
  Vector<T> curX{startX};

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
    const Vector<2 * N, double>& solverResult, double dt = 0.01) {
  auto approx{approximationFrom1D<N>(solverResult)};

  const PiecewiseLinearApproximation<2, double>& func{dt, approx.begin(),
                                                      approx.end()};

  const auto control = [&func](const Vector<3>&, double time) -> Vector<2> {
    return func(time);
  };
  Model robot{control, 1};

  Vector<3> x0{10, 10, 0};
  double curT{0};
  double endT{dt * N};
  return SolveDiffEqRungeKutte(curT, x0, robot, endT, dt);
}

template <int N>
double functional(const Vector<2 * N, double>& solverResult) {
  double dt = 0.1;
  const auto solvedX = getTrajectoryFromControl<N>(solverResult, dt);

  Vector<3> xf{0, 0, 0};
  int i{0};
  for (; i < N; ++i) {
    if (std::abs(solvedX[0][i] - xf[0]) + std::abs(solvedX[1][i] - xf[1]) +
            std::abs(solvedX[2][i] - xf[2]) <
        kEps) {
      break;
    }
  }

  int iFinal{i == N ? N - 1 : i};

  const auto subIntegrative = [dt](const Vector<3>& point) -> double {
    const double h1{2.5 - std::sqrt(std::pow(point[0] - 2.5, 2) +
                                    std::pow(point[1] - 2.5, 2))};
    const double h2{2.5 - std::sqrt(std::pow(point[0] - 7.5, 2) +
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

  return 100 * iFinal * dt +
         std::sqrt(std::pow(solvedX[0][iFinal] - xf[0], 2) +
                   std::pow(solvedX[1][iFinal] - xf[1], 2) +
                   std::pow(solvedX[2][iFinal] - xf[2], 2)) +
         integral;
}
}  // namespace two_wheeled_robot
