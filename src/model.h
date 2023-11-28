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

template <int N>
double functional(const Vector<2 * N, double>& solverResult) {
  double dt = 0.01;

  const auto& approx{approximationFrom1D(solverResult)};

  const ControlApproximation<N>& func{dt, approx.begin(), approx.end()};

  const auto control = [&func](const Vector<3>&, double time) -> double {
    return func(time);
  };
  Model robot{control, 1};

  Vector<3> x0{10, 10, 0};
  double curT{0};
  double endT{dt * N};
  const auto solvedX = SolveDiffEqRungeKutte(curT, x0, robot, endT, dt);

  Vector<3> xf{0, 0, 0};
  int i{0};
  for (i; i < N; ++i) {
    if (std::abs(solvedX[i][0] - xf[0]) + std::abs(solvedX[i][1] - xf[1]) +
            std::abs(solvedX[i][2] - xf[2]) >
        kEps) {
      break;
    }
  }

  double result{0};
  int iFinal{i == N ? N - 1 : i};

  result = iFinal * dt +
           std::sqrt(std::pow(solvedX[iFinal][0] - xf[0], 2) +
                     std::pow(solvedX[iFinal][1] - xf[1], 2) +
                     std::pow(solvedX[iFinal][2] - xf[2], 2)) +
           RungeKutteStep(
               0, x0,
               [&robot](const Vector<3>& point, double time) {
                 double h1{2.5 - std::sqrt(std::pow(point[0] - 2.5, 2) +
                                           std::pow(point[1] - 2.5, 2))};
                 double h2{2.5 - std::sqrt(std::pow(point[0] - 7.5, 2) +
                                           std::pow(point[1] - 7.5, 2))};
                 if (h1 > 0 || h2 > 0) {
                   double kBigNumber = 1e6;
                   return kBigNumber;
                 }
                 return h1 * robot(point, time) + h2 * robot(point, time);
               },
               iFinal * dt);
}

}  // namespace two_wheeled_robot