#include "matplotlibcpp.h"
#include "runge_kutte.h"
#include "two_wheel_robot.h"
// import runge_kutte1;

namespace plt = matplotlibcpp;

int main() {
  using namespace runge_kutte;

  auto u1 = [](double t) -> double { return std::cos(t); };
  auto u2 = [](double t) -> double { return std::sin(t); };
  two_wheeled_robot::Model robot(u1, u2, 2, 1);

  double delta{0.01};
  StatePoint<3> curX{0, 0, 0};
  double curT{0};
  double endT{100};
  const auto solvedFun = SolveDiffEqRungeKutte(curT, curX, robot, endT, delta);

  plt::figure();
  plt::plot(solvedFun[0], solvedFun[1]);
  plt::show();

  return 0;
}
