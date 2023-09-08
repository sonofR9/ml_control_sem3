#include "matplotlibcpp.h"
#include "runge_kutte.h"

namespace plt = matplotlibcpp;

int main() {
  using namespace runge_kutte;
  auto u1 = [](double t) -> double { return 1; };
  auto u2 = [](double t) -> double { return 1; };
  const auto fun = [u1, u2](StatePoint<3> state, double t) -> StatePoint<3> {
    return {(u1(t) + u2(t)) * std::cos(state[2]),
            (u1(t) + u2(t)) * std::sin(state[2]), u1(t) - u2(t)};
  };

  double delta{0.01};
  StatePoint<3> curX{0, 0, 0};
  double curT{0};
  double endT{10};
  const auto solvedFun = SolveDiffEqRungeKutte(curT, curX, (fun), endT, delta);

  plt::figure();
  plt::plot(solvedFun[0], solvedFun[1]);
  plt::show();

  return 0;
}
