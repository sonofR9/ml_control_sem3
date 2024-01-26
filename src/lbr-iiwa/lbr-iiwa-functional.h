/* Copyright (C) 2023-2024 Novak Alexander
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "lbr-iiwa/lbr-iiwa-forward-kinematics.h"
#include "lbr-iiwa/types.h"  // IWYU pragma: export

namespace kuka {
using namespace optimization;

template <typename T, template <typename> class Alloc>
Tensor2d<T, Alloc> optimizationResultToSequence(
    const Tensor<T, Alloc<T>>& result) {
  assert((result.size() % kNumDof == 0) &&
         "optimization result must be divisible by number degrees of freedom");

  auto sequence =
      Tensor2d<T, Alloc>(result.size() / kNumDof, Tensor<T, Alloc<T>>(kNumDof));
  for (std::size_t i = 0; i < result.size(); i += kNumDof) {
    sequence.emplace_back(Tensor<T, Alloc<T>>{
        result[i], result[i + 1], result[i + 2], result[i + 3], result[i + 4],
        result[i + 5], result[i + 6]});
  }
  return sequence;
}

template <typename T, template <typename> class Alloc>
class Functional {
 public:
  struct Coeffients {
    double time;
    double terminal;
    double obstacle;
  };

  constexpr explicit Functional(Coeffients coefficients,
                                double terminalTolerance,
                                Environment<T, Alloc> environment) noexcept
      : coeffients_{coefficients}, terminalTolerance_{terminalTolerance},
        environment_{std::move(environment)} {
  }

  /**
   * @param solverResult .shape (2 * N)
   * @param dt for integration
   */
  double operator()(const Tensor<T, Alloc<T>>& solverResult, double tMax = 10,
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

    const auto isColliding = [this, dt](const Tensor<T, Alloc>& point) -> bool {
      auto mySqr = [](auto x) { return x * x; };
      auto pointInside = [mySqr](const Tensor<T, Alloc>& point,
                                 const CircleData& obstacle) -> double {
        return obstacle.r - std::sqrt(mySqr(point[0] - obstacle.x) +
                                      mySqr(point[1] - obstacle.y));
      };
      for (const auto& obstacle : environment_) {
        if (pointInside(point, obstacle) > 0) {
          return true;
        }
      }

      return false;
    };

    double integral{0};
    for (std::size_t i{0}; i < iFinal; ++i) {
      constexpr double kBigNumber = 1e5;
      integral += isColliding({solvedX[0][i], solvedX[1][i], solvedX[2][i]})
                      ? kBigNumber * dt
                      : 0;
    }

    return 0;  // TODO(novak)
  }

 private:
  Coeffients coeffients_;
  double terminalTolerance_;
  Environment<T, Alloc> environment_;
};
}  // namespace kuka
