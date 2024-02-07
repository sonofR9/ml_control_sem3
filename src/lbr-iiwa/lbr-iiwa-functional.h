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
Tensor2d<T, Alloc> optimizationResultToPosesSequence(
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
    double speed;
    double obstacle;
  };

  constexpr explicit Functional(Coeffients coefficients,
                                double terminalTolerance,
                                Tensor<T, Alloc<T>> allowedJointSpeeds,
                                Environment<T, Alloc> environment,
                                Tensor<T, Alloc<T>> taskGoal) noexcept
      : coeffients_{coefficients}, terminalTolerance_{terminalTolerance},
        allowedJointSpeeds_{allowedJointSpeeds},
        environment_{std::move(environment)}, taskGoal_{std::move(taskGoal)} {
  }

  /**
   * @param solverResult .shape (2 * N)
   * @param dt for integration
   */
  double operator()(const Tensor<T, Alloc<T>>& solverResult, double tMax = 10,
                    double dt = 0.01) {
    constexpr double kBigNumber = 1e5;

    using Manipulator = KukaLbrIiwa<T, Alloc>;
    using AllJointsCoordinates =
        Tensor<typename Manipulator::CoordinatesTaskSpace,
               Alloc<typename Manipulator::CoordinatesTaskSpace>>;

    const auto solvedQ{
        optimizationResultToPosesSequence<T, Alloc>(solverResult)};

    // to task space
    auto solvedTaskSpace =
        Tensor<AllJointsCoordinates, Alloc<AllJointsCoordinates>>(
            solvedQ.size(), AllJointsCoordinates{});
    std::size_t i{0};
    for (const auto& q : solvedQ) {
      solvedTaskSpace[i] = Manipulator::forwardKinematics(q);
      ++i;
    };

    // calculate speed in generalized coordinates q
    auto speedQ =
        Tensor2d<T, Alloc>(solvedQ.size(), Tensor<T, Alloc<T>>(kNumDof));
    for (std::size_t i = 0; i < solvedQ.size() - 1; ++i) {
      for (std::size_t j = 0; j < kNumDof; ++j) {
        speedQ[i][j] = (solvedQ[i + 1][j] - solvedQ[i][j]) / dt;
      }
    }
    // last speed assumed to be the same as previous
    for (std::size_t j = 0; j < kNumDof; ++j) {
      speedQ.back[solvedQ.size() - 1][j] = speedQ.front()[j];
    }

    // does reach the end?
    i = 0;
    double tEnd{0};
    double endDiff{0};
    for (; tEnd < tMax - kEps; tEnd += dt) {
      endDiff = 0;
      for (std::size_t j = 0; j < kNumDof; ++j) {
        endDiff += std::abs(solvedTaskSpace[i][kNumDof - 1][j] - taskGoal_[j]);
      }
      if (endDiff < terminalTolerance_) {
        break;
      }
      ++i;
    }

    std::size_t iFinal{i == solvedQ.size() ? i - 1 : i};

    // does excees speed limits?
    double speedPenalty{0};
    for (std::size_t i = 0; i < speedQ.size() - 1; ++i) {
      for (std::size_t j = 0; j < kNumDof; ++j) {
        speedPenalty += speedQ[i][j] > allowedJointSpeeds_[j] ? kBigNumber : 0;
      }
    }

    // calculate allowed distances
    auto allowedDistances{
        calculateAllowedDistances(environment_, solvedTaskSpace, dt)};
    // TODO(novak)

    // colliding?
    // TODO(novak) allowedDistance here should be 3d (-environmentTimestamp and
    // joint)
    const auto isColliding =
        [this](const Tensor<T, Alloc<T>>& poseTaskSpace,
               const typename Environment<T, Alloc>::EnvironmentAtTimestamp&
                   environment,
               auto allowedDistance) -> bool {
      assert((poseTaskSpace.size() >= 3) &&
             "one of xyz coordinates is missing");

      const auto allowedToObstacle =
          [&allowedDistance](std::size_t i, std::size_t j) -> double {
        T min = std::numeric_limits<T>::max();
        for (std::size_t k = 0; k < allowedDistance.size(); ++k) {  // numDof
          for (std::size_t l{0}; l < allowedDistance[k][i][j].size(); ++l) {
            min = std::min(allowedDistance[k][i][j][l], min);
          }
        }
        return min;
      };

      std::size_t i{0};
      for (const auto& obstacles : environment) {
        std::size_t j{0};
        for (const auto& cylinder : obstacles.parts) {
          if (pointToCylinderDistance(poseTaskSpace, cylinder) <
              allowedToObstacle(i, j)) {
            return true;
          }
          ++j;
        }
        ++i;
      }
      return false;

      //   return std::ranges::any_of(
      //       environment_,
      //       [this, allowedDistance,
      //        &poseTaskSpace](const auto& obstacles) -> bool {
      //         return std::ranges::any_of(
      //             obstacles.parts,
      //             [this, allowedDistance,
      //              &poseTaskSpace](const auto& cylinder) -> bool {
      //               return pointToCylinderDistance(poseTaskSpace, cylinder) <
      //                      allowedDistance;
      //             });
      //       });
    };

    double obstaclePenalty{0};
    const double kEnvironmentTMax{
        environment_.dt * (environment_.obstaclesAtTimestamp.size() - 1)};
    const auto convertToEnvironmentIndex =
        [this, kEnvironmentTMax](double time) -> std::size_t {
      return time >= kEnvironmentTMax
                 ? environment_.obstaclesAtTimestamp.size() - 1
                 : static_cast<std::size_t>(time / environment_.dt);
    };
    double tBuf{0};
    for (std::size_t i{0}; i < iFinal; ++i) {
      const std::size_t obstaclesIndex{i / iFinal * tEnd / environment_.dt};
      for (const auto& xyz : solvedTaskSpace[i]) {
        obstaclePenalty +=
            isColliding(
                xyz,
                environment_
                    .obstaclesAtTimestamp[convertToEnvironmentIndex(tBuf)],
                allowedDistances[obstaclesIndex])
                ? kBigNumber * dt
                : 0;
      }
      tBuf += dt;
    }

    auto result{tEnd + coeffients_.terminal * endDiff +
                coeffients_.obstacle * obstaclePenalty +
                coeffients_.speed * speedPenalty};
    return 0;  // TODO(novak)
  }

 private:
  Coeffients coeffients_;
  double terminalTolerance_;

  Tensor<T, Alloc<T>> allowedJointSpeeds_;
  Environment<T, Alloc> environment_;

  Tensor<T, Alloc<T>> taskGoal_;
};
}  // namespace kuka
