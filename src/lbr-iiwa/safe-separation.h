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
#include "tensor.h"

namespace kuka {
using namespace optimization;

template <typename T, template <typename> class Alloc>
using Manipulator = KukaLbrIiwa<T, Alloc>;

template <typename T, template <typename> class Alloc>
using AllJointsCoordinates =
    Tensor<typename Manipulator<T, Alloc>::CoordinatesTaskSpace,
           Alloc<typename Manipulator<T, Alloc>::CoordinatesTaskSpace>>;

/// At every timestamp from every joint to every obstacle part begin and end.
/// Therefore the shape is [number of the timestamps, number of the joints,
/// number of the obstacles, number of parts of the obstacle, 2 (begin and end)]
template <typename T, template <typename> class Alloc>
using AllowedDistances = Tensor5d<T, Alloc>;

template <typename T, template <typename> class Alloc>
AllowedDistances<T, Alloc> fun(
    const Environment<T, Alloc>& environment,
    const Tensor<AllJointsCoordinates<T, Alloc>,
                 Alloc<AllJointsCoordinates<T, Alloc>>>& solvedTaskSpace,
    double dt) {
  using Manipulator = Manipulator<T, Alloc>;
  using AllJointsCoordinates = AllJointsCoordinates<T, Alloc>;

  if (environment.obstaclesAtTimestamp[0].size() == 0) {
    return AllowedDistances<T, Alloc>(
        solvedTaskSpace.size(),
        Tensor4d<T, Alloc>(kNumDof,
                           Tensor3d<T, Alloc>(0, Tensor2d<T, Alloc>())));
  }
  auto allowedDistances = AllowedDistances<T, Alloc>(
      solvedTaskSpace.size(),
      Tensor4d<T, Alloc>(
          kNumDof, Tensor3d<T, Alloc>(
                       environment.obstaclesAtTimestamp[0].size(),
                       Tensor2d<T, Alloc>(
                           environment.obstaclesAtTimestamp[0][0].parts.size(),
                           Tensor<T, Alloc<T>>({0, 0})))));

  // calculate speed task space
  // shape is [number of positions, number of joints, number of linear
  // coordinates (xyz)=3]
  auto speedTaskSpace =
      Tensor<AllJointsCoordinates, Alloc<AllJointsCoordinates>>(
          solvedTaskSpace.size(),
          AllJointsCoordinates(kNumDof,
                               Manipulator::CoordinatesTaskSpace(3, 0)));
  for (std::size_t i{0}; i < solvedTaskSpace.size() - 1; ++i) {
    for (std::size_t j{0}; j < kNumDof; ++j) {
      for (std::size_t k{0}; k < 3; ++k) {
        speedTaskSpace[i][j][k] =
            (solvedTaskSpace[i + 1][j][k] - solvedTaskSpace[i][j][k]) / dt;
      }
    }
  }
  // last speed assumed to be the same as previous
  for (std::size_t j = 0; j < kNumDof; ++j) {
    for (std::size_t k{0}; k < 3; ++k) {
      speedTaskSpace[solvedTaskSpace.size() - 1][j][k] =
          speedTaskSpace.back()[j][k];
    }
  }

  // calculate environment speed
  // shape is [number of environment positions, number of obstacles, number of
  // parts in obstacle, 2 (begin and end), number of linear coordinates (xyz)=3]
  using Environment = Environment<T, Alloc>;
  auto environmentSpeed = Tensor5d<T, Alloc>(
      environment.obstaclesAtTimestamp.size(),
      Tensor4d<T, Alloc>(
          environment.obstaclesAtTimestamp[0].size(),
          Tensor3d<T, Alloc>(
              environment.obstaclesAtTimestamp[0][0].parts.size(),
              Tensor2d<T, Alloc>(2, Tensor<T, Alloc<T>>({0, 0, 0})))));
  for (std::size_t i{0}; i < environment.obstaclesAtTimestamp.size(); ++i) {
    for (std::size_t j{0}; j < environment.obstaclesAtTimestamp[i].size();
         ++j) {
      if (environmentSpeed[i].size() <= j) {
        environmentSpeed[i].emplace_back(
            environment.obstaclesAtTimestamp[i][j].parts.size(),
            Tensor2d<T, Alloc>(2, Tensor<T, Alloc<T>>({0, 0, 0})));
      }

      for (std::size_t k{0};
           k < environment.obstaclesAtTimestamp[i][j].parts.size(); ++k) {
        if (environmentSpeed[i][j].size() <= k) {
          environmentSpeed[i][j].emplace_back(2,
                                              Tensor<T, Alloc<T>>({0, 0, 0}));
        }

        for (std::size_t l{0}; l < 3; ++l) {
          environmentSpeed[i][j][k][0][l] =
              (environment.obstaclesAtTimestamp[i][j]
                   .parts[k]
                   .segment.begin[l] -
               environment.obstaclesAtTimestamp[i][j]
                   .parts[k]
                   .segment.begin[l]) /
              environment.dt;

          environmentSpeed[i][j][k][1][l] =
              (environment.obstaclesAtTimestamp[i][j].parts[k].segment.end[l] -
               environment.obstaclesAtTimestamp[i][j].parts[k].segment.end[l]) /
              environment.dt;
        }
      }
    }
  }
  // last speed assumed to be the same as previous
  const auto lastTimestampIndex{environment.obstaclesAtTimestamp.size() - 1};
  for (std::size_t j{0};
       j < environment.obstaclesAtTimestamp[lastTimestampIndex].size(); ++j) {
    if (environmentSpeed[lastTimestampIndex].size() <= j) {
      environmentSpeed[lastTimestampIndex].emplace_back(
          environment.obstaclesAtTimestamp[lastTimestampIndex][j].parts.size(),
          Tensor2d<T, Alloc>(2, Tensor<T, Alloc<T>>({0, 0, 0})));
    }

    for (std::size_t k{0};
         k <
         environment.obstaclesAtTimestamp[lastTimestampIndex][j].parts.size();
         ++k) {
      if (environmentSpeed[lastTimestampIndex][j].size() <= k) {
        environmentSpeed[lastTimestampIndex][j].emplace_back(
            2, Tensor<T, Alloc<T>>({0, 0, 0}));
      }

      for (std::size_t l{0}; l < 3; ++l) {
        environmentSpeed[lastTimestampIndex][j][k][1][l] =
            environmentSpeed[lastTimestampIndex - 1][j][k][1][l];
        environmentSpeed[lastTimestampIndex][j][k][0][l] =
            environmentSpeed[lastTimestampIndex - 1][j][k][0][l];
      }
    }
  }
}
}  // namespace kuka
