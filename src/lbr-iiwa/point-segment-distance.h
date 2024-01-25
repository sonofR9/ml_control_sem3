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

#include "tensor.h"

namespace kuka {
using namespace optimization;

template <typename T, class Alloc>
struct LineSegment {
  const Tensor<T, Alloc>& begin;
  const Tensor<T, Alloc>& end;
};

template <typename T, class Alloc>
constexpr T pointSegmentDistance(const Tensor<T, Alloc>& point,
                                 const LineSegment<T, Alloc>& segment) {
  assert((segment.begin.size() == 3) && (segment.end.size() == 3) &&
         "it is not 3d segment");
  assert((segment.begin != segment.end) &&
         "it is not segment: start and end points must not be the same");
  assert((point.size() == 3) && "it is not 3d point");

  thread_local static auto direction = Tensor<T, Alloc>(3);
  direction = segment.end - segment.begin;

  thread_local static auto pointToSegment = Tensor<T, Alloc>(3);
  pointToSegment = point - segment.end;
  if (dotProduct(direction, pointToSegment) >= 0) {
    return std::sqrt(dotProduct(pointToSegment, pointToSegment));
  }

  pointToSegment = point - segment.begin;
  auto dot{dotProduct(direction, pointToSegment)};
  if (dot <= 0) {
    return std::sqrt(dotProduct(pointToSegment, pointToSegment));
  }

  pointToSegment -= dot / dotProduct(direction, direction) * direction;
  return std::sqrt(dotProduct(pointToSegment, pointToSegment));
}
}  // namespace kuka
