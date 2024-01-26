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

#include "tensor.h"  // IWYU pragma: export

namespace kuka {
using namespace optimization;

template <typename T, template <typename> class Alloc>
using Tensor2d = Tensor<Tensor<T, Alloc<T>>, Alloc<Tensor<T, Alloc<T>>>>;

template <typename T, class Alloc>
struct LineSegment {
  Tensor<T, Alloc> begin;
  Tensor<T, Alloc> end;
};

template <typename T, class Alloc>
struct Cylinder {
  LineSegment<T, Alloc> segment;
  T radius;
};

template <typename T, template <typename> class Alloc>
struct Obstacle {
  using Part = Cylinder<T, Alloc<T>>;
  Tensor<Part, Alloc<Part>> parts;
};

template <typename T, template <typename> class Alloc>
struct Environment {
  using ObstacleType = Obstacle<T, Alloc>;
  Tensor<ObstacleType, Alloc<ObstacleType>> obstacles;
};

template <typename T, template <typename> class Alloc>
struct Manipulator {
  using CylinderType = Cylinder<T, Alloc<T>>;
  Tensor<CylinderType, Alloc<CylinderType>> links;
};
}  // namespace kuka
