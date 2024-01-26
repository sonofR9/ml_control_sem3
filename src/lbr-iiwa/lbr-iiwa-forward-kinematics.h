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

template <typename T, template <typename> class Alloc>
using Tensor2d = Tensor<Tensor<T, Alloc<T>>, Alloc<Tensor<T, Alloc<T>>>>;

// NOLINTBEGIN(readability-identifier-naming)

// ------------------------readable form of functions---------------------------
template <typename T, template <typename> class Alloc>
constexpr Tensor2d<T, Alloc> Tx(T linkLength) {
  return {
      {1, 0, 0, linkLength},
      {0, 1, 0, 0},
      {0, 0, 1, 0},
      {0, 0, 0, 1},
  };
};

template <typename T, template <typename> class Alloc>
constexpr Tensor2d<T, Alloc> Ty(T linkLength) {
  return {
      {1, 0, 0, 0},
      {0, 1, 0, linkLength},
      {0, 0, 1, 0},
      {0, 0, 0, 1},
  };
};

template <typename T, template <typename> class Alloc>
constexpr Tensor2d<T, Alloc> Tz(T linkLength) {
  return {
      {1, 0, 0, 0},
      {0, 1, 0, 0},
      {0, 0, 1, linkLength},
      {0, 0, 0, 1},
  };
};

template <typename T, template <typename> class Alloc>
constexpr Tensor2d<T, Alloc> Rx(T q) {
  auto cq{std::cos(q)};
  auto sq{std::sin(q)};
  return {
      {1, 0, 0, 0},
      {0, cq, -sq, 0},
      {0, sq, cq, 0},
      {0, 0, 0, 1},
  };
};

template <typename T, template <typename> class Alloc>
constexpr Tensor2d<T, Alloc> Ry(T q) {
  auto cq{std::cos(q)};
  auto sq{std::sin(q)};
  return {
      {cq, 0, sq, 0},
      {0, 1, 0, 0},
      {-sq, 0, cq, 0},
      {0, 0, 0, 1},
  };
};

template <typename T, template <typename> class Alloc>
constexpr Tensor2d<T, Alloc> Rz(T q) {
  auto cq{std::cos(q)};
  auto sq{std::sin(q)};
  return {
      {cq, -sq, 0, 0},
      {sq, cq, 0, 0},
      {0, 0, 1, 0},
      {0, 0, 0, 1},
  };
};

// ------------------------optimized form of functions--------------------------
template <typename T, template <typename> class Alloc>
constexpr Tensor2d<T, Alloc>& multipleByTMatrix(Tensor2d<T, Alloc>& tensor,
                                                std::size_t row,
                                                std::size_t column,
                                                T linkLength) {
  assert((tensor.size() == 4) && (tensor[0].size() == tensor.size()) &&
         "smth wrong");
  // tensor + tensor * {0 matrix except for row and column}
  static thread_local Tensor<T, Alloc<T>> tmp{0, 0, 0, 0};
  for (int i{0}; i < tensor.size(); ++i) {
    tmp[i] = tensor[i][row] * linkLength;
  }
  for (int i{0}; i < tensor.size(); ++i) {
    tensor[i][column] += tmp[i];
  }

  return tensor;
};

template <typename T, template <typename> class Alloc>
constexpr Tensor2d<T, Alloc>& TxO(Tensor2d<T, Alloc>& tensor, T linkLength) {
  return multipleByTMatrix(tensor, 0, 3, linkLength);
};

template <typename T, template <typename> class Alloc>
constexpr Tensor2d<T, Alloc>& TyO(Tensor2d<T, Alloc>& tensor, T linkLength) {
  return multipleByTMatrix(tensor, 1, 3, linkLength);
};

template <typename T, template <typename> class Alloc>
constexpr Tensor2d<T, Alloc>& TzO(Tensor2d<T, Alloc>& tensor, T linkLength) {
  return multipleByTMatrix(tensor, 2, 3, linkLength);
};

template <typename T, template <typename> class Alloc>
constexpr Tensor2d<T, Alloc>& RxO(Tensor2d<T, Alloc>& tensor, T q) {
  assert((tensor.size() == 4) && (tensor[0].size() == tensor.size()) &&
         "smth wrong");

  static thread_local auto tmp = Tensor2d<T, Alloc>(4, Tensor<T, Alloc<T>>(2));
  auto cq{std::cos(q)};
  auto sq{std::sin(q)};
  for (int i{0}; i < tensor.size(); ++i) {
    tmp[i][0] = tensor[i][1] * cq + tensor[i][2] * sq;
    tmp[i][1] = -tensor[i][1] * sq + tensor[i][2] * cq;
  }

  for (int i{0}; i < tensor.size(); ++i) {
    tensor[i][1] += tmp[i][0];
    tensor[i][2] += tmp[i][1];
  }

  return tensor;
};

template <typename T, template <typename> class Alloc>
constexpr Tensor2d<T, Alloc>& RyO(Tensor2d<T, Alloc>& tensor, T q) {
  assert((tensor.size() == 4) && (tensor[0].size() == tensor.size()) &&
         "smth wrong");

  static thread_local auto tmp = Tensor2d<T, Alloc>(4, Tensor<T, Alloc<T>>(2));
  auto cq{std::cos(q)};
  auto sq{std::sin(q)};
  for (int i{0}; i < tensor.size(); ++i) {
    tmp[i][0] = tensor[i][0] * cq - tensor[i][2] * sq;
    tmp[i][1] = tensor[i][0] * sq + tensor[i][2] * cq;
  }

  for (int i{0}; i < tensor.size(); ++i) {
    tensor[i][0] += tmp[i][0];
    tensor[i][2] += tmp[i][1];
  }

  return tensor;
};

template <typename T, template <typename> class Alloc>
constexpr Tensor2d<T, Alloc>& RzO(Tensor2d<T, Alloc>& tensor, T q) {
  assert((tensor.size() == 4) && (tensor[0].size() == tensor.size()) &&
         "smth wrong");

  static thread_local auto tmp = Tensor2d<T, Alloc>(4, Tensor<T, Alloc<T>>(2));
  auto cq{std::cos(q)};
  auto sq{std::sin(q)};
  for (int i{0}; i < tensor.size(); ++i) {
    tmp[i][0] = tensor[i][0] * cq + tensor[i][1] * sq;
    tmp[i][1] = -tensor[i][0] * sq + tensor[i][1] * cq;
  }

  for (int i{0}; i < tensor.size(); ++i) {
    tensor[i][0] += tmp[i][0];
    tensor[i][1] += tmp[i][1];
  }

  return tensor;
};

// NOLINTEND(readability-identifier-naming)

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
template <typename T, template <typename> class Alloc>
class KukaLbrIiwa {
 public:
  /// @brief xyz, ypr
  using Coordinates3d = Tensor<T, Alloc<T>>;
  constexpr static Tensor<Coordinates3d, Alloc<Coordinates3d>> operator()(
      const Tensor<T, Alloc<T>>& q) {
    const auto& d0{kLinkLengths[0]};
    const auto& d1{kLinkLengths[1]};
    const auto& d2{kLinkLengths[2]};
    const auto& d3{kLinkLengths[3]};
    const auto& d4{kLinkLengths[4]};
    const auto& d6{kLinkLengths[5]};
    const auto& q0{q[0]};
    const auto& q1{q[1]};
    const auto& q2{q[2]};
    const auto& q3{q[3]};
    const auto& q4{q[4]};
    const auto& q5{q[5]};
    const auto& q6{q[6]};

    // readable form:
    // T1{Ty(q0)};
    // T2 = T1 * Tz(d0);
    // T3 = T2 * Rz(q1) * Tx(d1);
    // T4 = T3 * Ry(q2) * Tx(d2);
    // T5 = T4 * Ry(q3) * Tx(d3) * Tz(d4);
    // T6 = T5 * Rx(q4) * Ry(q5) * Rx(q6);
    // T7 = T6 * Tx(d6);
    // optimized form:
    static thread_local auto T01 =
        Tensor2d<T, Alloc>(4, Tensor<T, Alloc<T>>(4));
    static thread_local auto T02 =
        Tensor2d<T, Alloc>(4, Tensor<T, Alloc<T>>(4));
    static thread_local auto T03 =
        Tensor2d<T, Alloc>(4, Tensor<T, Alloc<T>>(4));
    static thread_local auto T04 =
        Tensor2d<T, Alloc>(4, Tensor<T, Alloc<T>>(4));
    static thread_local auto T05 =
        Tensor2d<T, Alloc>(4, Tensor<T, Alloc<T>>(4));
    static thread_local auto T06 =
        Tensor2d<T, Alloc>(4, Tensor<T, Alloc<T>>(4));
    static thread_local auto T07 =
        Tensor2d<T, Alloc>(4, Tensor<T, Alloc<T>>(4));
    T01 = Ty(q0);
    T02 = T01;
    TzO(T02, d0);
    T03 = T02;
    TxO(RzO(T03, q1), d1);
    T04 = T03;
    TxO(RyO(T04, q2), d2);
    T05 = T04;
    TzO(TxO(RyO(T05, q3), d3), d4);
    T06 = T05;
    RxO(RyO(RxO(T06, q4), q5), q6);
    T07 = T06;
    TxO(T07, d6);

    auto Coordinates3dFromTransform =
        [](const Tensor2d<T, Alloc>& transform) -> Coordinates3d {
      return {transform[0][3],
              transform[1][3],
              transform[2][3],
              std::atan2(transform[2][0], transform[2][1]),
              std::atan2(transform[0][2], -transform[1][2]),
              std::atan2(std::sqrt(transform[0][2] * transform[0][2] +
                                   transform[1][2] * transform[1][2]),
                         transform[2][2])};
    };
    return {Coordinates3dFromTransform(T01), Coordinates3dFromTransform(T02),
            Coordinates3dFromTransform(T03), Coordinates3dFromTransform(T04),
            Coordinates3dFromTransform(T05), Coordinates3dFromTransform(T06),
            Coordinates3dFromTransform(T07)};
  }

  constexpr static Tensor<Coordinates3d, Alloc<Coordinates3d>>
  forwardKinematics(const Tensor<T, Alloc<T>>& q) {
    return operator()(q);
  }

  template <int I>
  constexpr static Coordinates3d getLinkPosition(const Tensor<T, Alloc<T>>& q) {
    return forwardKinematics(q)[I];
  }

 private:
  static const Tensor<T, Alloc<T>> kLinkLengths{0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
};
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)

}  //  namespace kuka
