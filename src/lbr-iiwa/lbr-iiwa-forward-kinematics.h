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

template <typename T, class Alloc>
constexpr Tensor<T, Alloc> FK(const Tensor<T, Alloc>& q) {
  static Tensor<T, Alloc> link_lengths{0.1, 0.2, 0.3, 0.4, 0.5, 0.6};

  auto d0{link_lengths[0]};
  auto d1{link_lengths[1]};
  auto d2{link_lengths[2]};
  auto d3{link_lengths[3]};
  auto d4{link_lengths[4]};
  auto d6{link_lengths[5]};
  auto q0{q[0]};
  auto q1{q[1]};
  auto q2{q[2]};
  auto q3{q[3]};
  auto q4{q[4]};
  auto q5{q[5]};
  auto q6{q[6]};

  // readable form:
  // auto tBuf{Ty(q0)};
  // tBuf = tBuf * Tz(d0);
  // tBuf = tBuf * Rz(q1) * Tx(d1);
  // tBuf = tBuf * Ry(q2) * Tx(d2);
  // tBuf = tBuf * Ry(q3) * Tx(d3) * Tz(d4);
  // tBuf = tBuf * Rx(q4) * Ry(q5) * Rx(q6);
  // tBuf = tBuf * Tx(d6);
  // optimized form:
  static thread_local auto tBuf{Ty(q0)};
  tBuf = Ty(q0);
  TzO(tBuf, d0);
  TxO(RzO(tBuf, q1), d1);
  TxO(RyO(tBuf, q2), d2);
  TzO(TxO(RyO(tBuf, q3), d3), d4);
  RxO(RyO(RxO(tBuf, q4), q5), q6);
  TxO(tBuf, d6);

  return {
      tBuf[0][3],
      tBuf[1][3],
      tBuf[2][3],
      std::atan2(tBuf[2][0], tBuf[2][1]),
      std::atan2(tBuf[0][2], -tBuf[1][2]),
      std::atan2(std::sqrt(tBuf[0][2] * tBuf[0][2] + tBuf[1][2] * tBuf[1][2]),
                 tBuf[2][2])};
}

}  //  namespace kuka
