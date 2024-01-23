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

#include "utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <ostream>
#include <ranges>
#include <sstream>

namespace optimization {
/**
 * @brief class to represent state of system using T variables
 *
 * @tparam T number of state variables
 */
template <int N, typename T = double>
struct StaticTensor {
  constexpr StaticTensor() noexcept = default;
  constexpr ~StaticTensor() noexcept = default;
  constexpr StaticTensor(const StaticTensor&) = default;
  constexpr StaticTensor(StaticTensor&&) noexcept = default;
  constexpr StaticTensor& operator=(const StaticTensor&) = default;
  constexpr StaticTensor& operator=(StaticTensor&&) noexcept = default;

  constexpr StaticTensor(const std::initializer_list<T>& list) {
    if (list.size() != N) {
      throw std::length_error("Initializer list size differs from data size");
    }

    auto el = list.begin();
    for (int i{0}; i < N; ++i) {
      data_[i] = *(el++);
    }
  }

  /*implicit*/ constexpr StaticTensor(ConvertibleInputRangeTo<T> auto&& range) {
    if (range.size() != N) {
      throw std::length_error("Range size differs from data size");
    }
    std::copy(std::ranges::begin(range), std::ranges::end(range),
              data_.begin());
  }

  constexpr T& operator[](int i) noexcept {
    return data_[i];
  }
  constexpr T operator[](int i) const noexcept {
    return data_[i];
  }

  class Iterator;
  class ConstIterator;

  constexpr Iterator begin() noexcept;
  constexpr Iterator end() noexcept;
  [[nodiscard]] constexpr ConstIterator begin() const noexcept;
  [[nodiscard]] constexpr ConstIterator end() const noexcept;

  [[nodiscard]] constexpr ConstIterator cbegin() const noexcept;
  [[nodiscard]] constexpr ConstIterator cend() const noexcept;

  [[nodiscard]] static constexpr int size() const noexcept {
    return N;
  }

 private:
  std::array<T, N> data_;
};

// ------------------------------ iterators -----------------------------------

template <int N, typename T>
class StaticTensor<N, T>::ConstIterator {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = const T*;
  using reference = const T&;

  constexpr ConstIterator() noexcept = default;

  constexpr explicit ConstIterator(pointer ptr,
                                   difference_type offset = 0) noexcept
      : ptr_{ptr + offset} {
  }

  [[nodiscard]] constexpr reference operator*() const noexcept {
    return *ptr_;
  }
  [[nodiscard]] constexpr pointer operator->() const noexcept {
    return ptr_;
  }

  constexpr ConstIterator& operator++() noexcept {
    ++ptr_;
    return *this;
  }

  constexpr ConstIterator operator++(int) noexcept {
    ConstIterator tmp = *this;
    ++ptr_;
    return tmp;
  }

  constexpr ConstIterator& operator--() noexcept {
    --ptr_;
    return *this;
  }

  constexpr ConstIterator operator--(int) noexcept {
    ConstIterator tmp = *this;
    --ptr_;
    return tmp;
  }

  [[nodiscard]] constexpr ConstIterator operator+(
      const difference_type n) const noexcept {
    return {ptr_ + n};
  }

  [[nodiscard]] friend constexpr ConstIterator operator+(
      const difference_type n, ConstIterator iter) noexcept {
    iter += n;
    return iter;
  }

  [[nodiscard]] constexpr ConstIterator operator-(
      const difference_type n) const noexcept {
    return {ptr_, -n};
  }

  [[nodiscard]] friend constexpr ConstIterator operator-(
      const difference_type n, ConstIterator iter) noexcept {
    iter -= n;
    return iter;
  }

  [[nodiscard]] constexpr difference_type operator-(
      const ConstIterator& other) const noexcept {
    return ptr_ - other.ptr_;
  }

  constexpr ConstIterator& operator+=(difference_type n) noexcept {
    ptr_ += n;
    return *this;
  }

  constexpr ConstIterator& operator-=(difference_type n) noexcept {
    ptr_ -= n;
    return *this;
  }

  [[nodiscard]] constexpr reference operator[](
      const ptrdiff_t offset) const noexcept {
    return {ptr_, offset};
  }

  friend auto operator<=>(const ConstIterator&,
                          const ConstIterator&) noexcept = default;

 private:
  pointer ptr_{nullptr};
};

template <int N, typename T>
class StaticTensor<N, T>::Iterator : public StaticTensor<N, T>::ConstIterator {
  using Base = StaticTensor<N, T>::ConstIterator;

 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  constexpr Iterator() noexcept = default;

  constexpr explicit Iterator(pointer ptr, difference_type offset = 0) noexcept
      : Base(ptr, offset) {
  }

  [[nodiscard]] constexpr reference operator*() const noexcept {
    return const_cast<reference>(Base::operator*());
  }

  [[nodiscard]] constexpr pointer operator->() const noexcept {
    return const_cast<pointer>(Base::operator->());
  }

  constexpr Iterator& operator++() noexcept {
    Base::operator++();
    return *this;
  }

  constexpr Iterator operator++(int) noexcept {
    Iterator tmp = *this;
    Base::operator++();
    return tmp;
  }

  constexpr Iterator& operator--() noexcept {
    Base::operator--();
    return *this;
  }

  constexpr Iterator operator--(int) noexcept {
    Iterator tmp = *this;
    Base::operator--();
    return tmp;
  }

  [[nodiscard]] constexpr Iterator operator+(
      const difference_type n) const noexcept {
    Iterator tmp = *this;
    tmp += n;
    return tmp;
  }

  [[nodiscard]] friend constexpr Iterator operator+(const difference_type n,
                                                    Iterator iter) noexcept {
    iter += n;
    return iter;
  }

  [[nodiscard]] constexpr Iterator operator-(
      const difference_type n) const noexcept {
    Iterator tmp = *this;
    tmp -= n;
    return tmp;
  }

  [[nodiscard]] friend constexpr Iterator operator-(const difference_type n,
                                                    Iterator iter) noexcept {
    iter -= n;
    return iter;
  }

  [[nodiscard]] constexpr difference_type operator-(
      const Iterator& other) const noexcept {
    return Base::operator-(other);
  }

  constexpr Iterator& operator+=(difference_type n) noexcept {
    Base::operator+=(n);
    return *this;
  }

  constexpr Iterator& operator-=(difference_type n) noexcept {
    Base::operator-=(n);
    return *this;
  }

  [[nodiscard]] constexpr reference operator[](
      const ptrdiff_t offset) const noexcept {
    return const_cast<reference>(Base::operator[](offset));
  }

  friend auto operator<=>(const Iterator&, const Iterator&) noexcept = default;
};

template <int N, typename T>
constexpr typename StaticTensor<N, T>::Iterator
StaticTensor<N, T>::begin() noexcept {
  return Iterator{&data_[0]};
}

template <int N, typename T>
constexpr typename StaticTensor<N, T>::Iterator
StaticTensor<N, T>::end() noexcept {
  return Iterator{&data_[N - 1] + 1};
}

template <int N, typename T>
constexpr typename StaticTensor<N, T>::ConstIterator StaticTensor<N, T>::begin()
    const noexcept {
  return ConstIterator{&data_[0]};
}

template <int N, typename T>
constexpr typename StaticTensor<N, T>::ConstIterator StaticTensor<N, T>::end()
    const noexcept {
  return ConstIterator{&data_[N - 1] + 1};
}

template <int N, typename T>
constexpr typename StaticTensor<N, T>::ConstIterator
StaticTensor<N, T>::cbegin() const noexcept {
  return ConstIterator{&data_[0]};
}

template <int N, typename T>
constexpr typename StaticTensor<N, T>::ConstIterator StaticTensor<N, T>::cend()
    const noexcept {
  return ConstIterator{&data_[N - 1] + 1};
}

template <int N, typename T>
constexpr StaticTensor<N, T> operator+(const StaticTensor<N, T>& self,
                                       const StaticTensor<N, T>& other) {
  StaticTensor<N, T> result;
  for (int i{0}; i < N; ++i) {
    result[i] = other[i] + self[i];
  }
  return result;
}

template <int N, typename T>
constexpr StaticTensor<N, T>& operator+=(
    StaticTensor<N, T>& self, const StaticTensor<N, T>& other) noexcept {
  for (int i{0}; i < N; ++i) {
    self[i] += other[i];
  }
  return self;
}

template <int N, typename T>
constexpr StaticTensor<N, T> operator-(const StaticTensor<N, T>& self,
                                       const StaticTensor<N, T>& other) {
  StaticTensor<N, T> result;
  for (int i{0}; i < N; ++i) {
    result[i] = self[i] - other[i];
  }
  return result;
}

template <int N, typename T>
constexpr StaticTensor<N, T> operator-(const StaticTensor<N, T>& self) {
  StaticTensor<N, T> result;
  for (int i{0}; i < N; ++i) {
    result[i] = -self[i];
  }
  return result;
}

template <int N, typename T>
constexpr StaticTensor<N, T>& operator-=(
    StaticTensor<N, T>& self, const StaticTensor<N, T>& other) noexcept {
  for (int i{0}; i < N; ++i) {
    self[i] -= other[i];
  }
  return self;
}

template <int N, typename T, typename M>
constexpr StaticTensor<N, T> operator*(const StaticTensor<N, T>& self,
                                       M multiplier) {
  StaticTensor<N, T> result;
  for (int i{0}; i < N; ++i) {
    result[i] = multiplier * self[i];
  }
  return result;
}

template <int N, typename T, typename M>
constexpr StaticTensor<N, T> operator*(M multiplier,
                                       const StaticTensor<N, T>& self) {
  return self * multiplier;
}

template <int N, typename T>
constexpr StaticTensor<N, T>& operator*=(
    StaticTensor<N, T>& self, const StaticTensor<N, T>& other) noexcept {
  for (int i{0}; i < N; ++i) {
    self[i] *= other[i];
  }
  return self;
}

template <int N, typename T, typename M>
constexpr StaticTensor<N, T> operator/(const StaticTensor<N, T>& self,
                                       M divider) {
  StaticTensor<N, T> result;
  for (int i{0}; i < N; ++i) {
    result[i] = self[i] / divider;
  }
  return result;
}

template <int N, typename T>
constexpr bool operator==(const StaticTensor<N, T>& lhs,
                          const StaticTensor<N, T>& rhs) {
  constexpr double kEps{1e-3};
  auto diff{lhs - rhs};
  return !std::any_of(diff.begin(), diff.end(),
                      [kEps](const T& value) { return fabs(value) > kEps; });
}

template <int N, typename T>
constexpr bool operator!=(const StaticTensor<N, T>& lhs,
                          const StaticTensor<N, T>& rhs) {
  return !(lhs == rhs);
}
}  // namespace optimization

template <int N, typename T>
std::ostream& operator<<(std::ostream& stream,
                         const optimization::StaticTensor<N, T>& state) {
  for (int i{0}; i < N; ++i) {
    stream << state[i] << " ";
  }
  return stream;
}

template <int N, typename T>
std::stringstream& operator<<(std::stringstream& stream,
                              const optimization::StaticTensor<N, T>& state) {
  for (int i{0}; i < N; ++i) {
    stream << state[i] << " ";
  }
  return stream;
}

// constexpr void fun() {
//   constexpr optimization::StaticTensor<3, int> a{};
//   constexpr std::array<int, 3> v{};
//   static_assert(std::ranges::begin(v) == v.begin());

//   constexpr std::ranges::range<optimization::StaticTensor<3, int>> r{a};

//   static_assert(std::begin(a) == a.begin());
//   static_assert(std::ranges::begin(a) == a.begin());
// }

template <class _Rng>
concept range_test_v2 = requires(_Rng& __r) { ::std::ranges::begin(__r); };
static_assert(range_test_v2<optimization::StaticTensor<3, int>>);

static_assert(
    std::ranges::random_access_range<optimization::StaticTensor<100, int>>);
