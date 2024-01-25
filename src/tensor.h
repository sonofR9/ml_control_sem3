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

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <ostream>
#include <sstream>
#include <vector>

namespace optimization {
template <typename R, typename T>
concept ConvertibleInputRangeTo =
    std::ranges::input_range<R> &&
    std::convertible_to<std::ranges::range_value_t<R>, T>;

/**
 * @brief class to represent state of system using T variables
 *
 * @tparam T number of state variables
 */
template <typename T = double, class Alloc = std::allocator<T>>
struct Tensor {
  constexpr Tensor() noexcept = default;
  constexpr ~Tensor() noexcept = default;
  constexpr Tensor(const Tensor&) = default;
  constexpr Tensor(Tensor&&) noexcept = default;
  constexpr Tensor& operator=(const Tensor&) = default;
  constexpr Tensor& operator=(Tensor&&) noexcept = default;

  constexpr explicit Tensor(const std::size_t size) : data_(size) {
  }

  constexpr Tensor(const std::size_t size, T value) : data_(size, value) {
  }

  constexpr Tensor(std::initializer_list<T>&& list) : data_{std::move(list)} {
  }

  /*implicit*/ constexpr Tensor(ConvertibleInputRangeTo<T> auto&& range)
      : data_{range.begin(), range.end()} {
  }

  constexpr T& operator[](std::size_t i) noexcept {
    return data_[i];
  }

  constexpr T operator[](std::size_t i) const noexcept {
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

  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return data_.size();
  }

  [[nodiscard]] constexpr bool empty() const noexcept {
    return data_.empty();
  }

  constexpr void resize(const std::size_t newSize) {
    for (std::size_t i{size()}; i < newSize; ++i) {
      data_.emplace_back(T{});
    }
    while (size() > newSize) {
      data_.pop_back();
    }
  }

 private:
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & data_;
  }

  std::vector<T, Alloc> data_;
};

// ------------------------------ iterators -----------------------------------

template <typename T, class Alloc>
class Tensor<T, Alloc>::ConstIterator {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = const T*;
  using reference = const T&;

  constexpr ConstIterator() noexcept = default;

  constexpr explicit ConstIterator(pointer ptr) noexcept : ptr_{ptr} {
  }

  constexpr ConstIterator(pointer ptr, const difference_type offset) noexcept
      : ptr_{ptr + offset} {
  }

  [[nodiscard]] constexpr reference operator*() const noexcept {
    return *ptr_;
  }
  [[nodiscard]] constexpr pointer operator->() const noexcept {
    return ptr_.operator->();
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
    return ConstIterator{ptr_ + n};
  }

  [[nodiscard]] friend constexpr ConstIterator operator+(
      const difference_type n, ConstIterator iter) noexcept {
    iter += n;
    return iter;
  }

  [[nodiscard]] constexpr ConstIterator operator-(
      const difference_type n) const noexcept {
    return ConstIterator{ptr_ - n};
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
    return ptr_[offset];
  }

  friend auto operator<=>(const ConstIterator&,
                          const ConstIterator&) noexcept = default;

 private:
  constexpr explicit ConstIterator(
      std::vector<T, Alloc>::const_iterator iter) noexcept
      : ptr_{iter} {
  }

  friend Tensor<T, Alloc>;
  std::vector<T, Alloc>::const_iterator ptr_{};
};

template <typename T, class Alloc>
class Tensor<T, Alloc>::Iterator : public Tensor<T, Alloc>::ConstIterator {
  //   using Base = Tensor<T, Alloc>::ConstIterator;

 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  constexpr Iterator() noexcept = default;

  constexpr explicit Iterator(pointer ptr) noexcept : ptr_{ptr} {
  }

  constexpr explicit Iterator(pointer ptr,
                              const difference_type offset) noexcept
      : ptr_{ptr + offset} {
  }

  constexpr explicit Iterator(std::vector<T, Alloc>::iterator iter) noexcept
      : ptr_{std::move(iter)} {
  }

  [[nodiscard]] constexpr reference operator*() const noexcept {
    return *ptr_;
  }
  [[nodiscard]] constexpr pointer operator->() const noexcept {
    return ptr_.operator->();
  }

  constexpr Iterator& operator++() noexcept {
    ++ptr_;
    return *this;
  }

  constexpr Iterator operator++(int) noexcept {
    Iterator tmp = *this;
    ++ptr_;
    return tmp;
  }

  constexpr Iterator& operator--() noexcept {
    --ptr_;
    return *this;
  }

  constexpr Iterator operator--(int) noexcept {
    Iterator tmp = *this;
    --ptr_;
    return tmp;
  }

  [[nodiscard]] constexpr Iterator operator+(
      const difference_type n) const noexcept {
    return Iterator{ptr_ + n};
  }

  [[nodiscard]] friend constexpr Iterator operator+(const difference_type n,
                                                    Iterator iter) noexcept {
    iter += n;
    return iter;
  }

  [[nodiscard]] constexpr Iterator operator-(
      const difference_type n) const noexcept {
    return Iterator{ptr_ - n};
  }

  [[nodiscard]] friend constexpr Iterator operator-(const difference_type n,
                                                    Iterator iter) noexcept {
    iter -= n;
    return iter;
  }

  [[nodiscard]] constexpr difference_type operator-(
      const Iterator& other) const noexcept {
    return ptr_ - other.ptr_;
  }

  constexpr Iterator& operator+=(difference_type n) noexcept {
    ptr_ += n;
    return *this;
  }

  constexpr Iterator& operator-=(difference_type n) noexcept {
    ptr_ -= n;
    return *this;
  }

  [[nodiscard]] constexpr reference operator[](
      const ptrdiff_t offset) const noexcept {
    return ptr_[offset];
  }

  friend auto operator<=>(const Iterator&, const Iterator&) noexcept = default;

 private:
  friend Tensor<T, Alloc>;
  typename std::vector<T, Alloc>::iterator ptr_{};
};

template <typename T, class Alloc>
constexpr Tensor<T, Alloc>::Iterator Tensor<T, Alloc>::begin() noexcept {
  return Iterator{data_.begin()};
}

template <typename T, class Alloc>
constexpr Tensor<T, Alloc>::Iterator Tensor<T, Alloc>::end() noexcept {
  return Iterator{data_.end()};
}

template <typename T, class Alloc>
constexpr Tensor<T, Alloc>::ConstIterator Tensor<T, Alloc>::begin()
    const noexcept {
  return ConstIterator{data_.begin()};
}

template <typename T, class Alloc>
constexpr Tensor<T, Alloc>::ConstIterator Tensor<T, Alloc>::end()
    const noexcept {
  return ConstIterator{data_.end()};
}

template <typename T, class Alloc>
constexpr Tensor<T, Alloc>::ConstIterator Tensor<T, Alloc>::cbegin()
    const noexcept {
  return ConstIterator{data_.cbegin()};
}

template <typename T, class Alloc>
constexpr Tensor<T, Alloc>::ConstIterator Tensor<T, Alloc>::cend()
    const noexcept {
  return ConstIterator{data_.cend()};
}

template <typename T, class Alloc, class Alloc2>
constexpr Tensor<T, Alloc> operator+(const Tensor<T, Alloc>& self,
                                     const Tensor<T, Alloc2>& other) {
  assert((self.size() == other.size()));
  auto result = Tensor<T, Alloc>(self.size());
  for (std::size_t i{0}; i < self.size(); ++i) {
    result[i] = other[i] + self[i];
  }
  return result;
}

template <typename T, class Alloc, class Alloc2>
constexpr Tensor<T, Alloc>& operator+=(
    Tensor<T, Alloc>& self, const Tensor<T, Alloc2>& other) noexcept {
  assert((self.size() == other.size()));
  for (std::size_t i{0}; i < self.size(); ++i) {
    self[i] += other[i];
  }
  return self;
}

template <typename T, class Alloc, class Alloc2>
constexpr Tensor<T, Alloc> operator-(const Tensor<T, Alloc>& self,
                                     const Tensor<T, Alloc2>& other) {
  assert((self.size() == other.size()));
  auto result = Tensor<T, Alloc>(self.size());
  for (std::size_t i{0}; i < self.size(); ++i) {
    result[i] = self[i] - other[i];
  }
  return result;
}

template <typename T, class Alloc>
constexpr Tensor<T, Alloc> operator-(const Tensor<T, Alloc>& self) {
  auto result = Tensor<T, Alloc>(self.size());
  for (std::size_t i{0}; i < self.size(); ++i) {
    result[i] = -self[i];
  }
  return result;
}

template <typename T, class Alloc, class Alloc2>
constexpr Tensor<T, Alloc>& operator-=(
    Tensor<T, Alloc>& self, const Tensor<T, Alloc2>& other) noexcept {
  assert((self.size() == other.size()));
  for (std::size_t i{0}; i < self.size(); ++i) {
    self[i] -= other[i];
  }
  return self;
}


// TODO(novak) add *= overloads too
template <typename T>
concept Subscriptable = requires(T t) { t[0]; };

template <Subscriptable T, class Alloc>
constexpr Tensor<T, Alloc<T>> operator*(const Tensor<T, Alloc>& self, 
                                        const Tensor<T, Alloc>& other) {
  assert((self[0].size() == other.size()) && "dimensions must conform to matrix multiplication rule");
  // mxn * nxp
  int m{self.size()};
  int n{self[0].size()};
  int p{other[0].size()};

  Tensor<T, Alloc> result{m, T(p, 0)};

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < p; ++j) {
      for (int k = 0; k < n; ++k) {
        result[i][j] += self[i][k] * other[k][j];
      }
    }
  }

  return result;
}

// overloads of operator* for vectors multiplication
template <typename T>
concept NotSubscriptable = !Subscriptable<T>;

template <typename T, template <typename> class Alloc, NotSubscriptable U>
constexpr Tensor<T, Alloc<T>> operator*(const Tensor<T, Alloc<T>>& self, 
                                        const Tensor<U, Alloc<U>>& other) {
  // case mx1 * 1xp
  assert((self[0].size() == 1) && "dimensions must conform to matrix multiplication rule");
  int m{self.size()};
  int p{other.size()};

  Tensor<T, Alloc<T>> result{m, T(p, 0)};

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < p; ++j) {
      result[i][j] = self[i][0] * other[j];
    }
  }

  return result;
}

template <typename T, template <typename> class Alloc, NotSubscriptable U>
constexpr U operator*(const Tensor<U, Alloc<U>>& self, 
                      const Tensor<T, Alloc<T>>& other) {
  // case 1xm * mx1
  assert((other[0].size() == 1) && "dimensions must conform to matrix multiplication rule");
  assert((other.size() == self.size()) && "dimensions must conform to matrix multiplication rule");
  int m{self.size()};

  U result{0};

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      result += self[i] * other[j][0];
    }
  }

  return result;
}

template <typename T, class Alloc, typename M>
constexpr Tensor<T, Alloc> operator*(const Tensor<T, Alloc>& self,
                                     M multiplier) {
  auto result = Tensor<T, Alloc>(self.size());
  for (std::size_t i{0}; i < self.size(); ++i) {
    result[i] = multiplier * self[i];
  }
  return result;
}

template <typename T, class Alloc, typename M>
constexpr Tensor<T, Alloc> operator*(M multiplier,
                                     const Tensor<T, Alloc>& self) {
  return self * multiplier;
}

template <typename T, class Alloc, typename M>
constexpr Tensor<T, Alloc>& operator*=(Tensor<T, Alloc>& self,
                                       M multiplier) noexcept {
  for (std::size_t i{0}; i < self.size(); ++i) {
    self[i] *= multiplier;
  }
  return self;
}

template <typename T, class Alloc, typename M>
constexpr Tensor<T, Alloc> operator/(const Tensor<T, Alloc>& self, M divider) {
  auto result = Tensor<T, Alloc>(self.size());
  for (std::size_t i{0}; i < self.size(); ++i) {
    result[i] = self[i] / divider;
  }
  return result;
}

template <typename T, class Alloc, typename M>
constexpr Tensor<T, Alloc>& operator/=(Tensor<T, Alloc>& self,
                                       M divider) noexcept {
  for (std::size_t i{0}; i < self.size(); ++i) {
    self[i] /= divider;
  }
  return self;
}

template <typename T, class Alloc, class Alloc2>
constexpr bool operator==(const Tensor<T, Alloc>& lhs,
                          const Tensor<T, Alloc2>& rhs) {
  assert((lhs.size() == rhs.size()));
  constexpr double kEps{1e-3};
  auto diff{lhs - rhs};
  return !std::ranges::any_of(
      diff, [kEps](const T& value) { return fabs(value) > kEps; });
}

template <typename T, class Alloc, class Alloc2>
constexpr bool operator!=(const Tensor<T, Alloc>& lhs,
                          const Tensor<T, Alloc2>& rhs) {
  return !(lhs == rhs);
}

template <NotSubscriptable U, class Alloc>
constexpr U dotProduct(const Tensor<U, Alloc>& lhs, const Tensor<U, Alloc>& rhs) {
  assert((lhs.size() == rhs.size()) && "dimensions must be the same");
  U result{0};
  for (std::size_t i{0}; i < lhs.size(); ++i) {
    result += lhs[i] * rhs[i];
  }
  return result;
}

template <NotSubscriptable U, template <typename> class Alloc>
constexpr Tensor<Tensor<U, Alloc<U>>, Alloc<Tensor<U, Alloc<U>>>> transposeAndMultiple(const Tensor<U, Alloc<U>>& lhs, const Tensor<U, Alloc<U>>& rhs) {
  Tensor<Tensor<U, Alloc<U>>, Alloc<Tensor<U, Alloc<U>>>> result{lhs.size(), Tensor<U, Alloc<U>>(rhs.size())};
  for (std::size_t i{0}; i < lhs.size(); ++i) {
    for (std::size_t j{0}; j < rhs.size(); ++j) {
      result[i][j] = lhs[i] * rhs[j];
    }
  }
  return result;
}

// TODO(novak) add transpose function
}  // namespace optimization

template <typename T, class Alloc>
std::ostream& operator<<(std::ostream& stream,
                         const optimization::Tensor<T, Alloc>& state) {
  for (std::size_t i{0}; i < state.size(); ++i) {
    stream << state[i] << " ";
  }
  return stream;
}

template <typename T, class Alloc>
std::stringstream& operator<<(std::stringstream& stream,
                              const optimization::Tensor<T, Alloc>& state) {
  for (std::size_t i{0}; i < state.size(); ++i) {
    stream << state[i] << " ";
  }
  return stream;
}

static_assert(std::ranges::range<optimization::Tensor<int>>);
static_assert(std::ranges::random_access_range<optimization::Tensor<int>>);
