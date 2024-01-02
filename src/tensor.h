#pragma once

#include "utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <ostream>
#include <sstream>
#include <vector>

namespace optimization {
/**
 * @brief class to represent state of system using T variables
 *
 * @tparam T number of state variables
 */
template <typename T = double>
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

 private:
  std::vector<T> data_;
};

// ------------------------------ iterators -----------------------------------

template <typename T>
class Tensor<T>::ConstIterator {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = const T*;
  using reference = const T&;

  constexpr ConstIterator() noexcept = default;

  constexpr explicit ConstIterator(pointer ptr) noexcept : ptr_{ptr} {
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
    return ptr_[offset];
  }

  friend auto operator<=>(const ConstIterator&,
                          const ConstIterator&) noexcept = default;

 private:
  constexpr explicit ConstIterator(std::vector<T>::const_iterator iter) noexcept
      : ptr_{iter} {
  }

  friend Tensor<T>;
  std::vector<T>::const_iterator ptr_{};
};

template <typename T>
class Tensor<T>::Iterator : public Tensor<T>::ConstIterator {
  //   using Base = Tensor<T>::ConstIterator;

 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  constexpr Iterator() noexcept = default;

  constexpr explicit Iterator(pointer ptr) noexcept : ptr_{ptr} {
  }

  constexpr explicit Iterator(std::vector<T>::iterator iter) noexcept
      : ptr_{std::move(iter)} {
  }

  [[nodiscard]] constexpr reference operator*() const noexcept {
    return *ptr_;
  }
  [[nodiscard]] constexpr pointer operator->() const noexcept {
    return ptr_;
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
    return {ptr_ + n};
  }

  [[nodiscard]] friend constexpr Iterator operator+(const difference_type n,
                                                    Iterator iter) noexcept {
    iter += n;
    return iter;
  }

  [[nodiscard]] constexpr Iterator operator-(
      const difference_type n) const noexcept {
    return {ptr_, -n};
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
  friend Tensor<T>;
  typename std::vector<T>::iterator ptr_{};
};

template <typename T>
constexpr typename Tensor<T>::Iterator Tensor<T>::begin() noexcept {
  return Iterator{data_.begin()};
}

template <typename T>
constexpr typename Tensor<T>::Iterator Tensor<T>::end() noexcept {
  return Iterator{data_.end()};
}

template <typename T>
constexpr typename Tensor<T>::ConstIterator Tensor<T>::begin() const noexcept {
  return ConstIterator{data_.begin()};
}

template <typename T>
constexpr typename Tensor<T>::ConstIterator Tensor<T>::end() const noexcept {
  return ConstIterator{data_.end()};
}

template <typename T>
constexpr typename Tensor<T>::ConstIterator Tensor<T>::cbegin() const noexcept {
  return ConstIterator{data_.cbegin()};
}

template <typename T>
constexpr typename Tensor<T>::ConstIterator Tensor<T>::cend() const noexcept {
  return ConstIterator{data_.cend()};
}

template <typename T>
constexpr Tensor<T> operator+(const Tensor<T>& self, const Tensor<T>& other) {
  assert((self.size() == other.size()));
  auto result = Tensor<T>(self.size());
  for (std::size_t i{0}; i < self.size(); ++i) {
    result[i] = other[i] + self[i];
  }
  return result;
}

template <typename T>
constexpr Tensor<T>& operator+=(Tensor<T>& self,
                                const Tensor<T>& other) noexcept {
  assert((self.size() == other.size()));
  for (std::size_t i{0}; i < self.size(); ++i) {
    self[i] += other[i];
  }
  return self;
}

template <typename T>
constexpr Tensor<T> operator-(const Tensor<T>& self, const Tensor<T>& other) {
  assert((self.size() == other.size()));
  auto result = Tensor<T>(self.size());
  for (std::size_t i{0}; i < self.size(); ++i) {
    result[i] = self[i] - other[i];
  }
  return result;
}

template <typename T>
constexpr Tensor<T> operator-(const Tensor<T>& self) {
  auto result = Tensor<T>(self.size());
  for (std::size_t i{0}; i < self.size(); ++i) {
    result[i] = -self[i];
  }
  return result;
}

template <typename T>
constexpr Tensor<T>& operator-=(Tensor<T>& self,
                                const Tensor<T>& other) noexcept {
  assert((self.size() == other.size()));
  for (std::size_t i{0}; i < self.size(); ++i) {
    self[i] -= other[i];
  }
  return self;
}

template <typename T, typename M>
constexpr Tensor<T> operator*(const Tensor<T>& self, M multiplier) {
  auto result = Tensor<T>(self.size());
  for (std::size_t i{0}; i < self.size(); ++i) {
    result[i] = multiplier * self[i];
  }
  return result;
}

template <typename T, typename M>
constexpr Tensor<T> operator*(M multiplier, const Tensor<T>& self) {
  return self * multiplier;
}

template <typename T>
constexpr Tensor<T>& operator*=(Tensor<T>& self,
                                const Tensor<T>& other) noexcept {
  assert((self.size() == other.size()));
  for (std::size_t i{0}; i < self.size(); ++i) {
    self[i] *= other[i];
  }
  return self;
}

template <typename T, typename M>
constexpr Tensor<T> operator/(const Tensor<T>& self, M divider) {
  auto result = Tensor<T>(self.size());
  for (std::size_t i{0}; i < self.size(); ++i) {
    result[i] = self[i] / divider;
  }
  return result;
}

template <typename T>
constexpr bool operator==(const Tensor<T>& lhs, const Tensor<T>& rhs) {
  assert((lhs.size() == rhs.size()));
  constexpr double kEps{1e-3};
  auto diff{lhs - rhs};
  return !std::ranges::any_of(
      diff, [kEps](const T& value) { return fabs(value) > kEps; });
}

template <typename T>
constexpr bool operator!=(const Tensor<T>& lhs, const Tensor<T>& rhs) {
  return !(lhs == rhs);
}
}  // namespace optimization

template <typename T>
std::ostream& operator<<(std::ostream& stream,
                         const optimization::Tensor<T>& state) {
  for (std::size_t i{0}; i < state.size(); ++i) {
    stream << state[i] << " ";
  }
  return stream;
}

template <typename T>
std::stringstream& operator<<(std::stringstream& stream,
                              const optimization::Tensor<T>& state) {
  for (std::size_t i{0}; i < state.size(); ++i) {
    stream << state[i] << " ";
  }
  return stream;
}

static_assert(std::ranges::range<optimization::Tensor<int>>);
static_assert(std::ranges::random_access_range<optimization::Tensor<int>>);
