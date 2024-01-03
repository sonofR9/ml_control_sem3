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
template <typename T = double, class Alloc = std::allocator<T>>
struct Tensor {
  constexpr Tensor() noexcept = delete;

  constexpr ~Tensor() noexcept {
    delete[] data_;
    data_ = nullptr;
  }

  constexpr Tensor(const Tensor& arr) : size_{arr.size_}, data_{new T[size_]} {
    std::copy(arr.begin(), arr.end(), begin());
  }

  constexpr Tensor(Tensor&& arr) noexcept : size_{arr.size_}, data_{arr.data_} {
    arr.data_ = nullptr;
  }

  constexpr Tensor& operator=(const Tensor& arr) {
    if (this != &arr) {
      delete[] data_;
      size_ = arr.size_;
      data_ = new T[size_];
      std::copy(arr.begin(), arr.end(), begin());
    }
    return *this;
  }
  constexpr Tensor& operator=(Tensor&& arr) noexcept {
    delete[] data_;
    data_ = arr.data_;
    size_ = arr.size_;
    arr.data_ = nullptr;
    return *this;
  }

  constexpr explicit Tensor(const std::size_t size)
      : size_{size}, data_{new T[size_]} {
  }

  constexpr Tensor(const std::size_t size, T value)
      : size_{size}, data_{new T[size_]} {
    std::fill(begin(), end(), value);
  }

  constexpr Tensor(std::initializer_list<T>&& list)
      : size_{list.size()}, data_{new T[size_]} {
    std::move(list.begin(), list.end(), begin());
  }

  /*implicit*/ constexpr Tensor(ConvertibleSizedInputRangeTo<T> auto&& range)
      : size_{range.size()}, data_{new T[size_]} {
    std::move(range.begin(), range.end(), begin());
  }

  constexpr T& operator[](std::size_t i) noexcept {
    assert((i < size_));
    return data_[i];
  }
  constexpr T operator[](std::size_t i) const noexcept {
    assert((i < size_));
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
    return size_;
  }

 private:
  std::size_t size_{0};
  T* data_{nullptr};
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

  constexpr ConstIterator(pointer ptr, const std::size_t offset) noexcept
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
    return ptr_[offset];
  }

  friend auto operator<=>(const ConstIterator&,
                          const ConstIterator&) noexcept = default;

 private:
  T* ptr_{};
};

template <typename T>
class Tensor<T>::Iterator : public Tensor<T>::ConstIterator {
  using Base = Tensor<T>::ConstIterator;

 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  constexpr Iterator() noexcept = default;

  constexpr explicit Iterator(pointer ptr) noexcept : Base{ptr} {
  }

  constexpr Iterator(pointer ptr, const std::size_t offset) noexcept
      : Base{ptr, offset} {
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

template <typename T>
constexpr typename Tensor<T>::Iterator Tensor<T>::begin() noexcept {
  return Iterator{data_};
}

template <typename T>
constexpr typename Tensor<T>::Iterator Tensor<T>::end() noexcept {
  return Iterator{data_ + size_};
}

template <typename T>
constexpr typename Tensor<T>::ConstIterator Tensor<T>::begin() const noexcept {
  return ConstIterator{data_};
}

template <typename T>
constexpr typename Tensor<T>::ConstIterator Tensor<T>::end() const noexcept {
  return ConstIterator{data_ + size_};
}

template <typename T>
constexpr typename Tensor<T>::ConstIterator Tensor<T>::cbegin() const noexcept {
  return begin();
}

template <typename T>
constexpr typename Tensor<T>::ConstIterator Tensor<T>::cend() const noexcept {
  return end();
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

template <typename T, class Alloc, class Alloc2>
constexpr Tensor<T, Alloc>& operator*=(
    Tensor<T, Alloc>& self, const Tensor<T, Alloc2>& other) noexcept {
  assert((self.size() == other.size()));
  for (std::size_t i{0}; i < self.size(); ++i) {
    self[i] *= other[i];
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
