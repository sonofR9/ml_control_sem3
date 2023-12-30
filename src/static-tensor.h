#pragma once

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <functional>
#include <numeric>
#include <ostream>
#include <random>
#include <ranges>
#include <sstream>

namespace optimization {
constexpr double kEps = 1e-10;

/**
 * @brief class to represent state of system using T variables
 *
 * @tparam T number of state variables
 */
template <int N, typename T = double>
struct StaticTensor {
  StaticTensor(std::size_t size) {
    if constexpr (std::is_convertible_v<int, T>) {
      for (int i{0}; i < size; ++i) data_[i] = 0;
    }
  }
  ~StaticTensor() = default;
  StaticTensor(const StaticTensor&) = default;
  StaticTensor(StaticTensor&&) noexcept = default;
  StaticTensor& operator=(const StaticTensor&) = default;
  StaticTensor& operator=(StaticTensor&&) noexcept = default;

  StaticTensor(std::initializer_list<T> list) {
    if (list.size() != N)
      throw std::length_error("Initializer list size differs from data size");
    auto el = list.begin();
    for (int i{0}; i < N; ++i) data_[i] = *(el++);
  }

  T& operator[](int i) noexcept {
    return data_[i];
  }
  T operator[](int i) const noexcept {
    return data_[i];
  }

  class Iterator;
  class ConstIterator;

  Iterator begin();
  Iterator end();
  ConstIterator cbegin() const;
  ConstIterator cend() const;

  [[nodiscard]] static constexpr int size() noexcept {
    return N;
  }

 private:
  std::array<T, N> data_;
};

template <int N>
using StatePoint = StaticTensor<N>;

/**
 * @brief represents current derivatives (left-hand side of equations system)
 */
template <int N>
using StateDerivativesPoint = StaticTensor<N>;

template <int N, typename T>
StaticTensor<N, T> operator+(const StaticTensor<N, T>& self,
                             const StaticTensor<N, T>& other) {
  StaticTensor<N, T> result;
  for (int i{0}; i < N; ++i) {
    result[i] = other[i] + self[i];
  }
  return result;
}

template <int N, typename T>
StaticTensor<N, T>& operator+=(StaticTensor<N, T>& self,
                               const StaticTensor<N, T>& other) {
  for (int i{0}; i < N; ++i) {
    self[i] += other[i];
  }
  return self;
}

template <int N, typename T>
StaticTensor<N, T> operator-(const StaticTensor<N, T>& self,
                             const StaticTensor<N, T>& other) {
  StaticTensor<N, T> result;
  for (int i{0}; i < N; ++i) {
    result[i] = self[i] - other[i];
  }
  return result;
}

template <int N, typename T>
StaticTensor<N, T> operator-(const StaticTensor<N, T>& self) {
  StaticTensor<N, T> result;
  for (int i{0}; i < N; ++i) {
    result[i] = -self[i];
  }
  return result;
}

template <int N, typename T>
StaticTensor<N, T>& operator-=(StaticTensor<N, T>& self,
                               const StaticTensor<N, T>& other) {
  for (int i{0}; i < N; ++i) {
    self[i] -= other[i];
  }
  return self;
}

template <int N, typename T, typename M>
StaticTensor<N, T> operator*(const StaticTensor<N, T>& self, M multiplier) {
  StaticTensor<N, T> result;
  for (int i{0}; i < N; ++i) {
    result[i] = multiplier * self[i];
  }
  return result;
}
template <int N, typename T, typename M>
StaticTensor<N, T> operator*(M multiplier, const StaticTensor<N, T>& self) {
  return self * multiplier;
}
template <int N, typename T>
StaticTensor<N, T>& operator*=(StaticTensor<N, T>& self,
                               const StaticTensor<N, T>& other) {
  for (int i{0}; i < N; ++i) {
    self[i] *= other[i];
  }
  return self;
}

template <int N, typename T, typename M>
StaticTensor<N, T> operator/(const StaticTensor<N, T>& self, M divider) {
  StaticTensor<N, T> result;
  for (int i{0}; i < N; ++i) {
    result[i] = self[i] / divider;
  }
  return result;
}

template <int N, typename T>
bool operator==(const StaticTensor<N, T>& lhs, const StaticTensor<N, T>& rhs) {
  constexpr double eps{1e-3};
  auto diff{lhs - rhs};
  return !std::any_of(diff.begin(), diff.end(),
                      [eps](const T& value) { return fabs(value) > eps; });
}

template <int N, typename T>
bool operator!=(const StaticTensor<N, T>& lhs, const StaticTensor<N, T>& rhs) {
  return !(lhs == rhs);
}

// ------------------------------ iterators -----------------------------------
template <int N, typename T>
class StaticTensor<N, T>::Iterator {
 public:
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  /*implicit*/ Iterator(pointer ptr) : ptr_(ptr) {
  }

  reference operator*() const {
    return *ptr_;
  }
  pointer operator->() const {
    return ptr_;
  }

  Iterator& operator++() {
    ++ptr_;
    return *this;
  }

  Iterator operator++(int) {
    Iterator temp = *this;
    ++ptr_;
    return temp;
  }

  Iterator& operator--() {
    --ptr_;
    return *this;
  }

  Iterator operator--(int) {
    Iterator temp = *this;
    --ptr_;
    return temp;
  }

  Iterator operator+(difference_type n) const {
    return Iterator(ptr_ + n);
  }

  Iterator operator-(difference_type n) const {
    return Iterator(ptr_ - n);
  }

  difference_type operator-(const Iterator& other) const {
    return ptr_ - other.ptr_;
  }

  Iterator& operator+=(difference_type n) {
    ptr_ += n;
    return *this;
  }

  Iterator& operator-=(difference_type n) {
    ptr_ -= n;
    return *this;
  }

  friend auto operator<=>(const Iterator&, const Iterator&) = default;

 private:
  pointer ptr_;
};

template <int N, typename T>
class StaticTensor<N, T>::ConstIterator {
 public:
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = const T*;
  using reference = const T&;

  /*implicit*/ ConstIterator(pointer ptr) : ptr_(ptr) {
  }

  reference operator*() const {
    return *ptr_;
  }
  pointer operator->() const {
    return ptr_;
  }

  ConstIterator operator++() {
    ++ptr_;
    return *this;
  }

  ConstIterator operator++(int) {
    Iterator temp = *this;
    ++ptr_;
    return temp;
  }

  ConstIterator operator--() {
    --ptr_;
    return *this;
  }

  ConstIterator operator--(int) {
    Iterator temp = *this;
    --ptr_;
    return temp;
  }

  ConstIterator operator+(difference_type n) const {
    return ConstIterator(ptr_ + n);
  }

  ConstIterator operator-(difference_type n) const {
    return ConstIterator(ptr_ - n);
  }

  difference_type operator-(const ConstIterator& other) const {
    return ptr_ - other.ptr_;
  }

  ConstIterator operator+=(difference_type n) {
    ptr_ += n;
    return *this;
  }

  ConstIterator operator-=(difference_type n) {
    ptr_ -= n;
    return *this;
  }

  friend auto operator<=>(const ConstIterator&, const ConstIterator&) = default;

 private:
  pointer ptr_;
};

template <int N, typename T>
typename StaticTensor<N, T>::Iterator StaticTensor<N, T>::begin() {
  return {&data_[0]};
}

template <int N, typename T>
typename StaticTensor<N, T>::Iterator StaticTensor<N, T>::end() {
  return {&data_[N - 1] + 1};
}

template <int N, typename T>
typename StaticTensor<N, T>::ConstIterator StaticTensor<N, T>::cbegin() const {
  return {&data_[0]};
}

template <int N, typename T>
typename StaticTensor<N, T>::ConstIterator StaticTensor<N, T>::cend() const {
  return {&data_[N - 1] + 1};
}

template <int N, typename T>
double norm(StaticTensor<N, T> self) {
  return std::sqrt(
      std::transform_reduce(self.begin(), self.end(), 0.0, std::plus<>(),
                            [](const T& val) { return val * val; }));
}

double norm(StaticTensor<100, double> self) {
  return std::sqrt(std::transform_reduce(self.begin(), self.end(), 0.0,
                                         std::plus{},
                                         [](double val) { return val * val; }));
}

extern int seed;
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
