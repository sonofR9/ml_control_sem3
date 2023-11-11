#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <numeric>
#include <ostream>

namespace optimization {
constexpr double kEps = 1e-10;

// --------------------------------StatePoint start---------------------------
// export {
/**
 * @brief class to represent state of system using T variables
 *
 * @tparam T number of state variables
 */
template <int N, typename T = double>
struct Vector {
  Vector() {
    for (int i{0}; i < N; ++i) data_[i] = 0;
  }
  ~Vector() = default;
  Vector(const Vector&) = default;
  Vector(Vector&&) noexcept = default;
  Vector& operator=(const Vector&) = default;
  Vector& operator=(Vector&&) noexcept = default;

  Vector(std::initializer_list<T> list) {
    if (list.size() != N)
      throw std::length_error("Initializer list size dffers from data size");
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

  Iterator begin();

  Iterator end();

  [[nodiscard]] static constexpr int size() noexcept {
    return N;
  }

 private:
  std::array<T, N> data_;
};

template <int N>
using StatePoint = Vector<N>;

/**
 * @brief represents current derivatives (left-hand side of equations system)
 */
template <int N>
using StateDerivativesPoint = Vector<N>;

template <int N, typename T>
Vector<N, T> operator+(const Vector<N, T>& self, const Vector<N, T>& other) {
  Vector<N, T> result;
  for (int i{0}; i < N; ++i) {
    result[i] = other[i] + self[i];
  }
  return result;
}

template <int N, typename T>
Vector<N, T>& operator+=(Vector<N, T>& self, const Vector<N, T>& other) {
  for (int i{0}; i < N; ++i) {
    self[i] += other[i];
  }
  return self;
}

template <int N, typename T>
Vector<N, T> operator-(const Vector<N, T>& self, const Vector<N, T>& other) {
  Vector<N, T> result;
  for (int i{0}; i < N; ++i) {
    result[i] = other[i] - self[i];
  }
  return result;
}

template <int N, typename T>
Vector<N, T>& operator-=(Vector<N, T>& self, const Vector<N, T>& other) {
  for (int i{0}; i < N; ++i) {
    self[i] -= other[i];
  }
  return self;
}

template <int N, typename T, typename M>
Vector<N, T> operator*(const Vector<N, T>& self, M multiplier) {
  Vector<N, T> result;
  // for (int i{0}; i < N; ++i) {
  // result[i] = multiplier * self[i];
  // }
  std::transform(self.begin(), self.end(), result.begin(),
                 [&multiplier](const T& val) -> T { return val * multiplier; });
  return result;
}
template <int N, typename T, typename M>
Vector<N, T> operator*(M multiplier, const Vector<N, T>& self) {
  return self * multiplier;
}
template <int N, typename T>
Vector<N, T>& operator*=(Vector<N, T>& self, const Vector<N, T>& other) {
  for (int i{0}; i < N; ++i) {
    self[i] *= other[i];
  }
  return self;
}

template <int N, typename T, typename M>
Vector<N, T> operator/(const Vector<N, T>& self, M divider) {
  Vector<N, T> result;
  for (int i{0}; i < N; ++i) {
    result[i] = self[i] / divider;
  }
  return result;
}

template <int N, typename T>
bool operator==(const Vector<N, T>& lhs, const Vector<N, T>& rhs) {
  constexpr double eps{1e-3};
  const auto& diff{lhs - rhs};
  return !std::any_of(diff.begin(), diff.end(),
                      [eps](const T& value) { return fabs(value) > eps; });
}

template <int N, typename T>
bool operator!=(const Vector<N, T>& lhs, const Vector<N, T>& rhs) {
  return !(lhs == rhs);
}

template <int N, typename T>
class Vector<N, T>::Iterator {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  explicit Iterator(pointer ptr) : ptr_(ptr) {
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

  bool operator==(const Iterator& other) const {
    return ptr_ == other.ptr_;
  }

  bool operator!=(const Iterator& other) const {
    return ptr_ != other.ptr_;
  }

  bool operator<(const Iterator& other) const {
    return ptr_ < other.ptr_;
  }

  bool operator>(const Iterator& other) const {
    return ptr_ > other.ptr_;
  }

  bool operator<=(const Iterator& other) const {
    return ptr_ <= other.ptr_;
  }

  bool operator>=(const Iterator& other) const {
    return ptr_ >= other.ptr_;
  }

 private:
  pointer ptr_;
};

template <int N, typename T>
typename Vector<N, T>::Iterator Vector<N, T>::begin() {
  return data_.begin();
}

template <int N, typename T>
typename Vector<N, T>::Iterator Vector<N, T>::end() {
  return data_.end();
}

template <int N, typename T>
double norm(Vector<N, T> self) {
  return std::sqrt(std::transform_reduce(self.begin(), self.end(), 0.0));
}
// }
// --------------------------------StatePoint end------------------------------
// export {
template <typename F, int N>
concept StateSpaceFunction = requires(F func, Vector<N> point, double time) {
                               {
                                 func(point, time)
                                 } -> std::same_as<StateDerivativesPoint<N>>;
                             };
}  // namespace optimization

template <int N, typename T>
std::ostream& operator<<(std::ostream& stream,
                         const optimization::Vector<N, T>& state) {
  for (int i{0}; i < N; ++i) {
    stream << state[i] << " ";
  }
  return stream;
}
