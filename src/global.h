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
template <int T>
struct Vector {
  Vector() {
    for (int i{0}; i < T; ++i) data_[i] = 0;
  }
  ~Vector() = default;
  Vector(const Vector&) = default;
  Vector(Vector&&) noexcept = default;
  Vector& operator=(const Vector&) = default;
  Vector& operator=(Vector&&) noexcept = default;

  Vector(std::initializer_list<double> list) {
    if (list.size() != T)
      throw std::length_error("Initializer list size dffers from data size");
    auto el = list.begin();
    for (int i{0}; i < T; ++i) data_[i] = *(el++);
  }

  double& operator[](int i) {
    return data_[i];
  }
  double operator[](int i) const {
    return data_[i];
  }

  class Iterator;

  Iterator begin();

  Iterator end();

 private:
  std::array<double, T> data_;
};

template <int T>
using StatePoint = Vector<T>;

/**
 * @brief represents current derivatives (left-hand side of equations system)
 */
template <int T>
using StateDerivativesPoint = Vector<T>;

template <int T>
Vector<T> operator+(const Vector<T>& self, const Vector<T>& other) {
  Vector<T> result;
  for (int i{0}; i < T; ++i) {
    result[i] = other[i] + self[i];
  }
  return result;
}

template <int T>
Vector<T>& operator+=(Vector<T>& self, const Vector<T>& other) {
  for (int i{0}; i < T; ++i) {
    self[i] += other[i];
  }
  return self;
}

template <int T>
Vector<T> operator-(const Vector<T>& self, const Vector<T>& other) {
  Vector<T> result;
  for (int i{0}; i < T; ++i) {
    result[i] = other[i] - self[i];
  }
  return result;
}

template <int T>
Vector<T>& operator-=(Vector<T>& self, const Vector<T>& other) {
  for (int i{0}; i < T; ++i) {
    self[i] -= other[i];
  }
  return self;
}

template <int T, typename M>
Vector<T> operator*(const Vector<T>& self, M multiplier) {
  Vector<T> result;
  for (int i{0}; i < T; ++i) {
    result[i] = multiplier * self[i];
  }
  return result;
}
template <int T, typename M>
Vector<T> operator*(M multiplier, const Vector<T>& self) {
  return self * multiplier;
}
template <int T>
Vector<T>& operator*=(Vector<T>& self, const Vector<T>& other) {
  for (int i{0}; i < T; ++i) {
    self[i] *= other[i];
  }
  return self;
}

template <int T, typename M>
Vector<T> operator/(const Vector<T>& self, M divider) {
  Vector<T> result;
  for (int i{0}; i < T; ++i) {
    result[i] = self[i] / divider;
  }
  return result;
}

template <int N>
bool operator==(const Vector<N>& lhs, const Vector<N>& rhs) {
  constexpr double eps{1e-3};
  for (int i{0}; i < N; ++i) {
    if (fabs(lhs[i] - rhs[i]) > eps) {
      return false;
    }
  }
  return true;
}

template <int N>
bool operator!=(const Vector<N>& lhs, const Vector<N>& rhs) {
  return !(lhs == rhs);
}

template <int N>
class Vector<N>::Iterator {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = double;
  using difference_type = std::ptrdiff_t;
  using pointer = double*;
  using reference = double&;

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

template <int N>
typename Vector<N>::Iterator Vector<N>::begin() {
  return data_.begin();
}

template <int N>
typename Vector<N>::Iterator Vector<N>::end() {
  return data_.end();
}

template <int N>
double norm(Vector<N> self) {
  return std::sqrt(std::transform_reduce(self.begin(), self.end(), 0.0));
}
// }
// --------------------------------StatePoint end------------------------------
// export {
template <typename F, int T>
concept StateSpaceFunction = requires(F func, Vector<T> point, double time) {
                               {
                                 func(point, time)
                                 } -> std::same_as<StateDerivativesPoint<T>>;
                             };
}  // namespace optimization

template <int T>
std::ostream& operator<<(std::ostream& stream,
                         const optimization::Vector<T>& state) {
  for (int i{0}; i < T; ++i) {
    stream << state[i] << " ";
  }
  return stream;
}
