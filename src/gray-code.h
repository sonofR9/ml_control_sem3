#pragma once

#include <cstdint>
#include <iostream>
#include <ostream>

namespace optimization {
constexpr uint64_t to_gray(uint64_t n) {
  return (n ^ (n >> 1));
}

constexpr uint64_t from_gray(uint64_t n) {
  int ish = 1;
  uint64_t ans = n;
  uint64_t idiv = 0;

  for (;;) {
    idiv = (ans >> ish);
    ans ^= idiv;
    if (idiv <= 1 || ish == 16) return ans;
    ish <<= 1;
  }
  return ans;
}

/**
 * @brief
 *
 * @tparam D
 * @tparam Z zero offset
 */
template <uint64_t D, uint64_t Z = 1000>
class DoubleGrayCode {
 public:
  DoubleGrayCode() : DoubleGrayCode(static_cast<double>(Z)) {
  }
  /*implicit*/ DoubleGrayCode(double value) {
    code_ =
        to_gray(static_cast<uint64_t>(D * (value + static_cast<double>(Z))));
  }
  DoubleGrayCode& operator=(double value) {
    code_ =
        to_gray(static_cast<uint64_t>(D * (value + static_cast<double>(Z))));
    return *this;
  }
  // /*implicit*/ DoubleGrayCode(uint64_t value) {
  //   code_ = to_gray(D * (value + zero_));
  // }
  // DoubleGrayCode& operator=(uint64_t value) {
  //   code_ = to_gray(D * (value + zero_));
  // }

  DoubleGrayCode(const DoubleGrayCode& value) = default;
  DoubleGrayCode(DoubleGrayCode&& value) noexcept = default;
  DoubleGrayCode& operator=(const DoubleGrayCode& value) = default;
  DoubleGrayCode& operator=(DoubleGrayCode&& value) noexcept = default;

  [[nodiscard]] uint64_t getGray() const {
    return code_;
  }
  [[nodiscard]] double getDouble() const {
    return static_cast<double>(from_gray(code_)) / D - static_cast<double>(Z);
  }

  void changeBit(int bit) {
    code_ ^= (1 << bit);
  }

  DoubleGrayCode<D>& operator&=(uint64_t rhs) {
    code_ &= rhs;
    return *this;
  }
  DoubleGrayCode<D>& operator|=(uint64_t rhs) {
    code_ |= rhs;
    return *this;
  }

 private:
  uint64_t code_{0};
};

template <uint64_t D>
DoubleGrayCode<D> operator+(const DoubleGrayCode<D>& lhs,
                            const DoubleGrayCode<D>& rhs) {
  return DoubleGrayCode<D>(lhs.getDouble() + rhs.getDouble());
}

template <uint64_t D>
DoubleGrayCode<D> operator-(const DoubleGrayCode<D>& lhs,
                            const DoubleGrayCode<D>& rhs) {
  return DoubleGrayCode<D>(lhs.getDouble() - rhs.getDouble());
}

template <uint64_t D>
DoubleGrayCode<D>& operator+=(DoubleGrayCode<D>& lhs,
                              const DoubleGrayCode<D>& rhs) {
  lhs = lhs + rhs;
  return lhs;
}

template <uint64_t D>
DoubleGrayCode<D>& operator-=(DoubleGrayCode<D>& lhs,
                              const DoubleGrayCode<D>& rhs) {
  lhs = lhs - rhs;
  return lhs;
}

template <uint64_t D>
bool operator==(const DoubleGrayCode<D>& lhs, const DoubleGrayCode<D>& rhs) {
  return lhs.getGray() == rhs.getGray();
}

template <uint64_t D>
bool operator!=(const DoubleGrayCode<D>& lhs, const DoubleGrayCode<D>& rhs) {
  return !(lhs == rhs);
}

template <uint64_t D>
DoubleGrayCode<D> operator+(const DoubleGrayCode<D>& lhs, double rhs) {
  return DoubleGrayCode<D>(lhs.getDouble() + rhs);
}

template <uint64_t D>
DoubleGrayCode<D> operator-(const DoubleGrayCode<D>& lhs, double rhs) {
  return DoubleGrayCode<D>(lhs.getDouble() - rhs);
}

template <uint64_t D>
DoubleGrayCode<D> operator+(double lhs, const DoubleGrayCode<D>& rhs) {
  return rhs + lhs;
}

template <uint64_t D>
DoubleGrayCode<D> operator-(double lhs, const DoubleGrayCode<D>& rhs) {
  return rhs - lhs;
}

template <uint64_t D>
DoubleGrayCode<D>& operator+=(DoubleGrayCode<D>& lhs, double rhs) {
  lhs = lhs + rhs;
  return lhs;
}

template <uint64_t D>
DoubleGrayCode<D>& operator-=(DoubleGrayCode<D>& lhs, double rhs) {
  lhs = lhs - rhs;
  return lhs;
}

template <uint64_t D>
DoubleGrayCode<D> operator+(const DoubleGrayCode<D>& lhs, int rhs) {
  return lhs + static_cast<double>(rhs);
}

template <uint64_t D>
DoubleGrayCode<D> operator-(const DoubleGrayCode<D>& lhs, int rhs) {
  return lhs - static_cast<double>(rhs);
}

template <uint64_t D>
DoubleGrayCode<D> operator+(int lhs, const DoubleGrayCode<D>& rhs) {
  return rhs + lhs;
}

template <uint64_t D>
DoubleGrayCode<D> operator-(int lhs, const DoubleGrayCode<D>& rhs) {
  return rhs - lhs;
}
template <uint64_t D>
DoubleGrayCode<D>& operator+=(DoubleGrayCode<D>& lhs, int rhs) {
  lhs = lhs + rhs;
  return lhs;
}

template <uint64_t D>
DoubleGrayCode<D>& operator-=(DoubleGrayCode<D>& lhs, int rhs) {
  lhs = lhs - rhs;
  return lhs;
}

template <uint64_t D>
DoubleGrayCode<D> operator*(const DoubleGrayCode<D>& lhs,
                            const DoubleGrayCode<D>& rhs) {
  return DoubleGrayCode<D>(lhs.getDouble() * rhs.getDouble());
}

template <uint64_t D>
DoubleGrayCode<D> operator*(const DoubleGrayCode<D>& lhs, double rhs) {
  return DoubleGrayCode<D>(lhs.getDouble() * rhs);
}

template <uint64_t D>
DoubleGrayCode<D> operator*(const DoubleGrayCode<D>& lhs, int rhs) {
  return lhs * static_cast<double>(rhs);
}

template <uint64_t D>
DoubleGrayCode<D> operator*(int lhs, const DoubleGrayCode<D>& rhs) {
  return rhs * static_cast<double>(lhs);
}

template <uint64_t D>
DoubleGrayCode<D> operator*(double lhs, const DoubleGrayCode<D>& rhs) {
  return rhs * lhs;
}

template <uint64_t D>
DoubleGrayCode<D> operator/(const DoubleGrayCode<D>& lhs,
                            const DoubleGrayCode<D>& rhs) {
  return DoubleGrayCode<D>(lhs.getDouble() / rhs.getDouble());
}

template <uint64_t D>
DoubleGrayCode<D> operator/(const DoubleGrayCode<D>& lhs, double rhs) {
  return DoubleGrayCode<D>(lhs.getDouble() / rhs);
}

template <uint64_t D>
DoubleGrayCode<D> operator/(const DoubleGrayCode<D>& lhs, int rhs) {
  return lhs / static_cast<double>(rhs);
}

template <uint64_t D>
DoubleGrayCode<D> operator/(int lhs, const DoubleGrayCode<D>& rhs) {
  return DoubleGrayCode<D>(static_cast<double>(lhs) / rhs.getDouble());
}

template <uint64_t D>
DoubleGrayCode<D> operator/(double lhs, const DoubleGrayCode<D>& rhs) {
  return DoubleGrayCode<D>(lhs / rhs.getDouble());
}
}  // namespace optimization

template <uint64_t D>
std::ostream& operator<<(std::ostream& stream,
                         const optimization::DoubleGrayCode<D>& gray) {
  stream << gray.getDouble();
  return stream;
}
