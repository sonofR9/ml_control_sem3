#pragma once

#include <cstdint>
#include <ostream>

namespace optimization {
uint64_t to_gray(uint64_t n) {
  return (n ^ (n >> 1));
}

uint64_t from_gray(uint64_t n) {
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

template <uint64_t D>
class DoubleGrayCode {
  const uint64_t zero_{std::numeric_limits<uint64_t>::max() / D / 2};

 public:
  DoubleGrayCode() = default;
  /*implicit*/ DoubleGrayCode(double value) {
    code_ = to_gray(static_cast<uint64_t>(D * (value + zero_)));
  }
  DoubleGrayCode& operator=(double value) {
    code_ = to_gray(D * (value + zero_));
  }

  uint64_t getGray() const {
    return code_;
  }
  uint64_t getDouble() const {
    return static_cast<double>(from_gray(code_)) / D - zero_;
  }

  void changeBit(int bit) {
    code_ ^= (1 << bit);
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
