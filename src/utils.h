#pragma once

#include <ostream>
#include <ranges>
#include <vector>

namespace optimization {
template <typename R, typename T>
concept ConvertibleInputRangeTo =
    std::ranges::input_range<R> &&
    std::convertible_to<std::ranges::range_value_t<R>, T>;
}

namespace std {
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  for (auto item : vec) {
    os << item << ",";
  }
  return os;
}
}  // namespace std
