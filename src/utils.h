#pragma once

#include <cassert>
#include <fstream>
#include <ostream>
#include <ranges>
#include <vector>

namespace optimization {
constexpr double kEps = 1e-10;

template <typename R, typename T>
concept ConvertibleInputRangeTo =
    std::ranges::input_range<R> &&
    std::convertible_to<std::ranges::range_value_t<R>, T>;

template <class Alloc, class VectorAlloc>
void writeTrajectoryToFiles(
    const std::vector<std::vector<double, Alloc>, VectorAlloc>& trajectory) {
  assert((trajectory.size() == 4));

  std::ofstream fileX("trajectory_x.txt");
  std::ofstream fileY("trajectory_y.txt");

  if (!fileX.is_open() || !fileY.is_open()) {
    throw std::runtime_error("Error opening files");
  }

  for (size_t i = 0; i < trajectory[0].size(); ++i) {
    fileX << trajectory[0][i] << "\n";
    fileY << trajectory[1][i] << "\n";
  }

  fileX.close();
  fileY.close();
}

}  // namespace optimization

namespace std {
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  for (auto item : vec) {
    os << item << ",";
  }
  return os;
}
}  // namespace std
